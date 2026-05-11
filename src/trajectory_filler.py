import torch
import lietorch
from lietorch import SE3
from tqdm import tqdm
from src.factor_graph import FactorGraph
from src.utils.datasets import BaseDataset
from src.utils.Printer import FontColor
from src.modules import mast3r_extractor_feat, WildPoseNet
from src.utils.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth

class PoseTrajectoryFiller:
    """ This class is used to fill in non-keyframe poses 
        mainly inherited from DROID-SLAM
    """
    def __init__(self, cfg, net: WildPoseNet, video, printer, device='cuda:0'):
        self.cfg = cfg

        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        self.feature_extractor = net.feature_extractor
        if self.cfg['tracking']['mask_detector']:
            self.motion_mask_detector = net.motion_mask_detector
        else:
            self.motion_mask_detector = None

        self.count = 0
        self.video = video
        self.device = device
        self.printer = printer

        # mean, std for image normalization
        self.MEAN = torch.tensor([0.5, 0.5, 0.5], device=device)[:, None, None]
        self.STDV = torch.tensor([0.5, 0.5, 0.5], device=device)[:, None, None]

        self.metric_depth_estimator = get_metric_depth_estimator(cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        B,K,_,H,W = image.shape
        img_shape = torch.tensor((H,W), device=image.device)
        feat, _ = mast3r_extractor_feat(self.feature_extractor, image.reshape(B*K,3,H,W), img_shape)
        feat = feat.reshape(B,K,H//16,W//16,1024).permute(0,1,4,2,3).contiguous()

        net, inp = self.cnet(feat).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        B,K,_,H,W = image.shape
        img_shape = torch.tensor((H,W), device=image.device)
        feat, _ = mast3r_extractor_feat(self.feature_extractor, image.reshape(B*K,3,H,W), img_shape)
        feat = feat.reshape(B,K,H//16,W//16,1024).permute(0,1,4,2,3).contiguous()
        return self.fnet(feat).squeeze(0)

    def __fill(self, timestamps, images, depths, intrinsics):
        """ fill operator """
        tt = torch.tensor(timestamps, device=self.device)
        images = torch.stack(images, dim=0)
        if depths is not None:
            depths = torch.stack(depths, dim=0)
        intrinsics = torch.stack(intrinsics, 0)
        inputs = images.to(self.device)

        ### linear pose interpolation ###
        N = self.video.counter.value
        M = len(timestamps)

        ts = self.video.timestamp[:N]
        Ps = SE3(self.video.poses[:N])

        # found the location of current timestamp in keyframe queue
        t0 = torch.tensor([ts[ts<=t].shape[0] - 1 for t in timestamps])
        t1 = torch.where(t0 < N-1, t0+1, t0)

        # time interval between nearby keyframes
        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()

        v = dP.log() / dt.unsqueeze(dim=-1)
        w = v * (tt - ts[t0]).unsqueeze(dim=-1)
        Gs = SE3.exp(w) * Ps[t0]

        # extract features (no need for context features)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        fmap = self.__feature_encoder(inputs)

        # temporally put the non-keyframe at the end of keyframe queue
        self.video.counter.value += M
        self.video[N:N+M] = (tt, images[:, 0], Gs.data, 1, depths, intrinsics / float(self.video.down_scale), fmap, None, None)

        graph = FactorGraph(self.video, self.update, self.cfg,
                            motion_mask_detector=self.motion_mask_detector,
                            feature_extractor=self.feature_extractor)
        # build edge between current frame and nearby keyframes for optimization
        graph.add_factors(t0.cuda(), torch.arange(N, N+M).cuda())
        graph.add_factors(t1.cuda(), torch.arange(N, N+M).cuda())

        for _ in range(12):
            graph.update(N, N+M, motion_only=True)

        Gs = SE3(self.video.poses[N:N+M].clone())
        self.video.counter.value -= M

        # clean the decoding cache
        if self.cfg['tracking']['mask_detector']:
            for ix in range(N, N+M):
                self.video.mask_cache = {
                    key: value
                    for key, value in self.video.mask_cache.items()
                    if ix not in key
                }

        return [Gs]
    
    def _reset_full(self, timestamps, images, intrinsics, metric_depths, poses):
        """ reset the video with full frames (prepare for full BA) """
        N = len(timestamps)
        self.video.counter.value = N
        tt = torch.tensor(timestamps, device=self.device)
        self.video.poses[:N] = poses.data
        images = torch.stack(images, dim=0)
        metric_depths = torch.stack(metric_depths, dim=0)
        metric_depths = metric_depths[:, self.video.slice_h, self.video.slice_w]
        intrinsics = torch.stack(intrinsics, dim=0)

        inputs = images.to(self.device).clone()
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        fmap = self.__feature_encoder(inputs)
        net, inp = self.__context_encoder(inputs)

        self.video[0 : N] = (
            tt,
            images[:,0],
            poses.data,
            torch.where(metric_depths>0, 1.0/metric_depths, 0),
            None,
            intrinsics / float(self.video.down_scale),
            fmap,
            net[:,0],
            inp[:,0],
        )

        self.video.mask_cache = {}

    @torch.no_grad()
    def __call__(self, image_stream:BaseDataset):
        """ fill in poses of non-keyframe images. """

        # store all camera poses
        pose_list = []

        timestamps = []
        images = []
        intrinsics = []

        full_tstamps = []
        full_images = []
        full_intrinsics = []
        full_metric_depths = []


        self.printer.print("Filling full trajectory ...",FontColor.INFO)
        intrinsic = image_stream.get_intrinsic()
        for (timestamp, image, _ , _)  in tqdm(image_stream):
            timestamps.append(timestamp)
            images.append(image)
            intrinsics.append(intrinsic)
            if self.cfg['tracking']['full_ba']:
                mono_depth = predict_metric_depth(
                    self.metric_depth_estimator,timestamp,image,self.cfg,self.device,save_depth=False
                )
                full_metric_depths.append(mono_depth)

            if len(timestamps) == 16:
                pose_list += self.__fill(timestamps, images, None, intrinsics)
                full_tstamps += timestamps
                full_images += images
                full_intrinsics += intrinsics
                timestamps, images, intrinsics = [], [], []

        if len(timestamps) > 0:
            pose_list += self.__fill(timestamps, images, None, intrinsics)
            full_tstamps += timestamps
            full_images += images
            full_intrinsics += intrinsics

        if self.cfg['tracking']['full_ba']:
            self._reset_full(full_tstamps, full_images, full_intrinsics, full_metric_depths, lietorch.cat(pose_list, dim=0))

        # stitch pose segments together
        return lietorch.cat(pose_list, dim=0)