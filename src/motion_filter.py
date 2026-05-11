import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules import CorrBlock, mast3r_extractor_feat, WildPoseNet
from src.utils.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net: WildPoseNet, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update
        self.feature_extractor = net.feature_extractor

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.5, 0.5, 0.5], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.5, 0.5, 0.5], device=self.device)[:, None, None]
        
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.metric_depth_estimator = get_metric_depth_estimator(cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        B,K,_,H,W = image.shape
        img_shape = torch.tensor((H,W), device=image.device)
        feat, _ = mast3r_extractor_feat(self.feature_extractor, image.squeeze(0), img_shape)
        feat = feat.reshape(K,H//16,W//16,1024).permute(0,3,1,2).contiguous()

        net, inp = self.cnet(feat[None]).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        B,K,_,H,W = image.shape
        img_shape = torch.tensor((H,W), device=image.device)
        feat, _ = mast3r_extractor_feat(self.feature_extractor, image.squeeze(0), img_shape)
        feat = feat.reshape(K,H//16,W//16,1024).permute(0,3,1,2).contiguous()
        return self.fnet(feat[None]).squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None, is_last=False):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device).clone()
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        force_to_add_keyframe = is_last

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0])
        ### only add new frame if there is enough motion ###
        else:
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                timestamp_force = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']
                force_to_add_keyframe = force_to_add_keyframe or timestamp_force

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device)
                
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0])
            else:
                self.count += 1

        return force_to_add_keyframe
