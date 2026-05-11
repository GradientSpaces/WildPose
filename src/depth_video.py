import numpy as np
import torch
import lietorch
import droid_backends
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F

from src.modules import cvx_upsample, mast3r_extractor_feat
import src.geom.projective_ops as pops
from src.utils.Printer import FontColor
                
class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        self.printer = printer
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.device = cfg['device']
        self.down_scale = 8
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # To save gpu ram, we put images to cpu since it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.ones = torch.ones_like(self.zeros)
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        # self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.mask_cache = {}

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1].cpu()

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            if item[4].dim() == 2:
                mono_depth = item[4][self.slice_h,self.slice_w]
            else:
                assert item[4].dim() ==3
                mono_depth = item[4][:,self.slice_h,self.slice_w]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            # self.mono_disps_up[index] = torch.where(item[4]>0, 1.0/item[4], 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1,
           motion_only=False):

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()

            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False, self.zeros)
            else:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False, self.ones)
            self.disps.clamp_(min=1e-5)

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            est_disp = self.disps_up[index].clone().to(device)  # [h, w]
            est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
            depth_mask = self.valid_depth_mask[index].clone().to(device)
            c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,
                 poses=poses,
                 depths=depths,
                 timestamps=timestamps,
                 valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)