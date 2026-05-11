import os
import torch
import numpy as np
from collections import OrderedDict
from lietorch import SE3

from src.modules import WildPoseNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.backend import Backend
from src.utils.datasets import Sintel

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)
        if self.cfg['tracking']['mask_detector']:
            self.net: WildPoseNet = WildPoseNet(motion_head_type=self.cfg['tracking']['mask_head_type'])
        else:
            self.net: WildPoseNet = WildPoseNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.net.to(self.device).eval()
        self.net.share_memory()

        self.video = DepthVideo(cfg, self.printer)
        self.ba = Backend(self.net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.stream = stream

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        loaded = torch.load(droid_pretrained)
        if "model_state_dict" in loaded.keys():
            loaded = loaded["model_state_dict"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in loaded.items()
            ]
        )

        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if not missing:            # ― no keys are missing
            print("All keys loaded.")
        else:
            print("*"*20,":")
            print(missing)
            print("*"*20)

        self.net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self):
        self.tracker = Tracker(self)
        self.printer.print("Tracking initialized!", FontColor.TRACKER)

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)

        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print("Tracking Done!", FontColor.TRACKER)

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        self.ba = Backend(self.net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""
        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, Sintel):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj/before_final_ba",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, Sintel):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        traj_est_inv = self.traj_filler(self.stream)
        traj_est_lietorch = traj_est_inv.inv()
        traj_est = traj_est_lietorch.matrix().data.cpu().numpy()

        if not self.cfg['tracking']['full_ba']:
            kf_num = self.video.counter.value
            kf_timestamps = self.video.timestamp[:kf_num].cpu().int().numpy()
            kf_poses = SE3(self.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
            traj_est[kf_timestamps] = kf_poses
            traj_est_not_align = traj_est.copy()
        else:
            self.ba = Backend(self.net, self.video, self.cfg)
            print("current kfs:", self.video.timestamp[:self.video.counter.value])
            torch.cuda.empty_cache()
            n, n_edges, edge_i, edge_j = self.ba.dense_ba(15)
            np.savez(f"{self.save_dir}/final_ba_edges.npz",edge_i=edge_i, edge_j=edge_j)
            self.printer.print("Full BA Done!", FontColor.TRACKER)

            traj_est_not_align = SE3(
                self.video.poses[: self.video.counter.value].clone()
            ).inv().matrix().data.cpu().numpy()
            
        full_traj_eval(
            traj_est_not_align,
            f"{self.save_dir}/traj",
            "full_traj",
            self.stream,
            self.logger,
            self.printer,
        )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def run(self):
        self.tracking()
        self.terminate()

        self.printer.terminate()