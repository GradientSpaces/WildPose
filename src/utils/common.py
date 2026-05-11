# Copyright 2024 The Splat-SLAM Authors.
# Licensed under the Apache License, Version 2.0
# available at: https://github.com/google-research/Splat-SLAM/blob/main/LICENSE

import numpy as np
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = torch.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def update_cam(cfg):
    """
    Update the camera intrinsics according to the pre-processing config,
    such as resize or edge crop
    """
    # resize the input images to crop_size(variable name used in lietorch)
    H, W = cfg["cam"]["H"], cfg["cam"]["W"]
    fx, fy = cfg["cam"]["fx"], cfg["cam"]["fy"]
    cx, cy = cfg["cam"]["cx"], cfg["cam"]["cy"]

    h_edge, w_edge = cfg["cam"]["H_edge"], cfg["cam"]["W_edge"]
    H_out, W_out = cfg["cam"]["H_out"], cfg["cam"]["W_out"]

    fx = fx * (W_out + w_edge * 2) / W
    fy = fy * (H_out + h_edge * 2) / H
    cx = cx * (W_out + w_edge * 2) / W
    cy = cy * (H_out + h_edge * 2) / H
    H, W = H_out, W_out

    cx = cx - w_edge
    cy = cy - h_edge
    return H, W, fx, fy, cx, cy