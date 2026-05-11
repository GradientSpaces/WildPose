import numpy as np
import torch
from typing import Dict

from moge.model.v2 import MoGeModel


def get_metric_depth_estimator(cfg: Dict) -> torch.nn.Module:
    """
    Get the metric depth estimator model based on the configuration.

    Args:
        cfg (Dict): Configuration dictionary.

    Returns:
        torch.nn.Module: The metric depth estimator model.
    """
    device = cfg["device"]
    depth_model = cfg["mono_prior"]["depth"]

    if "moge-2" in depth_model:
        model = MoGeModel.from_pretrained(f"Ruicheng/{depth_model}")
    else:
        # If use other metric depth estimator as prior, write the code here
        raise NotImplementedError("Unsupported depth model")
    return model.to(device).eval()


@torch.no_grad()
def predict_metric_depth(
    model: torch.nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_depth: bool = True,
) -> torch.Tensor:
    """
    Predict metric depth using the given model.

    Args:
        model (torch.nn.Module): The depth estimation model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_depth (bool): Whether to save the depth map.

    Returns:
        torch.Tensor: Predicted depth map.
    """
    depth_model = cfg["mono_prior"]["depth"]
    if "moge-2" in depth_model:
        output = model.infer(input_tensor)
        output = output['depth'].squeeze(0).clone().to(device)
    else:
        # If use other metric depth estimator as prior, write the code here
        raise NotImplementedError("Unsupported depth model")
    output[torch.isinf(output)] = 0
    output[torch.isnan(output)] = 0

    if save_depth:
        _save_depth_map(output, cfg, idx)

    return output

def _save_depth_map(depth_map: torch.Tensor, cfg: Dict, idx: int) -> None:
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/depths/{idx:05d}.npy"
    final_depth = depth_map.detach().cpu().float().numpy()
    np.save(output_path, final_depth)
