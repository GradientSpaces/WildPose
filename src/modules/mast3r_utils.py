import torch

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R


def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "pretrained/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model

@torch.inference_mode()
def mast3r_extractor_feat(model, img, shape):
    feat, pos, _ = model._encode_image(
        img, shape
    )

    return feat, pos