"""Load the upstream DINOv2 and LoftUp checkpoints without network access."""

from pathlib import Path
import sys

import torch

from loftup.upsamplers.layers import ChannelNorm
from loftup.upsamplers.upsamplers import LoftUp, UpsamplerwithChannelNorm


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "vendor" / "weights"
DINO_SOURCE = ROOT / "vendor" / "dinov2"


def load_dino_vits14():
    weights = WEIGHTS / "dinov2_vits14_pretrain.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing DINOv2 weights: {weights}")
    if not (DINO_SOURCE / "dinov2" / "models" / "vision_transformer.py").is_file():
        raise FileNotFoundError(f"Missing DINOv2 source: {DINO_SOURCE}")
    if str(DINO_SOURCE) not in sys.path:
        sys.path.insert(0, str(DINO_SOURCE))
    from dinov2.models.vision_transformer import vit_small

    model = vit_small(
        img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
        block_chunks=0, num_register_tokens=0, interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model


def load_loftup_dinov2s():
    weights = WEIGHTS / "loftup_dinov2s_pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing LoftUp weights: {weights}")
    model = UpsamplerwithChannelNorm(
        LoftUp(384, lr_pe_type="sine", lr_size=16), ChannelNorm(384)
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model
