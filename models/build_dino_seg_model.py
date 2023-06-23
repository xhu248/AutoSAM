import torch

from functools import partial

from .SamFeatSeg import DinoViTSeg, SegDecoderCNN
from .dinov2_layers.vision_transformer import vit_small, vit_base, vit_large

vit_kwargs = dict(
    img_size=256,
    patch_size=16,
)

def _build_seg_model(
    image_encoder,
    checkpoint=None,
):
    dino_seg = DinoViTSeg(
        image_encoder=image_encoder,
        seg_decoder=SegDecoderCNN(),
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        loaded_keys = []
        for k in state_dict.keys():
            if k in dino_seg.image_encoder.state_dict().keys():
                loaded_keys.append(k)
        dino_seg.image_encoder.load_state_dict(state_dict, strict=False)
        print("loaded keys:", loaded_keys)

    return dino_seg


def build_dino_vit_s_seg_cnn(checkpoint=None):
    image_encoder = vit_small(**vit_kwargs)
    return _build_seg_model(
        image_encoder=image_encoder,
        checkpoint=checkpoint,
    )


build_dino_seg = build_dino_vit_s_seg_cnn


def build_dino_vit_b_seg_cnn(checkpoint=None):
    image_encoder = vit_base(**vit_kwargs)
    return _build_seg_model(
        image_encoder=image_encoder,
        checkpoint=checkpoint,
    )


def build_dino_vit_l_seg_cnn(checkpoint=None):
    image_encoder = vit_large(**vit_kwargs)
    return _build_seg_model(
        image_encoder=image_encoder,
        checkpoint=checkpoint,
    )


dino_seg_model_registry = {
    "default": build_dino_seg,
    "dino_vit_s": build_dino_seg,
    "dino_vit_b": build_dino_vit_b_seg_cnn,
    "dino_vit_l": build_dino_vit_l_seg_cnn,
}

