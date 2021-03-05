from functools import partial
from math import sqrt
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.helpers import load_pretrained
from timm.models.layers import to_2tuple as _pair
from timm.models.vision_transformer import PatchEmbed

from .build import BACKBONE_REGISTRY

def resize_pos_embed(pos_embed_state, model):
    ntok = model.pos_embed.shape[1] - 1
    pos_embed_token = pos_embed_state[:, :1]
    pos_embed_grid = pos_embed_state[0, 1:]

    ih, iw = model.patch_embed.img_size
    ph, pw = model.patch_embed.patch_size
    gh_new, gw_new = ih // ph, iw // pw
    gh_old = gw_old = int(sqrt(len(pos_embed_grid)))

    pos_embed_grid = pos_embed_grid.reshape(1, gh_old, gw_old, -1)
    pos_embed_grid = pos_embed_grid.permute(0, 3, 1, 2)
    pos_embed_grid = F.interpolate(
        pos_embed_grid, size=(gh_new, gw_new), mode="bilinear"
    )
    pos_embed_grid = pos_embed_grid.permute(0, 2, 3, 1)
    pos_embed_grid = pos_embed_grid.reshape(1, gh_new * gw_new, -1)
    pos_embed = torch.cat([pos_embed_token, pos_embed_grid], dim=1)

    return pos_embed


def filter_fn(state_dict, model):
    if "patch_embed.proj.weight" in state_dict:
        v = state_dict["patch_embed.proj.weight"]
        if len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        state_dict["patch_embed.proj.weight"] = v

    if "pos_embed" in state_dict:
        v = state_dict["pos_embed"]
        if v.shape != model.pos_embed.shape:
            assert isinstance(
                model.patch_embed, PatchEmbed
            ), "Must be PatchEmbed"
            v = resize_pos_embed(v, model)
        state_dict["pos_embed"] = v

    return state_dict


class VisionTransformer(nn.Module):
    def __init__(
        self,
        name,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()

        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        kwargs["img_size"] = self.img_size
        kwargs["pretrained"] = False
        kwargs["num_classes"] = 0
        self.base = timm.create_model(name, **kwargs)

        if isinstance(self.base.patch_embed, PatchEmbed):
            assert self.base.patch_embed.patch_size == self.patch_size

        self.num_features = self.base.num_features
        self.num_patches = self.base.patch_embed.num_patches

        if pretrained:
            load_pretrained(
                self.base,
                num_classes=0,
                in_chans=kwargs.get("in_chans", 3),
                filter_fn=partial(filter_fn, model=self.base),
            )

    def forward(self, x):
        B = x.shape[0]

        x = self.base.patch_embed(x)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.base.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.base.pos_embed
        x = self.base.pos_drop(x)

        for blk in self.base.blocks:
            x = blk(x)

        return x

    
@BACKBONE_REGISTRY.register()
def build_transformer_backbone(cfg):
    name = cfg.MODEL.BACKBONE.DEPTH
    kwargs = {
        "pretrained": cfg.MODEL.BACKBONE.PRETRAIN,
        "img_size": cfg.INPUT.SIZE_TRAIN,
        "patch_size": 32 if "patch32" in name else 16
    }

    return VisionTransformer(name, **kwargs)