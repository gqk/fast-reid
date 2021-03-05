# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from typing import Union, Tuple, Optional
from math import ceil
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple as _pair
from timm.models.vision_transformer import Mlp

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

class OverlapResize(nn.Module):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__()

        ih, iw = self.input_size = _pair(input_size)
        ph, pw = self.patch_size = _pair(patch_size)
        sh, sw = self.stride = _pair(stride or patch_size)

        self.output_size = self.input_size
        self.padding = (0, 0)
        if (ph, pw) != (sh, sw):
            oh = ceil((ih + sh - ph) / sh) * ph
            ow = ceil((iw + sw - pw) / sw) * pw
            self.output_size = (oh, ow)

            pdh = ((sh - (ih + sh - ph) % sh)) % sh // 2
            pdw = ((sh - (iw + sw - pw) % sw)) % sw // 2
            self.padding = (pdh, pdw)

    def forward(self, x):
        B, C, H, W = x.shape
        ih, iw = self.input_size
        oh, ow = self.output_size

        assert (H, W) == (ih, iw)

        if (H, W) != (oh, ow):
            x = F.unfold(
                x,
                kernel_size=self.patch_size,
                stride=self.stride,
                padding=self.padding,
            )
            x = F.fold(
                x,
                self.output_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

        return x


@META_ARCH_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # overlap
        patch_size = 32 if "patch32" in cfg.MODEL.BACKBONE.DEPTH else 16
        self.overlap = OverlapResize(
            cfg.INPUT.SIZE_TRAIN, 
            patch_size=patch_size, 
            stride=patch_size * 3 // 4
        )

        # input size replacing
        cfg_tmp = deepcopy(cfg)
        cfg_tmp.defrost()
        cfg_tmp.INPUT.SIZE_TRAIN = self.overlap.output_size
        cfg_tmp.freeze()

        # backbone
        self.backbone = build_backbone(cfg_tmp)
        self.local_feat = nn.Sequential(
            Mlp(self.backbone.num_patches, self.backbone.num_patches // 4, 1),
            nn.Flatten(1)
        )

        # head
        cfg_tmp.defrost()
        cfg_tmp.MODEL.BACKBONE.FEAT_DIM = self.backbone.num_features * 2
        cfg_tmp.freeze() 
        self.heads = build_heads(cfg_tmp)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        images = self.overlap(images)
        features = self.backbone(images)
        fg, fl = features[:,0], self.local_feat(features[:,1:].permute(0, 2, 1))
        features = torch.cat((fg, fl), dim=1).unsqueeze(-1).unsqueeze(-1)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE

        return loss_dict
