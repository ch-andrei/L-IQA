from collections import OrderedDict

import torch
from torch import nn

from iqa_metrics.vtamiq.VisionTransformer.transformer import VIT_VARIANT_B16


def get_vtamiq_default_params():
    # default ViT backbone parameters;
    # DO NOT MODIFY

    vit_config = OrderedDict(
        variant=VIT_VARIANT_B16,
        use_cls_token=True,
        pretrained=True,
        num_keep_layers=6,
        num_adapters=0,
        num_scales=1,
        num_extra_tokens=8,
        use_layer_scale=True,
        path_drop_prob=0.1,
    )

    vtamiq_config = OrderedDict(
        vit_config=vit_config,
        calibrate=True,
        diff_scale=True,
        num_rgs=4,
        num_rcabs=4,
        ca_reduction=16,
        rg_path_drop=0.1,
        predictor_dropout=0.1,
    )

    patch_dim = 16  # for VIT_VARIANT_B16

    return vtamiq_config, patch_dim


class PreferenceModule(nn.Module):
    """
        simple module used to remap from quality prediction difference (delta Q) to preference judgement.
    """
    def __init__(self, weight=1.):
        super(PreferenceModule, self).__init__()
        self.p = nn.Parameter(torch.Tensor(weight))

    def forward(self, q1, q2):
        return torch.sigmoid(self.p * (q2 - q1)).flatten()
