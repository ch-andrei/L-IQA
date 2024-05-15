from torch import nn
from .transformer import VisionTransformer, get_vit_config, \
    VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16


class VisionTransformerBackbone(nn.Module):
    @property
    def vit_hidden_size(self):
        return self.transformer.hidden_size

    @property
    def vit_num_layers(self):
        return len(self.transformer.encoder.layers)

    def __init__(
            self,
            variant=VIT_VARIANT_B16,  # choose from [VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16]
            use_patch_embedding=True,
            use_pos_embedding=True,
            use_cls_token=True,
            use_classifier=True,
            use_layer_scale=False,
            pretrained=True,
            num_keep_layers=-1,
            num_adapters=0,
            num_scales=0,
            num_extra_tokens=0,
            path_drop_prob=0.0,
            attn_bias_weight=0.0,
            return_layers=False,
            return_attention=False,
            **kwargs
    ):
        super().__init__()
        self.transformer = VisionTransformer(
            config=get_vit_config(variant),
            use_patch_embedding=use_patch_embedding,
            use_pos_embedding=use_pos_embedding,
            use_cls_token=use_cls_token,
            use_classifier=use_classifier,
            use_layer_scale=use_layer_scale,
            num_keep_layers=num_keep_layers,
            num_extra_tokens=num_extra_tokens,
            num_adapters=num_adapters,
            num_scales=num_scales,
            path_drop_prob=path_drop_prob,
            pretrained=pretrained,
            attn_bias_weight=attn_bias_weight,
            return_layers=return_layers,
            return_attention=return_attention,
        )

    def forward_vit(self, patches, patches_pos, patches_scale, tokens_only=True, adapter_num=None, attn_bias=None):
        if adapter_num is None or adapter_num < 0:
            # default case: if transformer has adapters, use them
            adapter_num = 0 if self.transformer.use_adapters else -1
        x, attn_weights, hidden_states = self.transformer.forward(
            patches, patches_pos, patches_scale, tokens_only=tokens_only, adapter_num=adapter_num, attn_bias=attn_bias)
        return x, attn_weights, hidden_states
