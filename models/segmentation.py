import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from models.backbones import RepMobileNet
from models.heads import SegmentationHead


class SegmentationNetwork(nn.Module):
    def __init__(
        self,
        layers,
        embed_dims,
        mlp_ratios,
        downsamples,
        ks=3,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode=False,
        head_dim=64,
    ):
        super().__init__()

        self.backbone = RepMobileNet(
            layers,
            embed_dims,
            mlp_ratios,
            downsamples,
            ks=ks,
            down_patch_size=down_patch_size,
            down_stride=down_stride,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        self.head = SegmentationHead(
            num_classes=19, in_channels=embed_dims, head_dim=head_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inputs = self.backbone(x)
        x = F.interpolate(
            self.head(inputs), scale_factor=4, mode="bilinear", align_corners=False
        )
        return x


if __name__ == "__main__":

    from models.modules import reparameterize_model

    model = SegmentationNetwork(
        layers=[2, 2, 4, 2],
        embed_dims=[32, 64, 128, 256],
        mlp_ratios=[3, 3, 3, 3],
        downsamples=[True, True, True, True],
        head_dim=64,
    )
    model.eval()
    model = reparameterize_model(model)

    x = torch.randn(1, 3, 1024, 2048)
    y = model(x)

    print(y.shape)

    backbone_params = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )

    print(f"Backbone parameters: {backbone_params / 1e6}M")

    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)

    print(f"Head parameters: {head_params / 1e6}M")

    total_params = backbone_params + head_params

    print(f"Total parameters: {total_params / 1e6}M")
