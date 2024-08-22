import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import RepCNNBlock


class SegmentationHead(nn.Module):
    def __init__(self, num_classes, in_channels, head_dim, dropout_ratio=0.1):
        super().__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.head_dim = head_dim
        self.num_classes = num_classes

        self.linear_c4 = RepCNNBlock(
            c4_in_channels, head_dim, ks=1, padding=0, use_act=False
        )

        self.linear_c3 = RepCNNBlock(
            c3_in_channels, head_dim, ks=1, padding=0, use_act=False
        )

        self.linear_c2 = RepCNNBlock(
            c2_in_channels, head_dim, ks=1, padding=0, use_act=False
        )

        self.linear_c1 = RepCNNBlock(
            c1_in_channels, head_dim, ks=1, padding=0, use_act=False
        )

        self.linear_fuse = RepCNNBlock(
            head_dim * 4, head_dim, ks=1, padding=0, use_act=False
        )

        self.linear_pred = nn.Conv2d(
            head_dim, num_classes, kernel_size=1, stride=1, padding=0
        )

        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = F.interpolate(
            self.linear_c4(c4), scale_factor=8, mode="bilinear", align_corners=True
        )

        _c3 = F.interpolate(
            self.linear_c3(c3), scale_factor=4, mode="bilinear", align_corners=True
        )

        _c2 = F.interpolate(
            self.linear_c2(c2), scale_factor=2, mode="bilinear", align_corners=True
        )

        # _c1 = F.interpolate(
        #     self.linear_c1(c1), size=c1.size()[2:], mode="bilinear", align_corners=True
        # )
        _c1 = self.linear_c1(c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)

        return self.linear_pred(x)
