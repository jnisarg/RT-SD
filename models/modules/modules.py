import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_


def reparameterize_model(model):
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


class RepConv(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        ks,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inference_mode=False,
        use_act=True,
    ):
        super().__init__()

        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ks = ks
        self.c1 = c1
        self.c2 = c2

        if use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def _conv_bn(self, ks, padding):
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                self.c1,
                self.c2,
                ks,
                self.stride,
                padding,
                self.dilation,
                self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(self.c2))
        return mod_list

    def _fuse_bn_tensor(self, branch):
        raise NotImplementedError(
            "fuse_bn_tensor is not implemented, it needs to be implemented in the child class"
        )

    def _get_kernel_bias(self):
        raise NotImplementedError(
            "get_kernel_bias is not implemented, it needs to be implemented in the child class"
        )

    def reparameterize(self):
        if self.inference_mode:
            return

        raise NotImplementedError(
            "reparameterize is not implemented, it needs to be implemented in the child class"
        )

    def forward(self, x):
        raise NotImplementedError(
            "forward is not implemented, it needs to be implemented in the child class"
        )


class RepCNNBlock(RepConv):
    def __init__(
        self,
        c1,
        c2,
        ks,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inference_mode=False,
        use_act=True,
        use_scale_branch=True,
        num_conv_branches=1,
    ):
        super().__init__(
            c1, c2, ks, stride, padding, dilation, groups, inference_mode, use_act
        )

        self.num_conv_branches = num_conv_branches

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                c1, c2, ks, stride, padding, dilation, groups, bias=True
            )
        else:
            self.rbr_skip = nn.BatchNorm2d(c2) if c2 == c1 and stride == 1 else None

            if num_conv_branches > 0:
                self.rbr_conv = nn.ModuleList(
                    [
                        self._conv_bn(ks=ks, padding=padding)
                        for _ in range(self.num_conv_branches)
                    ]
                )
            else:
                self.rbr_conv = None

            self.rbr_scale = None
            if ks > 1 and use_scale_branch:
                self.rbr_scale = self._conv_bn(ks=1, padding=0)

    def forward(self, x):
        if self.inference_mode:
            return self.act(self.reparam_conv(x))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.act(out)

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.groups
                kernel_value = torch.zeros(
                    (self.c1, input_dim, self.ks, self.ks),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, self.ks // 2, self.ks // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _get_kernel_bias(self):
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.ks // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def reparameterize(self):
        if self.inference_mode:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.c1,
            self.c2,
            self.ks,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True


class RepLKConv(RepConv):
    def __init__(
        self,
        c1,
        c2,
        ks,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inference_mode=False,
        use_act=True,
        small_kernel=3,
    ):
        super().__init__(
            c1, c2, ks, stride, padding, dilation, groups, inference_mode, use_act
        )

        self.ks = ks
        self.small_kernel = small_kernel
        self.padding = self.ks // 2

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                c1, c2, ks, stride, self.padding, dilation, groups, bias=True
            )
        else:
            self.lkb_origin = self._conv_bn(ks=ks, padding=self.padding)
            if small_kernel is not None:
                assert (
                    small_kernel <= ks
                ), "The kernel size for re-param cannot be larger than the large kernel"
                self.small_conv = self._conv_bn(
                    ks=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x):
        if self.inference_mode:
            return self.act(self.lkb_reparam(x))

        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out += self.small_conv(x)

        return self.act(out)

    @staticmethod
    def _fuse_bn_tensor(conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _get_kernel_bias(self):
        eq_k, eq_b = self._fuse_bn_tensor(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn_tensor(
                self.small_conv.conv, self.small_conv.bn
            )
            eq_b += small_b
            eq_k += F.pad(small_k, [(self.ks - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def reparameterize(self):
        if self.inference_mode:
            return

        eq_k, eq_b = self._get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            self.c1,
            self.c2,
            self.ks,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=True,
        )

        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b

        for param in self.parameters():
            param.detach_()
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

        self.inference_mode = True


def stem(c1, c2, inference_mode=False):
    return nn.Sequential(
        RepCNNBlock(c1, c2, ks=3, stride=2, padding=1, inference_mode=inference_mode),
        RepCNNBlock(
            c2, c2, ks=3, stride=2, padding=1, groups=c2, inference_mode=inference_mode
        ),
        RepCNNBlock(c2, c2, ks=1, inference_mode=inference_mode),
    )


class MHSA(nn.Module):
    """Multi-Head Self Attention Module"""

    def __init__(self, dim, head_dim, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x = torch.flatten(x, start_dim=2).transpose(1, 2)

        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Module):

    def __init__(self, c1, c2, patch_size, stride, inference_mode=False):
        super().__init__()
        block = list()
        block.append(
            RepLKConv(
                c1,
                c2,
                ks=patch_size,
                stride=stride,
                groups=c1,
                small_kernel=3,
                inference_mode=inference_mode,
            )
        )
        block.append(RepCNNBlock(c2, c2, ks=1, inference_mode=inference_mode))
        self.proj = nn.Sequential(*block)

    def forward(self, x):
        return self.proj(x)


class RepMixer(nn.Module):

    def __init__(
        self,
        dim,
        ks=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode=False,
    ):
        super().__init__()
        self.dim = dim
        self.ks = ks
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim,
                self.ks,
                stride=1,
                padding=self.ks // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = RepCNNBlock(
                dim,
                dim,
                ks,
                padding=self.ks // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = RepCNNBlock(
                dim, dim, ks, padding=self.ks // 2, groups=dim, use_act=False
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
                )

    def forward(self, x):
        if hasattr(self, "reparam_conv"):
            return self.reparam_conv(x)

        if self.use_layer_scale:
            x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
        else:
            x = x + self.mixer(x) - self.norm(x)

        return x

    def reparameterize(self):
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = self.mixer.id_tensor + (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            self.dim,
            self.dim,
            self.ks,
            stride=1,
            padding=self.ks // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for param in self.parameters():
            param.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")

        self.inference_mode = True


class ConvFFN(nn.Module):

    def __init__(self, c1, hidden_dims, drop=0.0):
        super().__init__()
        hidden_dims = hidden_dims or c1
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv", nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1, bias=False)
        )
        self.conv.add_module("bn", nn.BatchNorm2d(c1))
        self.fc1 = nn.Conv2d(c1, hidden_dims, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_dims, c1, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ks=3,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode=False,
    ):
        super().__init__()
        self.token_mixer = RepMixer(
            dim, ks, use_layer_scale, layer_scale_init_value, inference_mode
        )

        assert mlp_ratio > 0, f"MLP ratio should be greater than 0, but got {mlp_ratio}"
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(dim, mlp_hidden_dim, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x) + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x) + self.drop_path(self.convffn(x))

        return x


def basic_blocks(
    dim,
    block_index,
    num_blocks,
    ks=3,
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    inference_mode=False,
):
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx * sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        blocks.append(
            RepMixerBlock(
                dim,
                ks=ks,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks
