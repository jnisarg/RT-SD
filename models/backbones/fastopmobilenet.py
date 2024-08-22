import torch.nn as nn

from models.modules import PatchEmbed, basic_blocks, stem


class RepMobileNet(nn.Module):
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
    ):
        super().__init__()

        self.op_stem = stem(3, embed_dims[0], inference_mode)

        network = []
        for ix in range(len(layers)):
            stage = basic_blocks(
                embed_dims[ix],
                ix,
                layers,
                ks=ks,
                mlp_ratio=mlp_ratios[ix],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            network.append(stage)
            if ix >= len(layers) - 1:
                break
            if downsamples[ix] or embed_dims[ix] != embed_dims[ix + 1]:
                network.append(
                    PatchEmbed(
                        embed_dims[ix],
                        embed_dims[ix + 1],
                        down_patch_size,
                        down_stride,
                        inference_mode,
                    )
                )
        self.network = nn.ModuleList(network)

        self.out_indices = [0, 2, 4, 6]
        self.out_channels = embed_dims
        for i_emb, i_layer in enumerate(self.out_indices):
            layer = nn.BatchNorm2d(embed_dims[i_emb])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        x = self.op_stem(x)

        outs = []
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)
