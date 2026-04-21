from functools import partial

import spconv.pytorch as spconv
import torch.nn as nn


class Decoder(spconv.SparseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        indice_key: str = "embedding",
    ) -> None:
        super().__init__()
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        act_layer = nn.GELU

        self.up = spconv.SparseSequential(
            bn_layer(in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels, bias=False),
            bn_layer(out_channels),
            act_layer(),
            spconv.SparseInverseConv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                indice_key=indice_key,
                bias=False,
            ),
            bn_layer(out_channels),
            act_layer(),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        return self.up(x)
