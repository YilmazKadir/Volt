from functools import partial
import torch
import torch.nn as nn


class Detokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj = nn.Linear(
            in_channels,
            kernel_size**3 * out_channels,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, coarse_features, inverse, offset_id):
        K = self.kernel_size

        all_offsets = self.proj(coarse_features)
        all_offsets = all_offsets.view(-1, K**3, self.out_channels)

        fine_features = all_offsets[inverse, offset_id]
        fine_features = fine_features + self.bias

        return fine_features


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        act_layer = nn.GELU

        self.pre = nn.Sequential(
            bn_layer(in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels, bias=False),
            bn_layer(out_channels),
            act_layer(),
        )

        self.unembed = Detokenizer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
        )

        self.post = nn.Sequential(
            bn_layer(out_channels),
            act_layer(),
        )

    def forward(self, x, inverse, offset_id):
        x = self.pre(x)
        x = self.unembed(x, inverse, offset_id)
        x = self.post(x)
        return x
