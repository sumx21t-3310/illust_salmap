"""Pieces shared by more than one model.

The UNet-family decoders converged on one graph once the norm/activation ordering was
fixed, so `DecoderBlock` lives here and all three use it. The encoders have not
converged -- they still differ in whether they carry SE, a residual shortcut, or a
strided-conv downsample -- so each stays in its own model's file. `UNetLite` keeps its
own single-conv decoder too; sharing this one would double its decoder and stop it
being lite.

Note that a shared block couples the models: a change made here moves every baseline
that uses it, and previously logged metrics stop being comparable.
"""
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d, BatchNorm2d, Conv2d, ConvTranspose2d, Dropout2d, Linear, Module, Parameter, ReLU, Sigmoid,
)
from torch.nn.functional import interpolate


class SEBlock(Module):
    """Squeeze-and-excitation: per-channel gain from the channel's global average.

    Shared because UNetV2 and UNetV3 held byte-identical copies of it.
    """

    def __init__(self, in_channels: int, reduction: int = 16, bias: bool = False):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Linear(in_channels, in_channels // reduction, bias=bias)
        self.relu = ReLU()
        self.fc2 = Linear(in_channels // reduction, in_channels, bias=bias)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        squeeze = self.avg_pool(x).view(x.size(0), -1)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(x.size(0), x.size(1), 1, 1)
        return x * excitation


def align_skip(x: Tensor, y: Tensor, resize: bool) -> Tensor:
    """Bring the skip `y` to `x`'s spatial size."""
    if y.shape[2:] == x.shape[2:]:
        return y

    # Raised rather than left to torch: addition broadcasts a spatially smaller skip
    # instead of failing (a 1x1 skip against a 64x64 input is a legal broadcast), so a
    # mis-wired model would train on a silently wrong graph.
    if not resize:
        raise ValueError(
            f"skip is {tuple(y.shape[2:])} but the decoder input is {tuple(x.shape[2:])}; "
            f"pass resize=True to interpolate instead of requiring a match"
        )

    return interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=True)


class NoSkip(Module):
    """Discards the encoder feature.

    Exists so that "no skip connection" is a choice about wiring, made in one place,
    rather than a flag every decoder has to carry and honour.
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x


class ConcatSkip(Module):
    """Channel concatenation -- the skip the original U-Net uses.

    The fused tensor is `x.channels + y.channels` wide, so the decoder downstream has to
    be built for the fused width, not for `x`'s.
    """

    def __init__(self, resize: bool = False):
        super().__init__()
        self.resize = resize

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cat([x, align_skip(x, y, self.resize)], dim=1)


class AddSkip(Module):
    """Addition. Preserves the channel count, and therefore requires x and y to share it."""

    def __init__(self, resize: bool = False):
        super().__init__()
        self.resize = resize

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + align_skip(x, y, self.resize)


class EncoderBlock(Module):
    """`conv -> norm -> activation`, twice, with an optional SE gate and residual shortcut.

    Does not downsample. The model does that between blocks, so this block's output is
    the tensor the skip branches off -- at the level's own resolution. Taking the skip
    after the downsample instead leaves the finest one at half the input resolution, and
    the network's own output resolution with no encoder feature in it at all.

    `se` and `residual` are what separate the three models that use this: UNet has
    neither, the UNetV2 configuration has SE, and UNetV3 has both (the shortcut is what
    lets it stack deep, which is the whole point of that model).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout_prob: float = 0.1,
            activation: Module = ReLU(),
            se: bool = True,
            residual: bool = True,
    ):
        super().__init__()
        # "same" rather than an explicit padding so that dilation stays free: for a 5x5
        # kernel at dilation 1 the two are identical, but a fixed padding silently stops
        # matching as soon as dilation moves.
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, 1, "same", dilation=dilation)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, "same")

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        # Built only when used, so a block that has neither keeps them out of the
        # state_dict and out of the optimizer.
        self.se_block = SEBlock(out_channels) if se else None
        self.shortcut = Conv2d(in_channels, out_channels, 1, 1, "same", bias=False) if residual else None
        # Same treatment: a Dropout2d(p=0) is a dead layer, and UNet's bottleneck is
        # exactly this block with no dropout.
        self.dropout = Dropout2d(dropout_prob) if dropout_prob > 0 else None

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)

        if self.se_block is not None:
            x = self.se_block(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)

        if self.shortcut is not None:
            x = x + self.shortcut(identity)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class UpsampleBlock(Module):
    """Stride-2 transposed convolution, then norm and activation.

    A decoder level runs `upsample -> skip -> DecoderBlock`, so the transposed conv and
    the decoder's first conv are separated only by the skip fusion -- which is linear.
    The norm and activation here are what stop those two convolutions from folding into
    one. The original U-Net has exactly that adjacency (its up-conv feeds the concat and
    then a 3x3 conv with no nonlinearity in between); this does not.
    """

    def __init__(self, in_channels: int, out_channels: int = None, activation: Module = ReLU()):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.conv_transpose = ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.batch_norm = BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.batch_norm(self.conv_transpose(x)))


class DecoderBlock(Module):
    """`conv -> norm -> activation`, twice, then dropout.

    Scaling up is not this block's job -- `UpsampleBlock` does it before the skip is
    fused, so that the encoder feature enters at its own native resolution instead of
    being carried up through a transposed convolution.

    No two linear ops are adjacent anywhere in this block, and it ends on an activation
    rather than a norm, so stacking it does not produce affine stretches.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_prob: float = 0.2,
            activation: Module = ReLU(),
    ):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)

        self.batch_norm1 = BatchNorm2d(out_channels)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.dropout = Dropout2d(dropout_prob)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        # Without an activation here and below, batch norm is affine at inference and the
        # whole decoder collapses into a single affine map, whatever its depth.
        x = self.activation(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)

        return self.dropout(x)


class GatedSkip(Module):
    """Addition through a learned gate. Same channel constraint as `AddSkip`.

    The gate scales the skip: `sigmoid(skip_gate) * y`, so it can be turned down to
    nothing and the skip keeps the encoder feature's sign and magnitude.

    It used to be `sigmoid(skip_gate * y)`, with the gate *inside* the sigmoid. That is
    not a gate: the result is bounded to (0, 1) whatever the gate learns, always
    positive, and equal to 0.5 rather than 0 wherever the encoder feature is 0 -- so a
    "closed" skip still added a near-constant 0.5 to most of the map.
    """

    def __init__(
            self,
            skip_weight: float = 0.5,
            shape: int | Sequence[int] = 1,
            activation: Module = ReLU(),
            resize: bool = False,
    ):
        super().__init__()
        self.skip_gate = Parameter(torch.ones(shape) * skip_weight)
        self.activation = activation
        self.resize = resize

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        y = align_skip(x, y, self.resize)
        return self.activation(x + torch.sigmoid(self.skip_gate) * y)
