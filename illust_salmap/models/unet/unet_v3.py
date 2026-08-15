from argparse import ArgumentParser

from torch import Tensor
from torch.nn import *
from torchinfo import summary

from illust_salmap.models.blocks import DecoderBlock, EncoderBlock, GatedSkip, UpsampleBlock
from illust_salmap.models.ez_bench import benchmark
from illust_salmap.models.checkpoint import load_weights


class UNetV3(Module):
    def __init__(self, classes: int = 1,
                 in_channels: int = 3,
                 activation=SiLU(),
                 # Tanh, not Sigmoid: the datamodules normalize maps to [-1, 1], so a head
                 # that cannot go negative cannot represent background at all.
                 head=Tanh(),
                 # Depth. 4 is what these datasets (1000-4000 images) support; with
                 # max_channels below, raising it now costs roughly linearly rather than
                 # quadratically.
                 num_blocks: int = 4,
                 base_channels: int = 32,
                 scale_stride=2,
                 # Without this the width is tied to the depth: channels double every
                 # block, so the bottleneck is base_channels * 2**num_blocks and the
                 # convolution cost is quadratic in that. num_blocks=7 meant a
                 # 4096-channel bottleneck and 1.3B parameters. Capping the width lets
                 # num_blocks actually be the scale knob it was meant to be. 512 is a
                 # no-op at the default config, where the ladder tops out there anyway.
                 max_channels: int = 512,
                 # The knobs that separate this from the UNetV2 configuration.
                 kernel_size: int = 3,
                 se: bool = True,
                 residual: bool = True,
                 ):
        super().__init__()

        def width(level: int) -> int:
            return min(base_channels * (2 ** level), max_channels)

        encoder = dict(kernel_size=kernel_size, activation=activation, se=se, residual=residual)

        mid_encoders = []
        mid_downsamples = []
        mid_decoders = []
        mid_upsamples = []

        for i in range(num_blocks):
            mid_encoders.append(
                EncoderBlock(width(i), width(i + 1), dilation=2 ** (i + 1), **encoder))
            # Hoisted out of EncoderBlock so the skip can branch off ahead of it. Identity
            # where the level does not scale, which keeps this list aligned with encoders.
            mid_downsamples.append(
                Conv2d(width(i + 1), width(i + 1), 2, 2, bias=False) if i % scale_stride == 0 else Identity())

        for i in range(num_blocks):
            mid_decoders.append(DecoderBlock(width(i + 1), width(i), activation=activation))
            mid_upsamples.append(
                UpsampleBlock(width(i + 1), activation=activation) if i % scale_stride == 0 else Identity())

        self.encoders = ModuleList([
            EncoderBlock(in_channels, base_channels, dilation=1, **encoder),
            *mid_encoders
        ])

        self.downsamples = ModuleList([
            Conv2d(base_channels, base_channels, 2, 2, bias=False),
            *mid_downsamples
        ])

        bottleneck_channels = width(num_blocks)

        # Kernel 3 regardless of `kernel_size`, which only widens the shallow levels'
        # receptive field. Here dilation is already 16, so a 5x5 would span 65 pixels on a
        # 32-pixel feature map -- almost entirely padding -- while costing 2.8x the
        # weights at the widest point in the network.
        self.bottleneck = EncoderBlock(bottleneck_channels, bottleneck_channels, dilation=16,
                                       **{**encoder, "kernel_size": 3})

        self.decoders = ModuleList([
            *reversed(mid_decoders),
            DecoderBlock(base_channels, base_channels, activation=activation),
        ])

        # Reversing the encoder's scaling pattern makes each upsample land on exactly the
        # resolution of the encoder output it will be fused with.
        self.upsamples = ModuleList([
            *reversed(mid_upsamples),
            UpsampleBlock(base_channels, activation=activation),
        ])

        # One per decoder: each holds its own `skip_gate`.
        self.skips = ModuleList([GatedSkip() for _ in self.decoders])

        # Projected outside the last DecoderBlock, which normalizes and drops out. On a
        # 1-channel output those would rescale the prediction by its batch's statistics
        # and zero the entire map, respectively.
        self.output = Conv2d(base_channels, classes, 1)
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        encoder_outputs = []
        for encoder, downsample in zip(self.encoders, self.downsamples):
            x = encoder(x)
            # Taken before the downsample, so the skip carries its level's own resolution
            # -- the finest one is at the input resolution, not half of it.
            encoder_outputs.append(x)
            x = downsample(x)

        x = self.bottleneck(x)

        # Upsample first, then fuse: the encoder output enters at its own resolution
        # instead of being carried up through a transposed conv.
        for upsample, skip, decoder, encoder_output in zip(
                self.upsamples, self.skips, self.decoders, reversed(encoder_outputs)):
            x = decoder(skip(upsample(x), encoder_output))

        return self.head(self.output(x))


def unet_v3(ckpt_path=None) -> UNetV3:
    model = UNetV3()

    if ckpt_path:
        load_weights(model, ckpt_path)

    return model


if __name__ == '__main__':
    parser = ArgumentParser(description="Print a torchinfo summary of the network and benchmark it.")
    parser.add_argument("ckpt_path", nargs="?",
                        help="Optional checkpoint to load first, to summarise trained weights.")
    args = parser.parse_args()

    model = unet_v3(ckpt_path=args.ckpt_path)
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
