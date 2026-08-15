import torch
from torch.nn import LeakyReLU, Tanh
from torchinfo import summary

from illust_salmap.models.checkpoint import load_weights
from illust_salmap.models.ez_bench import benchmark
from illust_salmap.models.unet.unet_v3 import UNetV3


class UNetV2(UNetV3):
    """The v2 lineage expressed as a configuration of UNetV3.

    UNetV3 was written to make this model scalable, so the two were never independent
    designs -- v2 is v3 with the depth fixed at four levels of 64/128/256/512, a 5x5
    first convolution, and no residual shortcut (the shortcut is what lets v3 stack
    deeper, and at this depth it is not needed).

    A subclass rather than a factory returning a plain UNetV3, because
    `training/paths.py:get_save_path` derives the output directory from the network's
    class name: returning UNetV3 here would make `--model unet_v2` and `--model unet_v3`
    write their logs and checkpoints to the same place.

    Two things did change when this was folded in, and neither is reproducible from the
    old weights. Downsampling is now the strided convolution v3 uses rather than
    MaxPool2d, and the bottleneck is v3's dilated EncoderBlock rather than the old
    conv-plus-global-attention block -- the SE gate inside the encoder block covers
    roughly what that attention did.
    """

    def __init__(self, classes: int = 1, in_channels: int = 3, activation=LeakyReLU(), head=Tanh()):
        super().__init__(
            classes=classes,
            in_channels=in_channels,
            activation=activation,
            head=head,
            # base 64 with 3 mid blocks gives 64/128/256/512, and scale_stride=1 scales
            # at every level -- the ladder the hand-written UNetV2 had.
            num_blocks=3,
            base_channels=64,
            scale_stride=1,
            kernel_size=5,
            residual=False,
        )


def unet_v2(ckpt_path=None) -> UNetV2:
    model = UNetV2()

    if ckpt_path:
        load_weights(model, ckpt_path)

    return model


if __name__ == '__main__':
    model = UNetV2()
    shape = (4, 3, 256, 256)
    model(torch.randn(shape))
    summary(model, shape)
    benchmark(model, shape)
