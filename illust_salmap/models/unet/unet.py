from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, LeakyReLU, Tanh
from torchinfo import summary

from illust_salmap.models.blocks import ConcatSkip, DecoderBlock, EncoderBlock, UpsampleBlock
from illust_salmap.models.ez_bench import benchmark
from illust_salmap.models.checkpoint import load_weights


class UNet(Module):
    def __init__(
            self,
            num_classes: int = 1,
            in_channels: int = 3,
            activation: Module = LeakyReLU(),
            head: Module = Tanh()
    ):
        super().__init__()
        # The plain U-Net encoder: 5x5 first conv, no SE gate, no residual shortcut. Those
        # two are what separate this from the UNetV2/UNetV3 lineage.
        encoder = dict(kernel_size=5, dropout_prob=0.3, activation=activation, se=False, residual=False)

        self.encoder1 = EncoderBlock(in_channels, 64, **encoder)
        self.encoder2 = EncoderBlock(64, 128, **encoder)
        self.encoder3 = EncoderBlock(128, 256, **encoder)
        self.encoder4 = EncoderBlock(256, 512, **encoder)

        # The same block as the encoders, minus the dropout: a bottleneck here was only
        # ever `conv -> BN -> act` twice. UNetV3 does the same, with a dilation.
        self.bottleneck = EncoderBlock(512, 512, dropout_prob=0.0, activation=activation, se=False, residual=False)

        # Owned by the model, not by EncoderBlock: the skip has to branch off *before*
        # the pooling, so the block cannot be the thing that pools. Stateless, so one
        # instance serves every level.
        self.pool = MaxPool2d(2, 2)

        # Each upsample restores the resolution of the encoder feature it is about to be
        # fused with, so the skip enters at its own resolution rather than being carried
        # up through a transposed conv.
        self.up4 = UpsampleBlock(512, 512, activation=activation)
        self.up3 = UpsampleBlock(256, 256, activation=activation)
        self.up2 = UpsampleBlock(128, 128, activation=activation)
        self.up1 = UpsampleBlock(64, 64, activation=activation)

        # Stateless, so one instance serves every level.
        self.skip = ConcatSkip()

        # in_channels is the width *after* the skip is concatenated -- 512 from up4 plus
        # 512 from encoder4, and so on down.
        self.decoder4 = DecoderBlock(1024, 256, activation=activation)
        self.decoder3 = DecoderBlock(512, 128, activation=activation)
        self.decoder2 = DecoderBlock(256, 64, activation=activation)
        self.decoder1 = DecoderBlock(128, 64, activation=activation)

        # The projection to num_classes is kept out of DecoderBlock: the block normalizes
        # and drops out, and both are wrong on the output map. BatchNorm2d(1) would rescale
        # every prediction by its batch's statistics, and Dropout2d drops whole channels --
        # with one channel that is the entire map.
        self.output = Conv2d(64, num_classes, 1)
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        # enc1 is at the input resolution, enc2 at half, and so on: each is taken before
        # the pool that feeds the next level.
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottle = self.bottleneck(self.pool(enc4))

        dec4 = self.decoder4(self.skip(self.up4(bottle), enc4))
        dec3 = self.decoder3(self.skip(self.up3(dec4), enc3))
        dec2 = self.decoder2(self.skip(self.up2(dec3), enc2))
        dec1 = self.decoder1(self.skip(self.up1(dec2), enc1))
        return self.head(self.output(dec1))


def unet(ckpt_path=None) -> UNet:
    model = UNet()

    if ckpt_path:
        load_weights(model, ckpt_path)

    return model


if __name__ == '__main__':
    model = UNet()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
