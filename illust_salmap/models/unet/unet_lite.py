from torch.nn import *
from torchinfo import summary

from illust_salmap.models.blocks import AddSkip, NoSkip
from illust_salmap.models.ez_bench import benchmark
from illust_salmap.models.checkpoint import load_weights


class UNetLite(Module):
    def __init__(self, in_channels=3, out_channels=1, use_skip_connection=True):
        super(UNetLite, self).__init__()

        # resize=True because the two deepest levels skip across a resolution change:
        # the bottleneck halves 8px to 4px, so enc_512 has to come down to meet it.
        # Stateless, so one instance serves every level.
        self.skip = AddSkip(resize=True) if use_skip_connection else NoSkip()

        self.encoder_in_32 = Encoder(in_channels, 32)
        self.encoder_32_64 = Encoder(32, 64)
        self.encoder_64_128 = Encoder(64, 128)
        self.encoder_128_256 = Encoder(128, 256)
        self.encoder_256_512 = Encoder(256, 512)

        self.bottleneck = Bottleneck()

        self.decoder_512_512 = Decoder(512, 512)
        self.decoder_512_256 = Decoder(512, 256)
        self.decoder_256_128 = Decoder(256, 128)
        self.decoder_128_64 = Decoder(128, 64)
        self.decoder_64_32 = Decoder(64, 32)
        self.decoder_32_out = Decoder(32, 32)

        # Projected outside the last Decoder, which normalizes. BatchNorm2d on a
        # 1-channel output rescales every prediction by its own batch's statistics, so the
        # same image would score differently depending on what it was batched with.
        self.output = Conv2d(32, out_channels, 1)
        self.head = Tanh()

    def forward(self, x):
        # var name is "{layer}_{output_ch}"
        enc_32 = self.encoder_in_32(x)
        enc_64 = self.encoder_32_64(enc_32)
        enc_128 = self.encoder_64_128(enc_64)
        enc_256 = self.encoder_128_256(enc_128)
        enc_512 = self.encoder_256_512(enc_256)

        bottle_512 = self.bottleneck(enc_512)

        dec_512 = self.decoder_512_512(self.skip(bottle_512, enc_512))
        dec_256 = self.decoder_512_256(self.skip(dec_512, bottle_512))
        dec_128 = self.decoder_256_128(self.skip(dec_256, enc_256))
        dec_64 = self.decoder_128_64(self.skip(dec_128, enc_128))
        dec_32 = self.decoder_64_32(self.skip(dec_64, enc_64))
        dec_out = self.decoder_32_out(self.skip(dec_32, enc_32))

        return self.head(self.output(dec_out))


class Encoder(Module):
    def __init__(self, in_channels=3, out_channels=64, dropout_prob=0.1):
        super(Encoder, self).__init__()

        self.encoder = Sequential(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  MaxPool2d(2, 2),
                                  BatchNorm2d(out_channels),
                                  LeakyReLU(0.2),
                                  Dropout2d(dropout_prob), )

    def forward(self, x):
        return self.encoder(x)


class Decoder(Module):
    """Single conv after the upsample -- lighter than the shared `blocks.DecoderBlock`,
    which is why UNetLite keeps its own.

    Ends on an activation. It used to end on `Conv2d -> BatchNorm2d`, and since `AddSkip`
    carries no activation either, the stretch from that conv through the skip and into the
    next block's ConvTranspose2d was affine: the whole decoder held one LeakyReLU per
    block and half its linear layers folded together. The trailing `Dropout2d(p=0)` that
    sat here was a dead layer and is gone.
    """

    def __init__(self, in_channels=64, out_channels=3):
        super(Decoder, self).__init__()

        self.decoder = Sequential(ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
                                  LeakyReLU(0.2),
                                  Conv2d(in_channels, out_channels, 3, 1, 1),
                                  BatchNorm2d(out_channels),
                                  LeakyReLU(0.2), )

    def forward(self, x):
        return self.decoder(x)


class Bottleneck(Module):
    def __init__(self, channels=512, dropout_prob=0.3):
        super(Bottleneck, self).__init__()

        self.bottleneck = Sequential(Conv2d(channels, channels, 4, 2, 1),
                                     BatchNorm2d(channels),
                                     LeakyReLU(),
                                     Dropout(dropout_prob), )

    def forward(self, x):
        return self.bottleneck(x)


def unet_lite(ckpt_path=None) -> UNetLite:
    model = UNetLite()

    if ckpt_path:
        load_weights(model, ckpt_path)

    return model


if __name__ == '__main__':
    model = UNetLite()
    shape = (4, 3, 256, 256)
    summary(model, shape)
    benchmark(model, shape)
