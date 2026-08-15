"""The U-Net family.

Deliberately empty of imports. Re-exporting the factories here would let `cli.py` pull
all four from one line, but it also makes `python -m illust_salmap.models.unet.unet_v3`
-- the documented way to summarise and benchmark a network -- import the module twice:
once because this file pulls it in, then again as `__main__`. Python warns about it, and
the two copies really are separate class objects.

The building blocks stay in `models/blocks.py` rather than moving in here: the skip
fusions, `DecoderBlock` and `UpsampleBlock` are not specific to a convolutional encoder,
and a ViT-based model producing a dense map needs exactly the same decoder. Only
`EncoderBlock` is really tied to this family, and it is left with the rest until there is
a second user to split against.

Module names match the factory names they export (`unet_v3.py` defines `unet_v3`). Do not
shorten them to `v3.py`: the tests only check the factory's signature, so a drift between
the two is caught by nothing, and the history has a run of "fix factory function name"
commits from exactly that.
"""
