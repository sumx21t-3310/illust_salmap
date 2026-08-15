"""Matching images to their saliency maps."""

from pathlib import Path
from typing import Iterable

MISMATCH_EXAMPLES = 5


def paired_paths(image_paths: Iterable[Path], map_paths: Iterable[Path], context: str = "") -> list[tuple[Path, Path]]:
    """
    Pairs each image with the saliency map whose filename stem matches it.

    The two directories are globbed separately, so pairing them positionally rests on two
    assumptions that nothing enforces: that both listings have the same length, and that
    they sort into the same order. `zip` truncates to the shorter one without a word, so a
    single missing map shifts every pair after it and the model trains on images labelled
    with a neighbour's saliency -- which looks like a model that will not converge rather
    than like a bug. The extensions differ (.jpg against .png), so the sort order is not
    guaranteed to agree either.

    Args:
        image_paths (Iterable[Path]): Image files.
        map_paths (Iterable[Path]): Saliency map files.
        context (str): Named in the error, to say which directory is short.

    Returns:
        list[tuple[Path, Path]]: One (image, map) pair per stem, in stem order.

    Raises:
        FileNotFoundError: If any stem is missing from either side.
    """
    images = {path.stem: path for path in image_paths}
    maps = {path.stem: path for path in map_paths}

    missing_maps = sorted(images.keys() - maps.keys())
    missing_images = sorted(maps.keys() - images.keys())

    if missing_maps or missing_images:
        raise FileNotFoundError(_mismatch_message(context, missing_maps, missing_images))

    return [(images[stem], maps[stem]) for stem in sorted(images)]


def _mismatch_message(context: str, missing_maps: list[str], missing_images: list[str]) -> str:
    where = f" in {context}" if context else ""
    reasons = []

    if missing_maps:
        reasons.append(f"{len(missing_maps)} image(s) have no map: {_examples(missing_maps)}")

    if missing_images:
        reasons.append(f"{len(missing_images)} map(s) have no image: {_examples(missing_images)}")

    return (
        f"images and saliency maps do not line up{where}. " + "; ".join(reasons) +
        ". The archive is probably incompletely extracted -- delete it and let the "
        "installer fetch it again."
    )


def _examples(stems: list[str]) -> str:
    shown = ", ".join(stems[:MISMATCH_EXAMPLES])

    return shown if len(stems) <= MISMATCH_EXAMPLES else f"{shown}, ..."
