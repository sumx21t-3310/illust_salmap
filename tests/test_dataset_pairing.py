from pathlib import Path

import pytest

from illust_salmap.dataset.pairing import paired_paths


def paths(directory: str, names: list[str]) -> list[Path]:
    return [Path(directory) / name for name in names]


def test_pairs_are_matched_by_stem_not_by_position():
    """
    Regression: the two listings used to be zipped positionally. The extensions differ,
    so sorting them separately does not guarantee the same order.
    """
    images = paths("imgs", ["b.jpg", "a.jpg", "c.jpg"])
    maps = paths("maps", ["c.png", "a.png", "b.png"])

    assert [(image.stem, saliency_map.stem) for image, saliency_map in paired_paths(images, maps)] == [
        ("a", "a"), ("b", "b"), ("c", "c"),
    ]


def test_a_missing_map_is_an_error_rather_than_a_silent_shift():
    """
    Regression: `zip` truncated to the shorter listing, so one missing map moved every
    later image onto its neighbour's saliency map and nothing reported it.
    """
    images = paths("imgs", ["a.jpg", "b.jpg", "c.jpg"])
    maps = paths("maps", ["a.png", "c.png"])

    with pytest.raises(FileNotFoundError, match="1 image"):
        paired_paths(images, maps)


def test_a_missing_image_is_also_an_error():
    images = paths("imgs", ["a.jpg"])
    maps = paths("maps", ["a.png", "b.png"])

    with pytest.raises(FileNotFoundError, match="1 map"):
        paired_paths(images, maps)


def test_the_error_names_the_directory_and_the_offending_stems():
    with pytest.raises(FileNotFoundError) as error:
        paired_paths(paths("imgs", ["missing_one.jpg"]), [], "cat2000/Action")

    assert "cat2000/Action" in str(error.value)
    assert "missing_one" in str(error.value)


def test_the_error_does_not_list_every_stem_of_a_broken_archive():
    images = paths("imgs", [f"{index:04d}.jpg" for index in range(500)])

    with pytest.raises(FileNotFoundError) as error:
        paired_paths(images, [])

    assert "500 image(s)" in str(error.value)
    assert "..." in str(error.value)
    assert len(str(error.value)) < 400


def test_empty_directories_pair_to_nothing():
    assert paired_paths([], []) == []
