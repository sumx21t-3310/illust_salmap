from collections import Counter

import pytest
import torch

from illust_salmap.dataset.cat2000 import Cat2000
from illust_salmap.dataset.split import stratified_indices

CAT2000_CATEGORIES = [
    "Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal", "Indoor", "Inverted",
    "Jumbled", "LineDrawing", "LowResolution", "Noisy", "Object", "OutdoorManMade",
    "OutdoorNatural", "Pattern", "Random", "Satelite", "Sketch", "Social",
]


def cat2000_groups(per_category: int = 100) -> list[str]:
    """The public half of CAT2000: 20 categories, 100 images each."""
    return [category for category in CAT2000_CATEGORIES for _ in range(per_category)]


class FakeCat2000Dataset:
    def __init__(self, root, image_transform=None, map_transform=None, categories=None):
        self.pair_categories = cat2000_groups()

    def __len__(self):
        return len(self.pair_categories)

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), torch.zeros(1, 8, 8)


def test_split_covers_every_index_exactly_once():
    groups = cat2000_groups()

    train, val = stratified_indices(groups, [0.9, 0.1], seed=42)

    assert sorted(train + val) == list(range(len(groups)))
    assert not set(train) & set(val)


def test_split_sizes_follow_the_paper_convention():
    """CAT2000's 2000 public images are conventionally used as 1800 train / 200 val."""
    train, val = stratified_indices(cat2000_groups(), [0.9, 0.1], seed=42)

    assert len(train) == 1800
    assert len(val) == 200


def test_every_category_is_represented_proportionally():
    groups = cat2000_groups()

    train, val = stratified_indices(groups, [0.9, 0.1], seed=42)

    train_counts = Counter(groups[index] for index in train)
    val_counts = Counter(groups[index] for index in val)

    assert set(val_counts) == set(CAT2000_CATEGORIES)
    assert set(train_counts.values()) == {90}
    assert set(val_counts.values()) == {10}


def test_split_is_reproducible():
    first = stratified_indices(cat2000_groups(), [0.9, 0.1], seed=42)
    second = stratified_indices(cat2000_groups(), [0.9, 0.1], seed=42)

    assert first == second


def test_a_different_seed_gives_a_different_split():
    first = stratified_indices(cat2000_groups(), [0.9, 0.1], seed=42)
    other = stratified_indices(cat2000_groups(), [0.9, 0.1], seed=7)

    assert first != other


def test_uneven_categories_still_fill_every_split():
    """A category smaller than the split count must not vanish from the larger split."""
    groups = ["tiny"] * 3 + ["big"] * 97

    train, val = stratified_indices(groups, [0.9, 0.1], seed=42)

    assert len(train) + len(val) == 100
    assert any(groups[index] == "tiny" for index in train)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        stratified_indices(["a", "b"], [0.5, 0.2], seed=0)


def test_repeated_setup_keeps_the_same_partition(monkeypatch):
    """
    Regression: setup() is called once per stage, and re-splitting there put training
    images into the test split.
    """
    monkeypatch.setattr("illust_salmap.dataset.cat2000.Cat2000Dataset", FakeCat2000Dataset)

    datamodule = Cat2000(root="unused", batch_size=4, num_workers=0)

    datamodule.setup("fit")
    train_indices = list(datamodule.train.indices)
    val_indices = list(datamodule.val.indices)

    datamodule.setup("test")

    assert list(datamodule.train.indices) == train_indices
    assert list(datamodule.val.indices) == val_indices


def test_test_split_never_overlaps_training(monkeypatch):
    monkeypatch.setattr("illust_salmap.dataset.cat2000.Cat2000Dataset", FakeCat2000Dataset)

    datamodule = Cat2000(root="unused", batch_size=4, num_workers=0)
    datamodule.setup("fit")
    datamodule.setup("test")

    training = set(datamodule.train.indices)
    testing = set(datamodule.test_dataloader().dataset.indices)

    assert not training & testing
