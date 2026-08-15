from typing import Sequence

import torch


def stratified_indices(groups: Sequence[str], ratios: Sequence[float], seed: int) -> list[list[int]]:
    """
    Splits sample indices so that every group keeps the same proportion in each split.

    CAT2000 and Imp1k are built around their categories -- CAT2000 divides train and
    test 100/100 within every one of its 20 categories -- so a plain random split would
    let category balance drift between splits. Stratifying keeps the composition of each
    split comparable, and seeding it keeps the split identical across `setup()` calls.

    Args:
        groups (Sequence[str]): The group (category) of each sample, in index order.
        ratios (Sequence[float]): Split sizes as fractions, summing to 1.
        seed (int): Seed for the shuffling, so the partition is reproducible.

    Returns:
        list[list[int]]: One sorted index list per ratio. Every index appears exactly once.
    """
    if not ratios or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)

    splits: list[list[int]] = [[] for _ in ratios]

    for group in sorted(by_group):
        members = by_group[group]
        generator = torch.Generator().manual_seed(seed + hash_group(group))
        shuffled = [members[i] for i in torch.randperm(len(members), generator=generator).tolist()]

        # Hand out whole counts first, then give the rounding remainder to the largest
        # split. A category with fewer members than the number of splits may therefore
        # be missing from a small split, but it is never dropped entirely.
        counts = [int(len(shuffled) * ratio) for ratio in ratios]
        counts[ratios.index(max(ratios))] += len(shuffled) - sum(counts)

        start = 0
        for split, count in zip(splits, counts):
            split.extend(shuffled[start:start + count])
            start += count

    return [sorted(split) for split in splits]


def hash_group(group: str) -> int:
    """A stable per-group offset. `hash()` is salted per process, so it cannot be used."""
    return sum(ord(character) * (index + 1) for index, character in enumerate(group))
