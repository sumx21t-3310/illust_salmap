import os
from typing import Optional, Callable

from PIL import Image
from matplotlib import pyplot
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.v2 import Resize, Compose, ToTensor, Normalize, Grayscale

from illust_salmap.dataset.pairing import paired_paths
from illust_salmap.dataset.split import stratified_indices
from illust_salmap.installer import DatasetInstaller, HttpDownloader
from illust_salmap.dataset.stats import calculate_mean_std


class Cat2000Dataset(Dataset):
    URL = "http://saliency.mit.edu/trainSet.zip"

    def __init__(self,
                 root: str,
                 categories: Optional[list[str]] = None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None):
        self.categories = categories or ["*"]  # None の場合デフォルトで全カテゴリ
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.data_dir = DatasetInstaller(root=f"{root}/cat2000", downloader=HttpDownloader(self.URL)).install()

        self.image_map_pair_cache = []
        self.pair_categories = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        stimuli_path = self.data_dir / "Stimuli"
        fixation_path = self.data_dir / "FIXATIONMAPS"

        # カテゴリの展開（"*" を含む場合は全カテゴリ）
        if "*" in self.categories:
            expanded_categories = [p.name for p in stimuli_path.iterdir() if p.is_dir()]
        else:
            expanded_categories = [category for category in self.categories]

        expanded_categories.sort()

        for category in expanded_categories:
            pairs = paired_paths(
                (stimuli_path / category).glob("???.jpg"),
                (fixation_path / category).glob("???.jpg"),
                f"cat2000/{category}",
            )

            self.image_map_pair_cache.extend(pairs)
            # Kept alongside the pairs so the split can stay category-balanced.
            self.pair_categories.extend([category] * len(pairs))

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, index: int):
        image_path, map_path = self.image_map_pair_cache[index]

        image = Image.open(image_path).convert("RGB")
        map_image = Image.open(map_path).convert("L")

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.map_transform is not None:
            map_image = self.map_transform(map_image)

        return image, map_image


class Cat2000(LightningDataModule):
    """
    CAT2000, as far as the public data allows.

    The paper divides 4000 images into 2000 train and 2000 test, 100 per category on
    each side, and holds back every fixation of the test half -- those are only scored
    by the MIT/Tuebingen benchmark server. `trainSet.zip` is therefore the whole of what
    can be used locally, and this module splits it the way the literature does: 1800 for
    training and 200 for validation, stratified over the 20 categories.

    Since the real test set is unavailable, `test_dataloader` serves the validation
    split. Numbers it produces describe held-out validation data, not the benchmark.
    """

    VAL_RATIO = 0.1

    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers: int = os.cpu_count(),
                 img_size=(216, 384), image_transform=None, map_transform=None,
                 val_ratio: float = VAL_RATIO, seed: int = 42):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        self.train = None
        self.val = None

        self.image_transform = image_transform or Compose([
            Resize(img_size),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

        self.map_transform = map_transform or Compose([
            Grayscale(),
            Resize(img_size),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])
# 
    def prepare_data(self):
        Cat2000Dataset(self.root)

    def setup(self, stage: str = None):
        # Lightning calls this once per stage. Splitting only on the first call keeps
        # `fit` and `test` looking at the same partition instead of redrawing it.
        if self.train is not None:
            return

        cat2000 = Cat2000Dataset(self.root, map_transform=self.map_transform, image_transform=self.image_transform)

        train_indices, val_indices = stratified_indices(
            cat2000.pair_categories, [1 - self.val_ratio, self.val_ratio], self.seed
        )

        self.train = Subset(cat2000, train_indices)
        self.val = Subset(cat2000, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Serves the validation split: CAT2000's real test fixations are not public."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def __str__(self):
        return f"{self.__class__.__name__}(dataset={Cat2000Dataset.__name__})"


if __name__ == '__main__':
    dataset = Cat2000Dataset("./data")
    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()

    fig.show()

    dataset = Cat2000Dataset("./", image_transform=ToTensor(), map_transform=ToTensor())
    calculate_mean_std(dataset)
