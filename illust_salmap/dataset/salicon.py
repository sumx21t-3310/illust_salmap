import os
from typing import Optional, Callable

from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.v2 import Normalize, ToTensor, Compose, Resize, Grayscale

from illust_salmap.dataset.pairing import paired_paths
from illust_salmap.dataset.split import stratified_indices
from illust_salmap.dataset.stats import calculate_mean_std
from illust_salmap.installer import DatasetInstaller, GoogleDriveDownloader, install_all
from matplotlib import pyplot


class SALICONDataset(Dataset):
    IMAGE_ID = r"1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5"
    MAPS_ID = r"1PnO7szbdub1559LfjYHMy65EDC4VhJC8"

    def __init__(self,
                 root: str,
                 categories=None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None
                 ):

        self.categories = categories or ["test", "train"]

        self.image_transform = image_transform
        self.map_transform = map_transform

        self.images_dir, self.maps_dir = install_all([
            DatasetInstaller(f"{root}/salicon", GoogleDriveDownloader(self.IMAGE_ID, "images.zip")),
            DatasetInstaller(f"{root}/salicon", GoogleDriveDownloader(self.MAPS_ID, "maps.zip")),
        ])

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.pair_categories = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        for category in self.categories:
            images_dir = self.images_dir / category
            maps_dir = self.maps_dir / category

            pairs = paired_paths(images_dir.glob("*.jpg"), maps_dir.glob("*.png"), f"salicon/{category}")

            self.image_map_pair_cache.extend(pairs)
            # The archive's own folders. Kept so the split preserves their proportions.
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


class SALICON(LightningDataModule):
    """
    SALICON.

    Note that this does NOT follow SALICON's official 10000/5000/5000 split: the archives
    downloaded here are a repackaging whose folder layout has not been verified against
    it. The images are split 80/20, stratified over the archive's own folders, and
    `test_dataloader` serves the validation split.
    """

    VAL_RATIO = 0.2

    def __init__(self, root: str = "./data", batch_size: int = 32, num_workers: int = os.cpu_count(),
                 img_size=(256, 384), val_ratio: float = VAL_RATIO, seed: int = 42):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_ratio = val_ratio
        self.seed = seed

        self.train = None
        self.val = None

        # データ変換
        self.image_transform = Compose([
            Resize(self.img_size),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

        self.map_transform = Compose([
            Resize(self.img_size),
            Grayscale(),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

    def prepare_data(self):
        SALICONDataset(self.root)

    def setup(self, stage: str = None):
        # Lightning calls this once per stage; split only on the first call so `fit` and
        # `test` see the same partition.
        if self.train is not None:
            return

        salicon = SALICONDataset(self.root, map_transform=self.map_transform, image_transform=self.image_transform)

        train_indices, val_indices = stratified_indices(
            salicon.pair_categories, [1 - self.val_ratio, self.val_ratio], self.seed
        )

        self.train = Subset(salicon, train_indices)
        self.val = Subset(salicon, val_indices)

    def train_dataloader(self) -> DataLoader:
        # shuffle is not optional here: `stratified_indices` returns sorted indices, so the
        # samples arrive grouped by the archive folder they came from. Without it every
        # batch is drawn from a single category and never changes between epochs.
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=True,
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
        """Serves the validation split: no held-out test set is carved out here."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


if __name__ == '__main__':
    dataset = SALICONDataset("./data")

    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()
    fig.show()

    calculate_mean_std(SALICONDataset("./data", map_transform=ToTensor(), image_transform=ToTensor()))
