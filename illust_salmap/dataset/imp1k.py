import os

from PIL import Image
from matplotlib import pyplot
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms.v2 import Compose, Grayscale, Normalize, Resize, ToTensor, Transform

from illust_salmap.dataset.pairing import paired_paths
from illust_salmap.dataset.split import stratified_indices
from illust_salmap.installer import DatasetInstaller, HttpDownloader
from illust_salmap.dataset.stats import calculate_mean_std


class Imp1kCategories:
    ads = "ads"
    infographics = "infographics"
    movie_posters = "movie_posters"
    webpages = "webpages"
    all = [ads, infographics, movie_posters, webpages]


class Imp1kDataset(Dataset):
    URL = "https://predimportance.mit.edu/data/imp1k.zip"

    def __init__(self,
                 root,
                 categories=None,
                 image_transform=None,
                 map_transform=None
                 ):

        self.categories = categories or Imp1kCategories.all

        self.image_transform = image_transform
        self.map_transform = map_transform

        print(f"url: {self.URL}")

        self.data_dir = DatasetInstaller(root=f"{root}/imp1k", downloader=HttpDownloader(self.URL)).install()

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.pair_categories = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):

        for category in self.categories:
            images_dir = self.data_dir / "imgs" / category
            maps_dir = self.data_dir / "maps" / category
            pairs = paired_paths(images_dir.glob("*"), maps_dir.glob("*"), f"imp1k/{category}")

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


class Imp1k(LightningDataModule):
    """
    Imp1k.

    The dataset ships no canonical split, so the images are divided 80/20 here,
    stratified over the four design categories. `test_dataloader` serves the validation
    split -- there is no separate held-out set.
    """

    VAL_RATIO = 0.2

    def __init__(self, root: str = "./data", batch_size: int = 64, num_workers: int = os.cpu_count(),
                 img_size=(256, 256), val_ratio: float = VAL_RATIO, seed: int = 42):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        self.train = None
        self.val = None

        # データ変換
        self.image_transform = Compose([
            Resize(img_size),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        self.map_transform = Compose([
            Resize(img_size),
            Grayscale(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

    def prepare_data(self):
        Imp1kDataset(self.root)

    def setup(self, stage: str = None):
        # Lightning calls this once per stage; split only on the first call so `fit` and
        # `test` see the same partition.
        if self.train is not None:
            return

        imp1k = Imp1kDataset(self.root, map_transform=self.map_transform, image_transform=self.image_transform)

        train_indices, val_indices = stratified_indices(
            imp1k.pair_categories, [1 - self.val_ratio, self.val_ratio], self.seed
        )

        self.train = Subset(imp1k, train_indices)
        self.val = Subset(imp1k, val_indices)

    def train_dataloader(self) -> DataLoader:
        # shuffle is not optional here: `stratified_indices` returns sorted indices, so the
        # samples arrive grouped by design category. Without it every batch is drawn from a
        # single category and never changes between epochs.
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
        """Serves the validation split: Imp1k defines no held-out test set."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


class PadToSquare(Transform):
    def __init__(self, fill=0):
        super().__init__()
        self.fill = fill  # パディングの色（デフォルトは黒）

    def __call__(self, image):
        # 画像のサイズを取得
        width, height = image.size

        # 最長辺の長さを取得して正方形にするための新しいサイズを決定
        size = max(width, height)

        # 左右と上下に追加するパディングの量を計算
        padding_left = (size - width) // 2
        padding_top = (size - height) // 2
        padding_right = size - width - padding_left
        padding_bottom = size - height - padding_top

        # パディングを加えて正方形にする
        return transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom),
                                         fill=self.fill)


if __name__ == '__main__':
    pad_and_resize = Compose([
        Resize(256),
        PadToSquare()
    ])

    dataset = Imp1kDataset("./data", image_transform=pad_and_resize, map_transform=pad_and_resize)
    image, label = next(iter(dataset))

    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].set_axis_off()

    axes[1].imshow(label)
    axes[1].set_title("label")
    axes[1].set_axis_off()
    fig.show()

    dataset = Imp1kDataset("./data", image_transform=ToTensor(), map_transform=ToTensor())
    calculate_mean_std(dataset)
