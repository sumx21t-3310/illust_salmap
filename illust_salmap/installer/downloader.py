from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import gdown
import requests
from tqdm import tqdm

KIB = 2 ** 10


class DownloadError(RuntimeError):
    """Raised when a remote archive could not be retrieved."""


class Downloader(ABC):
    """
    Fetches a single remote archive to a local path.

    Subclasses implement `_fetch` for one transport; everything else (partial file
    handling, destination setup) is shared. A Downloader knows nothing about
    extraction or about whether the file is already installed.
    """

    @property
    @abstractmethod
    def filename(self) -> str:
        """The name the archive should be saved under."""

    @abstractmethod
    def _fetch(self, dest: Path) -> None:
        """Writes the remote content to `dest`. Raises DownloadError on failure."""

    def fetch(self, dest: Path) -> None:
        """
        Downloads the archive to `dest`.

        The transfer goes to a `.part` sibling first, so an interrupted download never
        leaves a truncated file that later looks like a complete one.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(f"{dest.name}.part")
        partial.unlink(missing_ok=True)

        try:
            self._fetch(partial)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

        partial.replace(dest)


class HttpDownloader(Downloader):
    """Streams an archive over plain HTTP(S)."""

    def __init__(self, url: str, filename: str = None):
        self.url = url
        self._filename = filename or Path(urlparse(url).path).name

    @property
    def filename(self) -> str:
        return self._filename

    def _fetch(self, dest: Path) -> None:
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
        except requests.RequestException as err:
            raise DownloadError(f"Failed to download {self.url}: {err}") from err

        total = int(response.headers.get("content-length", 0))

        with dest.open("wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1000) as progress:
                for chunk in response.iter_content(chunk_size=100 * KIB):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))

    def __str__(self) -> str:
        return self.url


class GoogleDriveDownloader(Downloader):
    """
    Downloads an archive from Google Drive via `gdown`.

    Drive URLs carry no usable filename, so one must be supplied explicitly.
    """

    def __init__(self, file_id: str, filename: str):
        self.file_id = file_id
        self.url = f"https://drive.google.com/uc?export=download&id={file_id}"
        self._filename = filename

    @property
    def filename(self) -> str:
        return self._filename

    def _fetch(self, dest: Path) -> None:
        # gdown reports failure by returning None rather than raising.
        if gdown.download(self.url, str(dest), quiet=False) is None:
            raise DownloadError(f"gdown could not download {self.url}")

    def __str__(self) -> str:
        return self.url
