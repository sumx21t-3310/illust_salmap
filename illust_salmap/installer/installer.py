import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Formatter, Logger, StreamHandler
from pathlib import Path
from typing import Dict, List, Optional

from .downloader import DownloadError, Downloader
from .extractor import CorruptArchiveError, Extractor, ZipExtractor


def create_default_logger(instance: object = None) -> Logger:
    """
    Creates and returns a default logger.

    Args:
        instance (object, optional): An object instance to use in naming the logger.

    Returns:
        Logger: A configured logger instance.
    """
    if instance:
        logger_name = f"{instance.__class__.__name__}_{id(instance)}"
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger(__name__)

    logger.setLevel("INFO")

    if not logger.handlers:
        console_handler = StreamHandler()
        console_handler.setFormatter(Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(console_handler)

    return logger


class DatasetInstaller:
    """
    Installs one dataset archive into `root` and reports where its contents live.

    Combines a Downloader and an Extractor, and owns everything they deliberately do
    not: retries, recovery from a corrupt archive, and the record of what is already
    installed.

    Installation is recorded in a per-archive marker file next to the data, so a repeat
    run can return the content directory without opening (or even keeping) the archive.
    `install()` returns that directory -- callers should use the return value rather
    than reconstructing the path themselves.
    """

    def __init__(self,
                 root: str,
                 downloader: Downloader,
                 extractor: Extractor = None,
                 redownload: bool = False,
                 reextract: bool = False,
                 max_retries: int = 3,
                 retry_delay: int = 2,
                 logger: Logger = None,
                 ):
        """
        Initializes the DatasetInstaller.

        Args:
            root (str): The directory the archive is downloaded into and extracted under.
            downloader (Downloader): Fetches the archive.
            extractor (Extractor, optional): Unpacks the archive. Defaults to ZipExtractor.
            redownload (bool, optional): Fetch the archive again even if present. Defaults to False.
            reextract (bool, optional): Unpack again even if already installed. Defaults to False.
            max_retries (int, optional): Attempts per step. Defaults to 3.
            retry_delay (int, optional): Seconds between download attempts. Defaults to 2.
            logger (Logger, optional): A custom logger instance. Defaults to None.
        """
        self.root = Path(root).resolve()
        self.downloader = downloader
        self.extractor = extractor or ZipExtractor()

        self.redownload = redownload
        self.reextract = reextract

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logger or create_default_logger(self)

    @property
    def archive_path(self) -> Path:
        """The local path of the downloaded archive."""
        return self.root / self.downloader.filename

    @property
    def marker_path(self) -> Path:
        """The record of a completed install. Named per archive, since one root may hold several."""
        return self.root / f".{self.downloader.filename}.installed.json"

    @property
    def is_installed(self) -> bool:
        return self._read_marker() is not None

    def install(self) -> Path:
        """
        Ensures the dataset is present and returns the directory holding its contents.

        Raises:
            DownloadError: If the archive could not be fetched.
            CorruptArchiveError: If the archive could not be read after retries.
        """
        if not (self.redownload or self.reextract):
            installed = self._read_marker()
            if installed is not None:
                self.logger.info(f"{self.downloader.filename} is already installed at {installed}, skipping.")
                return installed

        self._download()
        content_dir = self._extract()
        self._write_marker(content_dir)

        return content_dir

    def _download(self) -> None:
        if self.archive_path.exists() and not self.redownload:
            self.logger.info(f"Archive already present at {self.archive_path}, skipping download.")
            return

        self.logger.info(f"Downloading {self.downloader} to {self.archive_path}...")

        for attempt in range(1, self.max_retries + 1):
            try:
                self.downloader.fetch(self.archive_path)
                self.logger.info(f"Downloaded successfully to {self.archive_path}.")
                return
            except (DownloadError, OSError) as err:
                self.logger.warning(f"Download attempt {attempt}/{self.max_retries} failed: {err}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise DownloadError(f"Failed to download {self.downloader} after {self.max_retries} attempts.")

    def _extract(self) -> Path:
        for attempt in range(1, self.max_retries + 1):
            self.logger.info(f"Extracting {self.archive_path}...")
            try:
                content_dir = self.extractor.extract(self.archive_path, self.root)
                self.logger.info(f"Extracted successfully to {content_dir}.")
                return content_dir
            except CorruptArchiveError as err:
                self.logger.warning(f"{err} Discarding it and fetching again.")
                self.archive_path.unlink(missing_ok=True)
                if attempt < self.max_retries:
                    self._download()

        raise CorruptArchiveError(f"Could not extract {self.downloader.filename} after {self.max_retries} attempts.")

    def _read_marker(self) -> Optional[Path]:
        """Returns the recorded content directory, or None if it is missing or stale."""
        if not self.marker_path.exists():
            return None

        try:
            record = json.loads(self.marker_path.read_text(encoding="utf-8"))
            content_dir = self.root / record["content_dir"]
        except (ValueError, KeyError, OSError):
            self.logger.warning(f"Ignoring unreadable install marker at {self.marker_path}.")
            return None

        return content_dir if content_dir.exists() else None

    def _write_marker(self, content_dir: Path) -> None:
        record = {
            "archive": self.downloader.filename,
            # Stored relative to root so the dataset directory stays movable.
            "content_dir": content_dir.relative_to(self.root).as_posix(),
        }
        self.marker_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def install_all(installers: List[DatasetInstaller], max_workers: int = 4) -> List[Path]:
    """
    Installs several datasets in parallel.

    Args:
        installers (List[DatasetInstaller]): The installers to run.
        max_workers (int): Number of parallel workers. Default is 4.

    Returns:
        List[Path]: The content directories, in the same order as `installers`.

    Raises:
        Exception: Propagates the first failure, since a partially installed dataset
            cannot be used anyway.
    """
    content_dirs: Dict[DatasetInstaller, Path] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(installer.install): installer for installer in installers}

        for future in as_completed(futures):
            installer = futures[future]
            content_dirs[installer] = future.result()

    return [content_dirs[installer] for installer in installers]
