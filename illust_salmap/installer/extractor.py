from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional
from zipfile import BadZipFile, ZipFile

from tqdm import tqdm


class CorruptArchiveError(Exception):
    """Raised when an archive exists but cannot be read."""


class Extractor(ABC):
    """
    Unpacks an archive and reports where its contents ended up.

    `extract` returns the directory that actually holds the data, so callers never have
    to guess it from the archive name.
    """

    @abstractmethod
    def extract(self, archive: Path, dest_root: Path) -> Path:
        """
        Extracts `archive` under `dest_root` and returns the directory holding its contents.

        Raises:
            CorruptArchiveError: If the archive cannot be read.
        """


class ZipExtractor(Extractor):
    """
    Extracts ZIP archives.

    An archive that wraps everything in a single top-level directory is unpacked
    directly into `dest_root`, so that directory becomes the content root. Anything
    else is unpacked into `dest_root/<archive stem>` to avoid scattering loose files.
    """

    def extract(self, archive: Path, dest_root: Path) -> Path:
        try:
            with ZipFile(archive, "r") as zip_ref:
                names = zip_ref.namelist()
                wrapper = self._wrapper_dir(names)
                destination = dest_root if wrapper else dest_root / archive.stem

                with tqdm(total=len(names), unit="file") as progress:
                    for name in names:
                        zip_ref.extract(name, destination)
                        progress.update(1)
        except BadZipFile as err:
            raise CorruptArchiveError(f"{archive} is not a readable ZIP file: {err}") from err

        return dest_root / wrapper if wrapper else destination

    @staticmethod
    def _wrapper_dir(names: Iterable[str]) -> Optional[str]:
        """Returns the archive's sole top-level directory, or None if there isn't exactly one."""
        parts = [Path(name).parts for name in names]
        roots = {part[0] for part in parts}

        if len(roots) != 1:
            return None

        # A single top-level *file* is not a wrapper directory.
        if not any(len(part) > 1 for part in parts):
            return None

        return next(iter(roots))
