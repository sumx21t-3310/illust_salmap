import io
import zipfile
from pathlib import Path

import pytest

from illust_salmap.installer import DatasetInstaller, DownloadError, Downloader


def make_zip(entries: dict) -> bytes:
    """Builds an in-memory ZIP archive from a {path: content} mapping."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zip_ref:
        for name, content in entries.items():
            zip_ref.writestr(name, content)
    return buffer.getvalue()


class FakeDownloader(Downloader):
    """
    Serves prepared payloads instead of hitting the network.

    Each fetch consumes the next payload; the last one repeats. A payload that is an
    exception is raised instead of written.
    """

    def __init__(self, filename: str, *payloads):
        self._filename = filename
        self._payloads = list(payloads)
        self.calls = 0

    @property
    def filename(self) -> str:
        return self._filename

    def _fetch(self, dest: Path) -> None:
        self.calls += 1
        payload = self._payloads.pop(0) if len(self._payloads) > 1 else self._payloads[0]

        if isinstance(payload, Exception):
            raise payload

        dest.write_bytes(payload)


def installer(tmp_path: Path, downloader: FakeDownloader, **kwargs) -> DatasetInstaller:
    return DatasetInstaller(root=tmp_path / "dataset", downloader=downloader, retry_delay=0, **kwargs)


def test_wrapper_directory_becomes_the_content_root(tmp_path):
    """An archive whose entries all sit under one directory reports that directory."""
    archive = make_zip({"trainSet/Stimuli/a.jpg": "a", "trainSet/FIXATIONMAPS/a.jpg": "m"})

    content_dir = installer(tmp_path, FakeDownloader("trainSet.zip", archive)).install()

    assert content_dir == tmp_path / "dataset" / "trainSet"
    assert (content_dir / "Stimuli" / "a.jpg").exists()


def test_content_root_is_independent_of_the_archive_name(tmp_path):
    """The wrapper directory wins even when it does not match the downloaded filename."""
    archive = make_zip({"trainSet/a.jpg": "a"})

    content_dir = installer(tmp_path, FakeDownloader("download.zip", archive)).install()

    assert content_dir == tmp_path / "dataset" / "trainSet"
    assert (content_dir / "a.jpg").exists()


def test_loose_entries_are_unpacked_into_a_named_directory(tmp_path):
    """Without a single wrapper directory, the archive stem keeps the files together."""
    archive = make_zip({"imgs/a.jpg": "a", "maps/a.jpg": "m"})

    content_dir = installer(tmp_path, FakeDownloader("imp1k.zip", archive)).install()

    assert content_dir == tmp_path / "dataset" / "imp1k"
    assert (content_dir / "imgs" / "a.jpg").exists()


def test_single_top_level_file_is_not_treated_as_a_wrapper(tmp_path):
    archive = make_zip({"only.txt": "x"})

    content_dir = installer(tmp_path, FakeDownloader("pack.zip", archive)).install()

    assert content_dir == tmp_path / "dataset" / "pack"
    assert (content_dir / "only.txt").exists()


def test_reinstall_returns_the_same_path_without_refetching(tmp_path):
    """Regression: the second run used to recompute the content root from the archive name."""
    downloader = FakeDownloader("download.zip", make_zip({"trainSet/a.jpg": "a"}))

    first = installer(tmp_path, downloader).install()
    second = installer(tmp_path, downloader).install()

    assert second == first
    assert downloader.calls == 1


def test_reinstall_recovers_when_the_extracted_data_is_deleted(tmp_path):
    downloader = FakeDownloader("download.zip", make_zip({"trainSet/a.jpg": "a"}))

    first = installer(tmp_path, downloader).install()
    (first / "a.jpg").unlink()
    first.rmdir()

    second = installer(tmp_path, downloader).install()

    assert second == first
    assert (second / "a.jpg").exists()


def test_corrupt_archive_is_discarded_and_refetched(tmp_path):
    """Regression: the retry loop deleted the archive but never downloaded it again."""
    downloader = FakeDownloader("data.zip", b"not a zip at all", make_zip({"data/a.jpg": "a"}))

    content_dir = installer(tmp_path, downloader).install()

    assert downloader.calls == 2
    assert (content_dir / "a.jpg").exists()


def test_failed_download_leaves_nothing_behind(tmp_path):
    """Regression: a partial file used to satisfy the 'already downloaded' check."""
    downloader = FakeDownloader("data.zip", DownloadError("boom"))
    target = installer(tmp_path, downloader)

    with pytest.raises(DownloadError):
        target.install()

    assert downloader.calls == 3
    assert not target.archive_path.exists()
    assert list(target.root.glob("*.part")) == []


def test_interrupted_transfer_cleans_up_its_partial_file(tmp_path):
    class TruncatingDownloader(FakeDownloader):
        def _fetch(self, dest: Path) -> None:
            self.calls += 1
            dest.write_bytes(b"half of a zip")
            raise DownloadError("connection reset")

    downloader = TruncatingDownloader("data.zip", b"")
    target = installer(tmp_path, downloader)

    with pytest.raises(DownloadError):
        target.install()

    assert not target.archive_path.exists()
    assert list(target.root.glob("*.part")) == []


def test_download_is_retried_before_giving_up(tmp_path):
    downloader = FakeDownloader("data.zip", DownloadError("flaky"), make_zip({"data/a.jpg": "a"}))

    content_dir = installer(tmp_path, downloader).install()

    assert downloader.calls == 2
    assert (content_dir / "a.jpg").exists()


def test_redownload_forces_a_fresh_fetch(tmp_path):
    downloader = FakeDownloader("download.zip", make_zip({"trainSet/a.jpg": "a"}))

    installer(tmp_path, downloader).install()
    installer(tmp_path, downloader, redownload=True).install()

    assert downloader.calls == 2
