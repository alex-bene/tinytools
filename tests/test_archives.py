"""Tests for archive extraction helpers."""

from __future__ import annotations

import io
import stat
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

from src.tinytools.archives import safe_tar_extract_all, safe_zip_extract_all


def _create_tar_bytes(*, name: str, data: bytes, mtime: int, mode: int = 0o644) -> bytes:
    """Build an in-memory tar archive for testing.

    Args:
        name (str): Member path inside the tar archive.
        data (bytes): Member payload.
        mtime (int): Stored member modification time as a POSIX timestamp.
        mode (int, optional): Stored file mode. Default: 0o644.

    Returns:
        bytes: Tar archive bytes containing a single regular file.

    """
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar_ref:
        tar_info = tarfile.TarInfo(name=name)
        tar_info.size = len(data)
        tar_info.mtime = mtime
        tar_info.mode = mode
        tar_ref.addfile(tar_info, io.BytesIO(data))
    return tar_stream.getvalue()


def _create_zip_bytes(*, name: str, data: bytes, date_time: tuple[int, int, int, int, int, int]) -> bytes:
    """Build an in-memory zip archive for testing.

    Args:
        name (str): Member path inside the zip archive.
        data (bytes): Member payload.
        date_time (tuple[int, int, int, int, int, int]): Stored DOS timestamp
            as ``(Y, M, D, H, M, S)``.

    Returns:
        bytes: Zip archive bytes containing a single regular file.

    """
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, mode="w") as zip_ref:
        zip_info = zipfile.ZipInfo(filename=name, date_time=date_time)
        zip_ref.writestr(zip_info, data)
    return zip_stream.getvalue()


def test_safe_tar_extract_all_preserves_archived_mtime_by_default() -> None:
    """Default tar extraction should restore the stored member mtime."""
    archived_mtime = 946684800  # 2000-01-01 00:00:00 UTC
    tar_bytes = _create_tar_bytes(name="payload.txt", data=b"payload", mtime=archived_mtime)

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "payload.tar"
        output_dir = Path(tmp_dir) / "out"
        archive_path.write_bytes(tar_bytes)
        output_dir.mkdir()

        with tarfile.open(archive_path) as tar_ref:
            safe_tar_extract_all(tar_ref, output_dir, preserve_mtime=True)

        extracted_path = output_dir / "payload.txt"
        assert extracted_path.exists()
        assert int(extracted_path.stat().st_mtime) == archived_mtime


def test_safe_tar_extract_all_can_use_extraction_time() -> None:
    """Tar extraction can skip restoring the stored member mtime."""
    archived_mtime = 946684800  # 2000-01-01 00:00:00 UTC
    tar_bytes = _create_tar_bytes(name="bin/script.sh", data=b"#!/bin/sh\nexit 0\n", mtime=archived_mtime, mode=0o755)

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "payload.tar"
        output_dir = Path(tmp_dir) / "out"
        archive_path.write_bytes(tar_bytes)
        output_dir.mkdir()

        extraction_start = time.time()
        with tarfile.open(archive_path) as tar_ref:
            safe_tar_extract_all(tar_ref, output_dir, preserve_mtime=False)
        extraction_end = time.time()

        extracted_path = output_dir / "bin" / "script.sh"
        extracted_stat = extracted_path.stat()
        assert extracted_path.exists()
        assert extraction_start - 1 <= extracted_stat.st_mtime <= extraction_end + 1
        assert int(extracted_stat.st_mtime) != archived_mtime
        assert stat.S_IMODE(extracted_stat.st_mode) == 0o755


def test_safe_zip_extract_all_keeps_extraction_time_mtime() -> None:
    """Zip extraction should keep the filesystem write time."""
    zip_bytes = _create_zip_bytes(name="payload.txt", data=b"payload", date_time=(2000, 1, 1, 0, 0, 0))

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "payload.zip"
        output_dir = Path(tmp_dir) / "out"
        archive_path.write_bytes(zip_bytes)
        output_dir.mkdir()

        extraction_start = time.time()
        with zipfile.ZipFile(archive_path) as zip_ref:
            safe_zip_extract_all(zip_ref, output_dir)
        extraction_end = time.time()

        extracted_path = output_dir / "payload.txt"
        extracted_mtime = extracted_path.stat().st_mtime
        assert extracted_path.exists()
        assert extraction_start - 1 <= extracted_mtime <= extraction_end + 1
        assert int(extracted_mtime) != 946684800
