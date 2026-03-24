"""Archive extraction tools."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from .logger import get_logger

if TYPE_CHECKING:
    import tarfile
    import zipfile

logger = get_logger(__name__)


def safe_zip_extract_all(zip_ref: zipfile.ZipFile, output_dir: str | Path) -> None:
    """Extract all members from a zip archive, even if they are nested in a folder.

    Python's ``zipfile`` extraction path does not restore archived modification
    times, so extracted members keep the filesystem write time from extraction.

    Args:
        zip_ref (zipfile.ZipFile): Open zip archive to extract.
        output_dir (str | Path): Directory where members are extracted.

    """
    output_dir = Path(output_dir)
    safe_dest_dir = output_dir.resolve()
    for member in zip_ref.infolist():
        member_path = safe_dest_dir / member.filename
        if not member_path.resolve().as_posix().startswith(safe_dest_dir.as_posix()):
            logger.warning("Skipping potentially unsafe member: %s", member.filename)
            continue
        zip_ref.extract(member, path=output_dir)


def safe_tar_extract_all(tar_ref: tarfile.TarFile, output_dir: str | Path, preserve_mtime: bool = False) -> None:
    """Extract all members from a tar archive after path validation.

    Args:
        tar_ref (tarfile.TarFile): Open tar archive to extract.
        output_dir (str | Path): Directory where members are extracted.
        preserve_mtime (bool, optional): Whether to restore archived
            modification times on extracted members. If ``False``, extracted
            members keep the filesystem write time from extraction, similar to
            ``tar -m``. Default: True.

    """
    output_dir = Path(output_dir)
    safe_dest_dir = output_dir.resolve()
    for member in tar_ref.getmembers():
        member_path = safe_dest_dir / member.name
        if not member_path.resolve().as_posix().startswith(safe_dest_dir.as_posix()):
            logger.warning("Skipping potentially unsafe member: %s", member.name)
            continue
        member_to_extract = member
        if not preserve_mtime:
            member_to_extract = copy.copy(member)
            member_to_extract.mtime = None
        tar_ref.extract(member_to_extract, path=output_dir)
