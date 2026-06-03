"""GDAL ``/vsizip/`` + ``/vsitar/`` path builders and archive member discovery.

The clipped Landsat scenes are ``.tar`` (flat, ``..._B4.TIF``); Sentinel-2 / -1 are
``.zip`` SAFE trees (``IMG_DATA/..._B04.jp2``; ``measurement/...-vv-....tiff``). The
renderers open bands *inside* these archives via GDAL virtual file paths — no
extraction to disk — so a band read can still be decimated (overview/``out_shape``)
and never materialises the full ~146 MB S1 measurement TIFF.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def list_tar_members(archive: Path) -> list[str]:
    """Return the member names inside a ``.tar`` archive."""
    with tarfile.open(archive) as tar:
        return tar.getnames()


def list_zip_members(archive: Path) -> list[str]:
    """Return the member names inside a ``.zip`` archive."""
    with zipfile.ZipFile(archive) as zf:
        return zf.namelist()


def vsitar_path(archive: Path, member: str) -> str:
    """Build a GDAL ``/vsitar/`` path for ``member`` inside ``archive``."""
    return f"/vsitar/{archive.resolve()}/{member}"


def vsizip_path(archive: Path, member: str) -> str:
    """Build a GDAL ``/vsizip/`` path for ``member`` inside ``archive``."""
    return f"/vsizip/{archive.resolve()}/{member}"


def find_member(members: list[str], *, suffix: str, contains: str | None = None) -> str:
    """Return the single member ending in ``suffix`` (optionally containing ``contains``).

    Args:
        members: Archive member names.
        suffix: Required filename suffix (e.g. ``"_B04.jp2"``).
        contains: Optional substring the member must also contain (e.g.
            ``"IMG_DATA"`` to exclude QI-data masks of the same band name).

    Returns:
        The first matching member name.

    Raises:
        FileNotFoundError: If no member matches.
    """
    for name in members:
        if name.endswith(suffix) and (contains is None or contains in name):
            return name
    raise FileNotFoundError(f"no member matching suffix={suffix!r} contains={contains!r}")
