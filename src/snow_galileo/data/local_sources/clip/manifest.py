"""The per-source clip manifest (CLIPPING_PLAN §2.0 contract).

One row per input product, recording the gate decision and measured overlap.
This manifest is the audit artifact Phase 0 consumes and the proof the clip
stage created no zero-signal files.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .gate import ClipAction

MANIFEST_COLUMNS = [
    "product_id",
    "source",
    "footprint_bbox",
    "intersects",
    "aoi_overlap_km2",
    "valid_pixel_count",
    "action",
    "output_path",
]


@dataclass
class ManifestRow:
    """A single clip-manifest record.

    Attributes:
        product_id: Stem of the input product (e.g. the tar/zip/hdf filename).
        source: Modality directory name (``dem``, ``landsat9``, ``modis`` …).
        footprint_bbox: ``"lon_min,lat_min,lon_max,lat_max"`` of the footprint,
            or empty when the footprint could not be read.
        intersects: Whether the footprint intersects the AOI.
        aoi_overlap_km2: Area of footprint∩AOI in km².
        valid_pixel_count: Count of non-nodata pixels in the clipped output
            (0 for skips and for products whose pixels were not counted).
        action: One of ``CLIP``, ``SKIP_NO_OVERLAP``, ``SKIP_DEGENERATE_OVERLAP``.
        output_path: Relative path of the written output, or empty on skip.
    """

    product_id: str
    source: str
    footprint_bbox: str
    intersects: bool
    aoi_overlap_km2: float
    valid_pixel_count: int
    action: ClipAction
    output_path: str = ""

    def as_csv_dict(self) -> dict[str, object]:
        """Return a CSV-serialisable mapping (enum rendered as its value)."""
        row = asdict(self)
        row["action"] = self.action.value
        return row


def bbox_str(bounds: Optional[tuple[float, float, float, float]]) -> str:
    """Format a ``(lon_min, lat_min, lon_max, lat_max)`` tuple for the manifest."""
    if bounds is None:
        return ""
    return ",".join(f"{c:.6f}" for c in bounds)


def write_manifest(rows: list[ManifestRow], manifest_path: Path) -> None:
    """Write manifest rows to CSV, creating parent directories as needed.

    Args:
        rows: One :class:`ManifestRow` per input product.
        manifest_path: Destination CSV path.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_dict())


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read a manifest CSV back into a list of string-valued dict rows."""
    with manifest_path.open(newline="") as fh:
        return list(csv.DictReader(fh))
