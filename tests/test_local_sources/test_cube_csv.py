"""Generated cube CSV tests (TASK-001, SPEC AC-11b).

Asserts the canonical 8-column schema, the full cross-product row count, the
``EPSG:32611`` CRS column, and that the column set is exactly what
``EarthEngineExporterEval.export_from_csv_utm`` reads (``eo_eval.py:577-585``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from snow_galileo.data.local_sources.grid import (
    CUBE_CSV_COLUMNS,
    GRID_MATH_CRS,
    build_cube_csv,
    filter_cells,
    generate,
    load_aoi_polygon,
    load_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CSV = REPO_ROOT / "tests/fixtures/sampled_cells_bow_river_with_dates.csv"
AOI_PATH = REPO_ROOT / "data" / "bow_valley_inference_aoi.geojson"

EXPECTED_CENTRE_IN = 344
DEFAULT_WINDOW_DAYS = 53  # 2025-04-06 .. 2025-05-28 inclusive

# The exact columns export_from_csv_utm pulls via df["..."] at eo_eval.py:577-585.
GEE_REQUIRED_COLUMNS = {
    "date",
    "crs",
    "center_x",
    "center_y",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
}


@pytest.fixture(scope="module")
def kept_cells():
    aoi = load_aoi_polygon(AOI_PATH)
    cells = load_cells(LEGACY_CSV)
    kept, _ = filter_cells(cells, aoi, keep_rule="centre_in")
    return kept


def test_schema_is_canonical(kept_cells):
    """CSV has exactly the canonical 8 columns in order (SPEC AC-11b)."""
    df = build_cube_csv(kept_cells)
    assert list(df.columns) == CUBE_CSV_COLUMNS


def test_row_count_is_full_cross_product(kept_cells):
    """Row count == kept cells × window days (SPEC AC-11b)."""
    df = build_cube_csv(kept_cells)
    assert len(df) == EXPECTED_CENTRE_IN * DEFAULT_WINDOW_DAYS == 18232


def test_crs_column_is_utm11n(kept_cells):
    """Every row carries crs == EPSG:32611 (SPEC AC-11b)."""
    df = build_cube_csv(kept_cells)
    assert (df["crs"] == GRID_MATH_CRS).all()


def test_dates_span_full_window(kept_cells):
    """`date` covers every day in the window, each appearing for every cell."""
    df = build_cube_csv(kept_cells)
    unique_dates = sorted(df["date"].unique())
    assert len(unique_dates) == DEFAULT_WINDOW_DAYS
    assert unique_dates[0] == int(date(2025, 4, 6).strftime("%Y%m%d"))
    assert unique_dates[-1] == int(date(2025, 5, 28).strftime("%Y%m%d"))
    # each date appears once per kept cell
    counts = df["date"].value_counts()
    assert (counts == EXPECTED_CENTRE_IN).all()


def test_gee_exporter_column_contract(kept_cells):
    """The CSV satisfies export_from_csv_utm's column reads (SPEC AC-11b)."""
    df = build_cube_csv(kept_cells)
    assert GEE_REQUIRED_COLUMNS.issubset(set(df.columns))
    # date parses as YYYYMMDD (the exporter does strptime(str(date), "%Y%m%d")).
    from datetime import datetime

    for value in df["date"].unique():
        datetime.strptime(str(value), "%Y%m%d")


def test_filename_matches_gee_pattern(kept_cells):
    """The exporter builds PR_{date}_{cx:.16f}_{cy:.16f}.tif from these columns.

    Reproduces the eo_eval.py:599 filename build and asserts it parses through
    the ``PR`` branch of ``LandsatEvalDataset`` (month at parts[1][4:6]).
    """
    df = build_cube_csv(kept_cells).iloc[0]
    filename = f"PR_{df['date']}_{df['center_x']:.16f}_{df['center_y']:.16f}.tif"
    parts = filename.split("_")
    assert parts[0] == "PR"
    assert parts[1][4:6] == "04"  # month of 2025-04-06 window start day


def test_generate_writes_files(kept_cells, tmp_path):
    """`generate` writes both the cube CSV and the manifest, round-trippable."""
    out_csv = tmp_path / "cube_cells.csv"
    manifest = tmp_path / "manifest.csv"
    df = generate(
        legacy_csv=LEGACY_CSV,
        aoi_path=AOI_PATH,
        output_csv=out_csv,
        manifest_path=manifest,
    )
    assert out_csv.exists()
    assert manifest.exists()
    reread = pd.read_csv(out_csv)
    assert list(reread.columns) == CUBE_CSV_COLUMNS
    assert len(reread) == len(df)
    # manifest accounts for all 500 cells
    man = pd.read_csv(manifest)
    assert len(man) == 500
