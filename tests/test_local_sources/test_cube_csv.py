"""Generated cube CSV tests (TASK-001, SPEC AC-11b).

Asserts the canonical 8-column UTM schema (:func:`build_cube_dataframe`), the full
cross-product row count, the ``EPSG:32611`` CRS column, and — for the GEE UTM
reader dialect (:func:`build_cube_dataframe_for_gee_utm`) — that its column set is
exactly what ``EarthEngineExporterEval.export_from_csv_utm`` reads (``center_lat`` /
``center_lon``, in true degrees).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from snow_galileo.data.local_sources.grid import (
    CUBE_CSV_COLUMNS,
    GEE_UTM_CSV_COLUMNS,
    GRID_MATH_CRS,
    _filter_cells,
    build_cube_dataframe,
    build_cube_dataframe_for_gee_utm,
    generate,
    load_aoi_polygon,
    load_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CSV = REPO_ROOT / "tests/fixtures/sampled_cells_bow_river_with_dates.csv"
AOI_PATH = REPO_ROOT / "data" / "bow_valley_inference_aoi.geojson"

DEFAULT_WINDOW_START: date = date(2025, 4, 6)
DEFAULT_WINDOW_END: date = date(2025, 5, 28)

EXPECTED_CENTRE_IN = 344
DEFAULT_WINDOW_DAYS = 53  # 2025-04-06 .. 2025-05-28 inclusive

# The exact columns export_from_csv_utm pulls via df["..."] (eo_eval.py). It reads the
# centre as center_lat/center_lon (degrees) — NOT the canonical center_x/center_y.
GEE_REQUIRED_COLUMNS = {
    "date",
    "crs",
    "center_lat",
    "center_lon",
    "min_x",
    "max_x",
    "min_y",
    "max_y",
}


@pytest.fixture(scope="module")
def kept_cells():
    aoi = load_aoi_polygon(AOI_PATH)
    cells = load_cells(LEGACY_CSV)
    kept, _ = _filter_cells(cells, aoi, keep_rule="centre_in")
    return kept


def test_schema_is_canonical(kept_cells):
    """CSV has exactly the canonical 8 columns in order (SPEC AC-11b)."""
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert list(df.columns) == CUBE_CSV_COLUMNS


def test_row_count_is_full_cross_product(kept_cells):
    """Row count == kept cells × window days (SPEC AC-11b)."""
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert len(df) == EXPECTED_CENTRE_IN * DEFAULT_WINDOW_DAYS == 18232


def test_crs_column_is_utm11n(kept_cells):
    """Every row carries crs == EPSG:32611 (SPEC AC-11b)."""
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert (df["crs"] == GRID_MATH_CRS).all()


def test_dates_span_full_window(kept_cells):
    """`date` covers every day in the window, each appearing for every cell."""
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    unique_dates = sorted(df["date"].unique())
    assert len(unique_dates) == DEFAULT_WINDOW_DAYS
    assert unique_dates[0] == int(date(2025, 4, 6).strftime("%Y%m%d"))
    assert unique_dates[-1] == int(date(2025, 5, 28).strftime("%Y%m%d"))
    # each date appears once per kept cell
    counts = df["date"].value_counts()
    assert (counts == EXPECTED_CENTRE_IN).all()


def test_gee_utm_schema_is_reader_dialect(kept_cells):
    """The GEE-UTM adapter emits exactly the reader's dialect columns, in order."""
    df = build_cube_dataframe_for_gee_utm(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert list(df.columns) == GEE_UTM_CSV_COLUMNS


def test_canonical_csv_lacks_reader_centre_columns(kept_cells):
    """Regression: the canonical CSV does NOT carry the reader's centre columns.

    Feeding a canonical frame to export_from_csv_utm's degree-dialect path would
    KeyError — the reason build_cube_dataframe_for_gee_utm exists. Guards against anyone
    silently renaming build_cube_dataframe's columns back.
    """
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert "center_lat" not in df.columns
    assert "center_lon" not in df.columns
    assert {"center_x", "center_y"}.issubset(df.columns)


def test_gee_exporter_column_contract(kept_cells):
    """The GEE-UTM CSV satisfies export_from_csv_utm's column reads (SPEC AC-11b)."""
    df = build_cube_dataframe_for_gee_utm(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert GEE_REQUIRED_COLUMNS.issubset(set(df.columns))
    # date parses as YYYYMMDD (the exporter does strptime(str(date), "%Y%m%d")).
    from datetime import datetime

    for value in df["date"].unique():
        datetime.strptime(str(value), "%Y%m%d")


def test_gee_utm_centre_is_true_degrees(kept_cells):
    """center_lat/center_lon carry true degrees, not mislabelled UTM eastings.

    Bow Valley sits near 51 N, -115 E; the values must land in that geographic
    range (never the ~5.6e6 / ~6e5 metre magnitudes of the UTM bounds).
    """
    df = build_cube_dataframe_for_gee_utm(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    )
    assert df["center_lat"].between(49.0, 53.0).all()
    assert df["center_lon"].between(-117.0, -113.0).all()
    # And they must differ from the UTM bounds they were derived from.
    assert (df["center_lat"] != df["min_y"]).all()
    assert (df["center_lon"] != df["min_x"]).all()


def test_filename_matches_gee_pattern(kept_cells):
    """The exporter builds PR_{date}_{cx:.16f}_{cy:.16f}.tif from these columns.

    Reproduces the eo_eval.py:599 filename build and asserts it parses through
    the ``PR`` branch of ``LandsatEvalDataset`` (month at parts[1][4:6]).
    """
    df = build_cube_dataframe(
        kept=kept_cells, window_start=DEFAULT_WINDOW_START, window_end=DEFAULT_WINDOW_END
    ).iloc[0]
    filename = f"PR_{df['date']}_{df['center_x']:.16f}_{df['center_y']:.16f}.tif"
    parts = filename.split("_")
    assert parts[0] == "PR"
    assert parts[1][4:6] == "04"  # month of 2025-04-06 window start day


def test_generate_writes_files(kept_cells, tmp_path):
    """`generate` writes both the cube CSV and the manifest, round-trippable."""
    out_csv = tmp_path / "cube_cells.csv"
    manifest = tmp_path / "manifest.csv"
    df = generate(
        cube_cells_csv=LEGACY_CSV,
        aoi_path=AOI_PATH,
        output_csv=out_csv,
        manifest_path=manifest,
        window_start=DEFAULT_WINDOW_START,
        window_end=DEFAULT_WINDOW_END,
    )
    assert out_csv.exists()
    assert manifest.exists()
    reread = pd.read_csv(out_csv)
    assert list(reread.columns) == CUBE_CSV_COLUMNS
    assert len(reread) == len(df)
    # manifest accounts for all 500 cells
    man = pd.read_csv(manifest)
    assert len(man) == 500
