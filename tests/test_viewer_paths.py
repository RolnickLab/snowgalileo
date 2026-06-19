"""Unit tests for the clip-viewer pure path-building helpers.

Covers GDAL ``/vsitar/`` + ``/vsizip/`` path construction and archive member
lookup (``snow_galileo.data.local_sources.viewer.archives``) and manifest bbox parsing + output-path
resolution (``snow_galileo.data.local_sources.viewer.manifest``). No GDAL/rasterio I/O — these are the
deterministic string/path functions the renderers depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snow_galileo.data.local_sources.viewer.archives import find_member, vsitar_path, vsizip_path
from snow_galileo.data.local_sources.viewer.manifest import (
    _parse_bbox,
    _resolve_path,
    load_products,
)
from snow_galileo.data.local_sources.viewer.settings import ViewerSettings


class TestVsiPaths:
    """GDAL virtual-file-system path builders."""

    def test_vsitar_path_uses_absolute_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "scene.tar"
        archive.touch()
        result = vsitar_path(archive, "LC09_B4.TIF")
        assert result == f"/vsitar/{archive.resolve()}/LC09_B4.TIF"

    def test_vsizip_path_uses_absolute_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "S2.zip"
        archive.touch()
        result = vsizip_path(archive, "IMG_DATA/T11_B04.jp2")
        assert result == f"/vsizip/{archive.resolve()}/IMG_DATA/T11_B04.jp2"

    def test_vsi_paths_have_no_collapsed_double_slash(self, tmp_path: Path) -> None:
        # GDAL needs /vsitar/<abs>/<member>; the leading scheme must survive and the
        # member separator must not collapse the way Path("/vsitar//...") would.
        archive = tmp_path / "scene.tar"
        archive.touch()
        result = vsitar_path(archive, "band.TIF")
        assert result.startswith("/vsitar/")
        assert "//" not in result.removeprefix("/vsitar/")

    def test_vsi_path_is_str_not_path(self, tmp_path: Path) -> None:
        # Returned as ``str`` precisely so pathlib normalisation never touches it.
        archive = tmp_path / "scene.tar"
        archive.touch()
        assert isinstance(vsitar_path(archive, "b.TIF"), str)


class TestFindMember:
    """Archive member discovery by suffix / substring."""

    MEMBERS = [
        "S2A_.../IMG_DATA/T11UQR_B04.jp2",
        "S2A_.../IMG_DATA/T11UQR_B03.jp2",
        "S2A_.../QI_DATA/T11UQR_B04.jp2",  # same band suffix, wrong dir
        "S2A_.../MTD_TL.xml",
    ]

    def test_finds_by_suffix(self) -> None:
        assert find_member(self.MEMBERS, suffix="_B03.jp2").endswith("_B03.jp2")

    def test_contains_disambiguates_same_suffix(self) -> None:
        result = find_member(self.MEMBERS, suffix="_B04.jp2", contains="IMG_DATA")
        assert "IMG_DATA" in result
        assert "QI_DATA" not in result

    def test_returns_first_match_in_order(self) -> None:
        # Two B04 members exist; without ``contains`` the first in list order wins.
        result = find_member(self.MEMBERS, suffix="_B04.jp2")
        assert result == self.MEMBERS[0]

    def test_raises_when_absent(self) -> None:
        with pytest.raises(FileNotFoundError, match="no member matching"):
            find_member(self.MEMBERS, suffix="_B12.jp2")

    def test_raises_when_contains_excludes_all(self) -> None:
        with pytest.raises(FileNotFoundError):
            find_member(self.MEMBERS, suffix="_B04.jp2", contains="NOPE")


class TestParseBbox:
    """``footprint_bbox`` CSV-cell parsing."""

    def test_parses_four_floats(self) -> None:
        assert _parse_bbox("-132.682,49.8966,-107.171,62.9214") == (
            -132.682,
            49.8966,
            -107.171,
            62.9214,
        )

    def test_rejects_wrong_arity(self) -> None:
        with pytest.raises(ValueError, match="4 values"):
            _parse_bbox("1.0,2.0,3.0")


class TestResolvePath:
    """Manifest ``output_path`` → on-disk file/dir resolution."""

    def test_skip_rows_resolve_to_none(self, tmp_path: Path) -> None:
        assert (
            _resolve_path(
                source="worldcover",
                output_path="anything.tif",
                action="SKIP_NO_OVERLAP",
                root=tmp_path,
            )
            is None
        )

    def test_direct_file_under_source_dir(self, tmp_path: Path) -> None:
        (tmp_path / "sentinel3").mkdir()
        target = tmp_path / "sentinel3" / "S3A_OL_1.zip"
        target.touch()
        result = _resolve_path(
            source="sentinel3",
            output_path="S3A_OL_1.zip",
            action="CLIP",
            root=tmp_path,
        )
        assert result == target

    def test_nested_basename_resolved_via_rglob(self, tmp_path: Path) -> None:
        # DEM case: output_path is a bare basename buried several dirs deep.
        nested = tmp_path / "dem" / "SAFE" / "tiles" / "dem_tile.tif"
        nested.parent.mkdir(parents=True)
        nested.touch()
        result = _resolve_path(
            source="dem",
            output_path="dem_tile.tif",
            action="CLIP",
            root=tmp_path,
        )
        assert result == nested

    def test_missing_output_resolves_to_none(self, tmp_path: Path) -> None:
        (tmp_path / "dem").mkdir()
        assert (
            _resolve_path(
                source="dem",
                output_path="ghost.tif",
                action="CLIP",
                root=tmp_path,
            )
            is None
        )


_MANIFEST_HEADER = (
    "product_id,source,footprint_bbox,intersects,aoi_overlap_km2,"
    "valid_pixel_count,action,output_path"
)


class TestLoadProducts:
    """``load_products`` manifest-row handling."""

    def test_skips_legacy_sentinel1_manifest_rows(self, tmp_path: Path) -> None:
        """Legacy ``sentinel1`` manifest rows (raw ``*.zip``) are dropped — S1 is processed
        (SNAP), not clipped, so its rows come from the SNAP cache, never the manifest. A
        zip is not a raster; keeping the row produced a broken ``plain_image`` fallback.
        """
        # A real DEM file so its row resolves; an S1 row pointing at a (non-existent) zip.
        (tmp_path / "dem").mkdir()
        (tmp_path / "dem" / "tile_DEM.tif").touch()
        manifest = tmp_path / "clip_manifest.csv"
        manifest.write_text(
            _MANIFEST_HEADER
            + "\n"
            + 'dem_tile,dem,"-116.5,50.7,-114.5,52.3",True,100.0,9999,CLIP,tile_DEM.tif\n'
            + 'S1C_x,sentinel1,"-119.4,49.8,-115.3,51.7",True,7480.9,1385,CLIP,S1C_x.zip\n'
        )
        # No sentinel1_snap/ dir under tmp_path → _discover_s1_products returns [], so the
        # only way an S1 row could appear is the (now-filtered) manifest row.
        settings = ViewerSettings(clipped_root=tmp_path)

        rows = load_products(settings)

        assert [r.source for r in rows] == ["dem"], "legacy sentinel1 manifest row not skipped"
        assert all(r.source != "sentinel1" for r in rows)
