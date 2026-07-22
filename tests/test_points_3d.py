"""3D point-cloud sampling: reader z-plane reads, point loading/placement, e2e embed."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.points import (
    load_point_cloud,
    map_points_to_grid,
    point_cloud_fingerprint,
)
from raw2features.data import write_sample_points, write_sample_slide
from raw2features.patcher.grid import GridPatcher
from raw2features.patcher.points import PointSampler
from raw2features.readers.omezarr import OmeZarrReader


@pytest.fixture
def zstack(tmp_path) -> str:
    """A synthetic 6-plane z-stack at 0.5 µm/px, 2 µm plane spacing."""
    return write_sample_slide(
        str(tmp_path / "z.ome.zarr"), mpp0=0.5, size=512, z=6, z_spacing=2.0
    )


# -- reader: z-plane selection -------------------------------------------------
def test_reader_reads_each_z_plane_equivalent_to_direct_slice(zstack):
    with OmeZarrReader(zstack) as r:
        assert r.z_size(0) == 6
        assert r.z_scale_um == 2.0
        arr0 = r._arrays[0]  # dims (t, c, z, y, x)
        for k in (0, 2, 5):
            got = r.read_region(Region(0, Point(64, 96), Size(48, 48), z=k))
            direct = np.transpose(np.asarray(arr0[0, :, k, 96:144, 64:112]), (1, 2, 0))
            assert np.array_equal(got, direct), f"z={k} mismatch"


def test_reader_chunk_cache_does_not_alias_across_z(zstack):
    # Reading plane 0 then plane 3 at the same (x, y) must return different pixels
    # (the planes differ by a per-plane brightness step); a z-blind cache key would
    # wrongly return plane 0 for the second read.
    with OmeZarrReader(zstack) as r:
        p0 = r.read_region(Region(0, Point(100, 100), Size(32, 32), z=0))
        p3 = r.read_region(Region(0, Point(100, 100), Size(32, 32), z=3))
        assert not np.array_equal(p0, p3)


def test_z_index_for_um_rounds_to_nearest_plane(zstack):
    with OmeZarrReader(zstack) as r:
        assert r.z_index_for_um(0.0) == 0
        assert r.z_index_for_um(4.0) == 2  # 4 µm / 2 µm spacing
        assert r.z_index_for_um(4.9) == 2  # nearest
        assert r.z_index_for_um(1000.0) == 5  # clipped to the last plane


def test_z_index_for_um_uncalibrated_raises(tmp_path):
    # A z axis with no unit has no physical spacing; mapping a µm depth must fail loud.
    slide = write_sample_slide(str(tmp_path / "u.ome.zarr"), size=256, z=4)
    with OmeZarrReader(slide) as r:
        r._z_scale_um = None  # simulate an uncalibrated z axis
        with pytest.raises(ValueError, match="z axis has no physical spacing"):
            r.z_index_for_um(2.0)
        r.apply_source_z_spacing(3.0)  # supply it explicitly
        assert r.z_index_for_um(6.0) == 2


def test_2d_reader_reports_flat_z(tmp_path):
    slide = write_sample_slide(str(tmp_path / "flat.ome.zarr"), size=256, z=1)
    with OmeZarrReader(slide) as r:
        assert r.z_size(0) == 1
        # A flat source ignores a nonzero z request (serves its single plane).
        a = r.read_region(Region(0, Point(0, 0), Size(16, 16), z=0))
        b = r.read_region(Region(0, Point(0, 0), Size(16, 16), z=3))
        assert np.array_equal(a, b)


# -- point loading -------------------------------------------------------------
def test_load_point_cloud_csv(tmp_path):
    path = write_sample_points(str(tmp_path / "p.csv"), size=512, z=6, n_side=3)
    cloud = load_point_cloud(path)
    assert len(cloud) == 9
    assert cloud.xyz_um.shape == (9, 3)
    assert cloud.ids[0] == "pt_000"
    assert cloud.sha256 == point_cloud_fingerprint(path)


def test_load_point_cloud_missing_column_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        load_point_cloud(str(p))


def test_load_point_cloud_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    tbl = pa.table({"x": [10.0, 20.0], "y": [30.0, 40.0], "z": [0.0, 2.0],
                    "id": ["a", "b"]})
    path = str(tmp_path / "p.parquet")
    pq.write_table(tbl, path)
    cloud = load_point_cloud(path)
    assert len(cloud) == 2
    assert list(cloud.ids) == ["a", "b"]


# -- placement -----------------------------------------------------------------
def test_map_points_center_to_topleft_and_z(zstack):
    with OmeZarrReader(zstack) as r:
        # one point at (100 µm, 50 µm, 4 µm) -> level-0 px (200, 100), plane 2
        cloud = load_point_cloud(_write_csv(zstack, [(100.0, 50.0, 4.0, "c")]))
        coords, z_index, pum, ids = map_points_to_grid(cloud, r, level0_patch=64)
        assert coords.tolist() == [[200 - 32, 100 - 32]]  # center - patch/2
        assert z_index.tolist() == [2]
        assert ids.tolist() == ["c"]
        assert np.allclose(pum[0], [100.0, 50.0, 4.0])


def test_map_points_drops_out_of_bounds(zstack):
    with OmeZarrReader(zstack) as r:
        # 512 px * 0.5 µm = 256 µm slide extent; 1000 µm is off-slide.
        pts = [(100.0, 100.0, 0.0, "in"), (1000.0, 100.0, 0.0, "out")]
        cloud = load_point_cloud(_write_csv(zstack, pts))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coords, z_index, pum, ids = map_points_to_grid(cloud, r, level0_patch=64)
        assert coords.shape[0] == 1
        assert ids.tolist() == ["in"]
        assert any("outside" in str(w.message) for w in caught)


def test_point_sampler_shares_read_geometry_with_grid(zstack):
    with OmeZarrReader(zstack) as r:
        pts = write_sample_points(str(zstack) + ".pts.csv", size=512, z=6, n_side=2)
        sampler = PointSampler(pts, target_mpp=1.0, patch_px=64)
        grid = GridPatcher(target_mpp=1.0, patch_px=64).build_grid(r)
        pgrid = sampler.build_grid(r)
        assert (pgrid.read_level, pgrid.read_px, pgrid.level0_patch) == (
            grid.read_level, grid.read_px, grid.level0_patch
        )


def _write_csv(near_path: str, rows) -> str:
    import csv
    import os

    path = os.path.join(os.path.dirname(near_path), "tmp_points.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "id"])
        w.writerows(rows)
    return path


# -- end to end ----------------------------------------------------------------
def test_run_slide_point_sampling_e2e(zstack, tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from conftest import MockEmbedder
    from raw2features.core.store import open_grid
    from raw2features.pipeline.runner import RunConfig, run_slide
    from raw2features.spec import validate_store

    pts = write_sample_points(str(tmp_path / "pts.csv"), size=512, z=6, z_spacing=2.0,
                              n_side=3)
    cfg = RunConfig(
        models=["mock"], target_mpp=0.5, patch_px=64, device="cpu", amp="fp32",
        batch_size=4, points_path=pts, points_sha256=point_cloud_fingerprint(pts),
    )
    out, rec = str(tmp_path / "out"), str(tmp_path / "rec")
    s = run_slide(zstack, out, cfg, receipts_dir=rec, embedders=[MockEmbedder()])
    assert s["status"] == "complete"
    n = s["n_patches"]
    assert n == 9

    g = open_grid(s["output_uri"])
    assert g["coords"].shape == (n, 2)
    assert g["z_index"].shape == (n,)
    assert g["source_points"].shape == (n, 3)
    assert g["features"]["mock"].shape == (n, 8)
    assert np.isfinite(g["features"]["mock"][:]).all()
    assert "grid_index" not in g and "mask" not in g  # point mode omits both

    header = dict(g.attrs)["raw2features"]
    assert header["sampling"]["mode"] == "points"
    assert header["sampling"]["z_scale_um"] == 2.0
    assert header["patching"]["coords_convention"] == "level0_xy_z_points"
    assert header["patching"]["grid_shape"] is None

    assert validate_store(s["output_uri"].removeprefix("file://")) == []

    # identical rerun short-circuits via the receipt (points_sha256 in the hash)
    again = run_slide(zstack, out, cfg, receipts_dir=rec, embedders=[MockEmbedder()])
    assert again["status"] == "skipped"
