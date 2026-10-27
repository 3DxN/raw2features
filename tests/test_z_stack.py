"""z-stack plane iteration, physical-size grids with offset, and per-z store layout."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("zarr")
from conftest import MockEmbedder
from raw2features.core.falsecolor import PseudoHEParams
from raw2features.core.geometry import Point, Region, Size
from raw2features.core.store import grid_key
from raw2features.patcher.grid import GridPatcher
from raw2features.pipeline.runner import RunConfig, parse_z_planes
from raw2features.readers.omezarr import OmeZarrReader


@pytest.fixture
def fluo_stack(tmp_path) -> str:
    from raw2features.data import write_sample_fluorescence_stack

    return write_sample_fluorescence_stack(
        str(tmp_path / "s3d.ome.zarr"), z=3, size=512, levels=2
    )


# -- reader ----------------------------------------------------------------------
def test_reader_serves_the_selected_plane_without_cache_bleed(fluo_stack):
    import zarr

    raw = zarr.open_array(f"{fluo_stack}/0", mode="r")
    region = Region(0, Point(64, 96), Size(32, 32))
    with OmeZarrReader(fluo_stack) as r:
        assert r.z_count == 3 and r.z_scale_um == pytest.approx(2.0)
        for z in (0, 2, 1, 0):  # revisit planes: cached chunks must not leak across z
            r.set_plane(z)
            got = r.read_region_channels(region)
            want = np.moveaxis(np.asarray(raw[0, :, z, 96:128, 64:96]), 0, -1)
            np.testing.assert_array_equal(got, want)
            assert r.z_um == pytest.approx(2.0 * z)
        # Coarser levels keep the z extent here, so plane 1 maps to plane 1.
        r.set_plane(1)
        assert r._level_z(1) == 1
        with pytest.raises(ValueError):
            r.set_plane(3)


def test_selecting_a_plane_suppresses_the_z_collapse_warning(fluo_stack):
    with OmeZarrReader(fluo_stack) as r, warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        r.set_plane(0)
        r.read_region(Region(0, Point(0, 0), Size(8, 8)))
    assert not [w for w in c if "only z=0" in str(w.message)]


def test_2d_reader_rejects_nonzero_plane(synthetic_ngff):
    from raw2features.readers.base import WSISource

    with OmeZarrReader(synthetic_ngff) as r:
        assert r.z_count == 1
    with pytest.raises(NotImplementedError):
        WSISource.set_plane(r, 1)


@pytest.mark.parametrize(
    ("spec", "want"),
    [("all", [0, 1, 2, 3]), ("0,2", [0, 2]), ("1:", [1, 2, 3]), ("::2", [0, 2])],
)
def test_parse_z_planes(spec, want):
    assert parse_z_planes(spec, 4) == want


@pytest.mark.parametrize("spec", ["", "a", "0:4:0", "-1", "9"])
def test_parse_z_planes_rejects(spec):
    with pytest.raises(ValueError):
        parse_z_planes(spec, 4)


# -- grid --------------------------------------------------------------------------
def test_grid_offset_shifts_origin_by_patch_fraction(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        base = GridPatcher(target_mpp=0.5, patch_px=64).build_grid(r)
        off = GridPatcher(
            target_mpp=0.5, patch_px=64, offset_x_frac=0.5, offset_y_frac=0.25
        ).build_grid(r)
    assert base.level0_offset == (0, 0)
    assert off.level0_offset == (base.level0_patch // 2, base.level0_patch // 4)
    assert off.xs[0] == base.level0_patch // 2 and off.ys[0] == base.level0_patch // 4
    # dense, non-overlapping: stride == patch; partial edge tiles are dropped
    assert np.diff(off.xs).tolist() == [off.level0_patch] * (len(off.xs) - 1)
    w0, h0 = r.level_dimensions[0]
    assert off.xs[-1] + off.level0_patch <= w0 and off.ys[-1] + off.level0_patch <= h0


@pytest.mark.parametrize("frac", [-0.1, 1.0])
def test_grid_offset_must_be_a_fraction(frac):
    with pytest.raises(ValueError):
        GridPatcher(offset_x_frac=frac)


def test_patch_um_defines_target_mpp():
    cfg = RunConfig(models=["mock"], patch_um=112.0, patch_px=224)
    assert cfg.target_mpp == pytest.approx(0.5)
    with pytest.raises(ValueError):
        RunConfig(models=["mock"], patch_um=112.0, snap_to_level=True)


def test_default_2d_hashes_are_unchanged():
    # Pinned from the pre-z-stack code: new options must not perturb old identities.
    cfg = RunConfig(models=["resnet50"], target_mpp=0.5, patch_px=224)
    assert cfg.content_hash() == "224aedbb77171ad1"
    assert cfg.grid_hash() == "b06597cf985fa1ae"


def test_new_content_fields_change_the_grid_identity():
    cfg = RunConfig(models=["resnet50"], target_mpp=0.5, patch_px=224)
    variants = [
        RunConfig(models=["resnet50"], target_mpp=0.5, patch_px=224, z_index=0),
        RunConfig(
            models=["resnet50"], target_mpp=0.5, patch_px=224, grid_offset=(0.5, 0)
        ),
        RunConfig(
            models=["resnet50"],
            target_mpp=0.5,
            patch_px=224,
            pseudo_he=PseudoHEParams("DAPI", "Eosin"),
        ),
    ]
    hashes = {cfg.grid_hash(), *(v.grid_hash() for v in variants)}
    assert len(hashes) == 4
    # The requested plane set is runtime-only: each plane's grid identity is z_index.
    assert (
        RunConfig(models=["resnet50"], z_planes="all", z_index=1).grid_hash()
        == RunConfig(models=["resnet50"], z_planes="0:2", z_index=1).grid_hash()
    )


def test_grid_key_suffixes_z():
    assert grid_key(0.5, 224) == "mpp0.5_px224"
    assert grid_key(0.5, 224, 3) == "mpp0.5_px224_z3"


# -- end to end ------------------------------------------------------------------
def test_embed_slide_writes_one_pseudo_he_grid_per_plane(fluo_stack, tmp_path):
    pytest.importorskip("torch")
    from raw2features.core.store import GRIDS, open_root
    from raw2features.pipeline.runner import embed_slide
    from raw2features.spec import validate_store

    cfg = RunConfig(
        models=["mock"],
        patch_um=32.0,
        patch_px=64,
        grid_offset=(0.5, 0.25),
        z_planes="all",
        pseudo_he=PseudoHEParams("DAPI", "Eosin"),
        device="cpu",
        amp="fp32",
        batch_size=8,
    )
    out = str(tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no "only z=0 is read"
        warnings.filterwarnings("ignore", message=".*recommended")
        summary = embed_slide(
            fluo_stack,
            out,
            cfg,
            requested_mpp=0.5,
            requested_patch_px=64,
            embedders=[MockEmbedder()],
        )
    assert list(summary["grids"]) == [f"mpp0.5_px64_z{z}" for z in range(3)]
    store = f"{out}/s3d.embeddings.zarr"
    assert validate_store(store) == []
    root = open_root(store)
    counts = []
    for z in range(3):
        g = root[GRIDS][f"mpp0.5_px64_z{z}"]
        header = dict(g.attrs)["raw2features"]
        patching = header["patching"]
        assert header["schema_version"] == "0.2"
        assert patching["z_index"] == z and patching["z_um"] == pytest.approx(2.0 * z)
        assert patching["level0_offset"] == [32, 16]
        assert patching["patch_um"] == pytest.approx(32.0)
        assert header["source"]["z_count"] == 3
        assert header["source"]["reader"] == "omezarr"
        assert header["source"]["pseudo_he"]["nuclear_name"] == "DAPI"
        coords = np.asarray(g["coords"])
        assert ((coords[:, 0] - 32) % 64 == 0).all()
        assert ((coords[:, 1] - 16) % 64 == 0).all()
        counts.append(len(coords))
    # The synthetic tissue shrinks with depth, so each plane has its own mask.
    assert counts[0] > counts[-1] > 0

    again = embed_slide(
        fluo_stack,
        out,
        cfg,
        requested_mpp=0.5,
        requested_patch_px=64,
        embedders=[MockEmbedder()],
    )
    assert again["status"] == "skipped"
