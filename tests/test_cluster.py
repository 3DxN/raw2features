"""``raw2features cluster``: placeholder trimodal clustering -> JSON, labels, glTF."""

from __future__ import annotations

import hashlib
import json
import struct
import warnings

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("torch")
from conftest import MockEmbedder
from raw2features.clustering import cluster_store
from raw2features.clustering.labels import label_name, local_source_path
from raw2features.clustering.trimodal import PLACEHOLDER_TRIMODAL_SHA256
from raw2features.core.falsecolor import PseudoHEParams
from raw2features.pipeline.runner import RunConfig, embed_slide


@pytest.fixture
def embedded(tmp_path):
    """A 3-plane fluorescence stack embedded per z plane; returns (source, store)."""
    from raw2features.data import write_sample_fluorescence_stack

    source = write_sample_fluorescence_stack(
        str(tmp_path / "s3d.ome.zarr"), z=3, size=512, levels=2
    )
    cfg = RunConfig(
        models=["mock"],
        z_planes="all",
        pseudo_he=PseudoHEParams("DAPI", "Eosin"),
        device="cpu",
        amp="fp32",
        batch_size=8,
    )
    out = str(tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embed_slide(
            source,
            out,
            cfg,
            requested_mpp=0.5,
            requested_patch_px=64,
            embedders=[MockEmbedder()],
        )
    return source, f"{out}/s3d.embeddings.zarr"


def _load(summary) -> dict:
    with open(summary["json"], encoding="utf-8") as fh:
        return json.load(fh)


def test_json_assigns_one_placeholder_cluster_per_location(embedded, tmp_path):
    from raw2features.core.store import GRIDS, grid_keys, open_root

    _source, store = embedded
    (summary,) = cluster_store(store, out_dir=str(tmp_path / "c"), gltf=False)
    record = _load(summary)

    root = open_root(store)
    expected = [
        f"{key}/{row}"
        for key in grid_keys(root)
        for row in range(root[GRIDS][key]["coords"].shape[0])
    ]
    assert record["feature_ids"] == expected
    assert len({k.split("/")[0] for k in expected}) == 3  # all z planes
    assert set(record["clusters"]) == set(expected)
    assert set(record["clusters"].values()) <= {7, 23}
    assert record["trimodal_model"] == {
        "sha256": PLACEHOLDER_TRIMODAL_SHA256,
        "path": None,
        "placeholder": True,
    }
    emb = record["embedding_model"]
    assert emb["name"] == "mock" and len(emb["sha256"]) == 64
    assert record["label_image"]["name"] == label_name(
        emb["sha256"], PLACEHOLDER_TRIMODAL_SHA256
    )
    # Deterministic for a fixed seed.
    (again,) = cluster_store(store, out_dir=str(tmp_path / "d"), gltf=False)
    assert _load(again)["clusters"] == record["clusters"]


def test_trimodal_model_file_hash_is_recorded(embedded, tmp_path):
    _source, store = embedded
    model = tmp_path / "trimodal.pt"
    model.write_bytes(b"trimodal weights")
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), trimodal_model=str(model), gltf=False
    )
    tri = _load(summary)["trimodal_model"]
    assert tri["sha256"] == hashlib.sha256(b"trimodal weights").hexdigest()
    assert tri["placeholder"] is False


def test_label_image_mirrors_source_pyramid_with_footprint_quads(embedded, tmp_path):
    import ngff_zarr
    import zarr

    source, store = embedded
    image = zarr.open_group(source, mode="r+")
    image.require_group("labels").attrs["labels"] = ["existing"]
    image["labels"].require_group("existing")

    (summary,) = cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True)
    record = _load(summary)
    name = record["label_image"]["name"]
    assert name.startswith("clusters_") and len(name.split("_")[1]) == 8
    assert summary["labels"] == f"{source}/labels/{name}"

    labels = zarr.open_group(f"{source}/labels", mode="r")
    assert dict(labels.attrs)["labels"] == ["existing", name]
    lab = labels[name]
    attrs = dict(lab.attrs)
    assert attrs["image-label"]["source"] == {"image": "../../"}
    src_ms = dict(image.attrs)["multiscales"][0]
    assert attrs["multiscales"][0]["datasets"] == src_ms["datasets"]
    for ds in src_ms["datasets"]:
        src, out = image[ds["path"]], lab[ds["path"]]
        assert out.shape == (1, 1, *src.shape[2:])
        assert out.chunks == (1, 1, *src.chunks[2:])
        assert out.metadata.dimension_separator == "/"
    ngff_zarr.from_ngff_zarr(f"{source}/labels/{name}")  # a valid NGFF image

    # Every level equals the footprints, side / downsample per x/y (non-overlapping
    # dense grid, z not downsampled): exact size and position, background elsewhere.
    sides = {tuple(f["size_zyx"]) for f in record["footprints"].values()}
    (side,) = {s[1] for s in sides}
    assert sides == {(1, side, side)} and side > 1
    for k, d in enumerate(src_ms["datasets"]):
        level = np.asarray(lab[d["path"]][0, 0])
        ds = 2**k
        expected = np.zeros_like(level)
        for fid, foot in record["footprints"].items():
            z, y0, x0 = foot["origin_zyx"]
            assert record["locations_zyx"][fid] == [z, y0 + side // 2, x0 + side // 2]
            ys = slice(-(-y0 // ds), -(-(y0 + side) // ds))  # centres inside the box
            xs = slice(-(-x0 // ds), -(-(x0 + side) // ds))
            assert ys.stop - ys.start == side // ds
            expected[z, ys, xs] = record["clusters"][fid]
        np.testing.assert_array_equal(level, expected)


def test_label_name_collision_needs_overwrite(embedded, tmp_path):
    _source, store = embedded
    cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True, gltf=False)
    with pytest.raises(FileExistsError):
        cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True, gltf=False)
    cluster_store(
        store,
        out_dir=str(tmp_path / "c"),
        write_labels=True,
        labels_overwrite=True,
        gltf=False,
    )


def test_remote_sources_are_refused():
    assert local_source_path("file:///data/s.ome.zarr") == "/data/s.ome.zarr"
    with pytest.raises(ValueError, match="remote"):
        local_source_path("s3://bucket/s.ome.zarr")


def _read_glb(path: str) -> dict:
    data = open(path, "rb").read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    js_len, js_type = struct.unpack_from("<II", data, 12)
    assert js_type == 0x4E4F534A
    return json.loads(data[20 : 20 + js_len])


def test_gltf_has_one_named_mesh_per_location(embedded, tmp_path):
    _source, store = embedded
    (summary,) = cluster_store(store, out_dir=str(tmp_path / "c"))
    record = _load(summary)
    doc = _read_glb(summary["gltf"])
    ids = record["feature_ids"]
    assert [m["name"] for m in doc["meshes"]] == ids
    assert [n["name"] for n in doc["nodes"]] == ids
    assert [n["mesh"] for n in doc["nodes"]] == list(range(len(ids)))
    for node in doc["nodes"]:
        foot = record["footprints"][node["name"]]
        z, y0, x0 = foot["origin_zyx"]
        _, h, w = foot["size_zyx"]
        assert node["translation"] == [x0 + (w - 1) / 2, y0 + (h - 1) / 2, z]
    # One shared quad: side x side in the z = 0 plane, double-sided.
    (side,) = {f["size_zyx"][1] for f in record["footprints"].values()}
    pos = doc["accessors"][0]
    assert pos["count"] == 4
    assert pos["min"] == [-side / 2, -side / 2, 0.0]
    assert pos["max"] == [side / 2, side / 2, 0.0]
    assert {m["primitives"][0]["indices"] for m in doc["meshes"]} == {2}
    assert doc["materials"] == [{"name": "footprint", "doubleSided": True}]
    assert (
        doc["asset"]["extras"]["embedding_sha256"]
        == (record["embedding_model"]["sha256"])
    )


def test_cli_cluster(embedded, tmp_path):
    from typer.testing import CliRunner

    from raw2features.cli.main import app

    _source, store = embedded
    result = CliRunner().invoke(
        app, ["cluster", store, "--out-dir", str(tmp_path / "c"), "--seed", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "mock:" in result.output
    assert (tmp_path / "c" / "s3d.mock.clusters.json").exists()
    assert (tmp_path / "c" / "s3d.mock.clusters.glb").exists()


def test_overlapping_footprints_split_by_nearest_centre():
    from raw2features.clustering.footprint import Footprints
    from raw2features.clustering.labels import _Level, _rasterise_level

    axes = ["z", "y", "x"]
    # Two 8x8 patches with a stride of 4 on plane 0 and one on plane 1.
    feet = Footprints(
        np.asarray([[0, 0, 0], [0, 0, 4], [1, 0, 0]]), np.asarray([[1, 8, 8]] * 3)
    )
    clusters = np.asarray([7, 23, 5])
    level0 = np.zeros((2, 8, 12), dtype=np.uint8)
    lv = _Level("0", level0.shape, level0.shape, dict.fromkeys(axes, 1.0),
                dict.fromkeys(axes, 0.0))  # fmt: skip
    _rasterise_level(level0, axes, axes, lv, feet, clusters)
    assert (level0[0, :, :6] == 7).all() and (level0[0, :, 6:] == 23).all()
    assert (level0[1, :, :8] == 5).all() and (level0[1, :, 8:] == 0).all()

    # z downsampled 2x: both planes fall into one level voxel; the quads still show
    # and the tie at the shared z voxel goes by xy distance, then feature order.
    coarse = np.zeros((1, 4, 6), dtype=np.uint8)
    lv = _Level("1", coarse.shape, coarse.shape, dict.fromkeys(axes, 2.0),
                dict.fromkeys(axes, 0.0))  # fmt: skip
    _rasterise_level(coarse, axes, axes, lv, feet, clusters)
    assert coarse[0, 0].tolist() == [7, 7, 7, 23, 23, 23]
