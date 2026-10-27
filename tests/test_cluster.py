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

RADIUS = 16


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


def test_label_image_mirrors_source_pyramid_with_scaled_spheres(embedded, tmp_path):
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

    levels = [np.asarray(lab[d["path"]][0, 0]) for d in src_ms["datasets"]]
    for fid in record["feature_ids"]:
        z, y, x = record["locations_zyx"][fid]
        value = record["clusters"][fid]
        for k, level in enumerate(levels):
            ds, r = 2**k, RADIUS // 2**k  # radius 16 at level 0, 8 at level 1, ...
            yl, xl = y // ds, x // ds
            assert level[z, yl, xl] == value  # each location keeps its centre
            assert level[z, yl, xl + r] != 0
            assert level[z, yl, xl + r + 1] == 0


def test_label_name_collision_needs_overwrite(embedded, tmp_path):
    _source, store = embedded
    cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True, gltf=False)
    with pytest.raises(FileExistsError):
        cluster_store(
            store, out_dir=str(tmp_path / "c"), write_labels=True, gltf=False
        )
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
        z, y, x = record["locations_zyx"][node["name"]]
        assert node["translation"] == [x, y, z]
    pos = doc["accessors"][0]
    assert pos["max"][0] == pytest.approx(RADIUS)
    assert doc["asset"]["extras"]["embedding_sha256"] == (
        record["embedding_model"]["sha256"]
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
