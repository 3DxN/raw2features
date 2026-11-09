"""``raw2features cluster``: clusterer plugins -> JSON, labels, glTF."""

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
from raw2features.clustering import ClusterContext, Clusterer, cluster_store
from raw2features.clustering.base import ClustererIdentity
from raw2features.clustering.footprint import Footprints
from raw2features.clustering.labels import (
    label_name,
    local_source_path,
    write_label_image,
)
from raw2features.clustering.trimodal import (
    PLACEHOLDER_TRIMODAL_SHA256,
    TrimodalClusterer,
)
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


def test_json_assigns_one_placeholder_cluster_per_location(
    embedded, tmp_path, monkeypatch
):
    from raw2features.core.store import GRIDS, grid_keys, open_root

    _source, store = embedded
    monkeypatch.chdir(tmp_path)  # no model.pt here: placeholder identity, warned
    with pytest.warns(UserWarning, match="model.pt"):
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
    tri = record["clustering_model"]
    assert tri.pop("version")
    assert tri == {
        "plugin": "trimodal",
        "sha256": PLACEHOLDER_TRIMODAL_SHA256,
        "path": None,
        "placeholder": True,
        "inputs": {},
    }
    emb = record["embedding_model"]
    assert emb["name"] == "mock" and len(emb["sha256"]) == 64
    assert record["label_image"]["name"] == label_name(
        emb["sha256"], PLACEHOLDER_TRIMODAL_SHA256
    )
    # Deterministic for a fixed seed.
    (again,) = cluster_store(
        store, out_dir=str(tmp_path / "d"), clusterer="placeholder", gltf=False
    )
    assert _load(again)["clusters"] == record["clusters"]


def test_trimodal_model_file_hash_is_recorded(embedded, tmp_path, monkeypatch):
    _source, store = embedded
    model = tmp_path / "trimodal.pt"
    model.write_bytes(b"trimodal weights")
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer_model=str(model), gltf=False
    )
    tri = _load(summary)["clustering_model"]
    assert tri["sha256"] == hashlib.sha256(b"trimodal weights").hexdigest()
    assert tri["path"] == str(model)
    assert tri["placeholder"] is True  # inference itself is still a placeholder
    assert summary["labels"] is None

    # The default model file is model.pt in the working directory.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"default weights")
    ident = TrimodalClusterer().identity()
    assert ident.model_path == str(tmp_path / "model.pt")
    assert ident.model_sha256 == hashlib.sha256(b"default weights").hexdigest()

    with pytest.raises(FileNotFoundError):
        TrimodalClusterer(str(tmp_path / "missing.pt")).load()


class _RecordingClusterer(Clusterer):
    name = "recording"

    def __init__(self, model_file=None, value=3):
        super().__init__(model_file)
        self.value, self.calls, self.loaded = value, [], 0

    def load(self):
        self.loaded += 1

    def identity(self):
        return ClustererIdentity("recording", "1.0", "ab" * 32, None)

    def predict(self, features, context: ClusterContext):
        self.calls.append((np.asarray(features).shape, context))
        return np.full(len(features), self.value)


class _ZeroClusterer(_RecordingClusterer):
    def __init__(self, model_file=None):
        super().__init__(model_file, value=0)


def test_plugin_contract(embedded, tmp_path, monkeypatch):
    from raw2features.core import plugins

    _source, store = embedded
    engine = _RecordingClusterer()
    (summary,) = cluster_store(
        store,
        out_dir=str(tmp_path / "c"),
        clusterer=engine,
        clusterer_inputs={"rna": "/data/rna.h5ad"},
        seed=5,
        gltf=False,
    )
    record = _load(summary)
    # Loaded once; one predict call with every location of the slide (all z planes).
    assert engine.loaded == 1
    ((shape, ctx),) = engine.calls
    assert shape[0] == len(record["feature_ids"])
    assert ctx.feature_ids == record["feature_ids"] and ctx.seed == 5
    assert ctx.inputs == {"rna": "/data/rna.h5ad"} and ctx.embedding_model == "mock"
    assert ctx.footprints.origin_zyx.shape == (shape[0], 3)
    np.testing.assert_array_equal(
        ctx.other_features("mock").shape, (shape[0], shape[1])
    )
    assert set(record["clusters"].values()) == {3}
    assert record["clustering_model"]["plugin"] == "recording"
    assert record["label_image"]["name"].endswith("_abababab")

    # Resolved by name, as an entry point would be; 0 (background) is rejected.
    registry = plugins._REGISTRY.setdefault("clusterers", {})
    monkeypatch.setitem(registry, "zero", _ZeroClusterer)
    with pytest.raises(ValueError, match=">= 1"):
        cluster_store(store, out_dir=str(tmp_path / "z"), clusterer="zero")
    assert {"trimodal", "placeholder", "zero"} <= set(plugins.available("clusterers"))


def test_label_image_mirrors_source_pyramid_with_footprint_quads(embedded, tmp_path):
    import ngff_zarr
    import zarr

    source, store = embedded
    image = zarr.open_group(source, mode="r+")
    image.require_group("labels").attrs["labels"] = ["existing"]
    image["labels"].require_group("existing")

    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer="placeholder", write_labels=True
    )
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
    assert [a["name"] for a in attrs["multiscales"][0]["axes"]] == ["t", "z", "y", "x"]
    for ds, out_ds in zip(
        src_ms["datasets"], attrs["multiscales"][0]["datasets"], strict=True
    ):
        assert out_ds["path"] == ds["path"]
        src_scale = ds["coordinateTransformations"][0]["scale"]
        out_scale = out_ds["coordinateTransformations"][0]["scale"]
        assert out_scale == [src_scale[0], *src_scale[2:]]
        src, out = image[ds["path"]], lab[ds["path"]]
        assert out.shape == (src.shape[0], *src.shape[2:])
        assert out.chunks == (src.chunks[0], *src.chunks[2:])
        assert out.metadata.dimension_separator == "/"
    ngff_zarr.from_ngff_zarr(f"{source}/labels/{name}")  # a valid NGFF image

    # Every level equals the footprints, side / downsample per x/y (non-overlapping
    # dense grid, z not downsampled): exact size and position, background elsewhere.
    sides = {tuple(f["size_zyx"]) for f in record["footprints"].values()}
    (side,) = {s[1] for s in sides}
    assert sides == {(1, side, side)} and side > 1
    for k, d in enumerate(src_ms["datasets"]):
        level = np.asarray(lab[d["path"]][0])
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


def _write_ngff_source(path, axes, shape, version, global_scale=None):
    """A two-level image with per-axis scales and translations."""
    import zarr

    zarr_format = 2 if version == "0.4" else 3
    root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    types = {"t": "time", "c": "channel", "ch": "channel"}
    datasets = []
    for k in range(2):
        factors = [2**k if a in ("y", "x") else 1 for a in axes]
        level = tuple(s // f for s, f in zip(shape, factors, strict=True))
        kwargs = {"dimension_names": axes} if zarr_format == 3 else {}
        root.create_array(str(k), shape=level, chunks=level, dtype="uint16", **kwargs)
        datasets.append(
            {
                "path": str(k),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(f) for f in factors]},
                    {"type": "translation", "translation": [0.0] * len(axes)},
                ],
            }
        )
    multiscales = {
        "name": "source",
        "axes": [{"name": a, "type": types.get(a, "space")} for a in axes],
        "datasets": datasets,
    }
    if global_scale is not None:
        multiscales["coordinateTransformations"] = [
            {"type": "scale", "scale": global_scale}
        ]
    if version == "0.4":
        root.attrs["multiscales"] = [{"version": "0.4", **multiscales}]
    else:
        root.attrs["ome"] = {"version": "0.5", "multiscales": [multiscales]}


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize(
    ("axes", "shape", "expected"),
    [
        (["t", "c", "z", "y", "x"], (2, 3, 4, 32, 32), ["t", "z", "y", "x"]),
        (["t", "ch", "z", "y", "x"], (2, 3, 4, 32, 32), ["t", "z", "y", "x"]),
        (["c", "z", "y", "x"], (3, 4, 32, 32), ["z", "y", "x"]),
        (["c", "y", "x"], (3, 32, 32), ["y", "x"]),
        (["z", "y", "x"], (4, 32, 32), ["z", "y", "x"]),
        (["y", "x"], (32, 32), ["y", "x"]),
    ],
)
def test_label_image_drops_only_the_channel_axis(
    tmp_path, version, axes, shape, expected
):
    import ngff_zarr
    import zarr

    source = str(tmp_path / "source.ome.zarr")
    _write_ngff_source(source, axes, shape, version)
    feet = Footprints(np.asarray([[0, 8, 16]]), np.asarray([[1, 8, 8]]))
    path = write_label_image(source, "clusters", feet, np.asarray([3]))

    lab = zarr.open_group(path, mode="r")
    attrs = dict(lab.attrs)
    ms = (attrs["ome"] if version == "0.5" else attrs)["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == expected
    keep = [axes.index(a) for a in expected]
    for k, ds in enumerate(ms["datasets"]):
        scale, translation = ds["coordinateTransformations"]
        factors = [float(2**k) if a in ("y", "x") else 1.0 for a in expected]
        assert scale["scale"] == factors
        assert translation["translation"] == [0.0] * len(expected)
        level = lab[ds["path"]]
        assert level.shape == tuple(
            int(shape[i] / f) for i, f in zip(keep, factors, strict=True)
        )
        if version == "0.5":
            assert level.metadata.dimension_names == tuple(expected)
        first = np.asarray(level[0] if "t" in expected else level[...])
        assert (first == 3).sum() == (8 // 2**k) ** 2
        if "t" in expected:  # footprints have no time: only t = 0 is written
            assert not np.asarray(level[1:]).any()
    ngff_zarr.from_ngff_zarr(path, validate=True)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_label_image_drops_channel_from_global_transforms(tmp_path, version):
    # Not checked with ngff_zarr validate=True: 0.37.1 crashes on any global transform.
    import zarr

    source = str(tmp_path / "source.ome.zarr")
    _write_ngff_source(
        source, ["c", "z", "y", "x"], (3, 4, 32, 32), version, [9.0, 2.0, 3.0, 4.0]
    )
    feet = Footprints(np.asarray([[0, 8, 16]]), np.asarray([[1, 8, 8]]))
    path = write_label_image(source, "clusters", feet, np.asarray([3]))

    attrs = dict(zarr.open_group(path, mode="r").attrs)
    ms = (attrs["ome"] if version == "0.5" else attrs)["multiscales"][0]
    assert ms["coordinateTransformations"] == [
        {"type": "scale", "scale": [2.0, 3.0, 4.0]}
    ]


def test_label_name_collision_needs_overwrite(embedded, tmp_path):
    _source, store = embedded
    kw = {"clusterer": "placeholder", "gltf": False}
    cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True, **kw)
    with pytest.raises(FileExistsError):
        cluster_store(store, out_dir=str(tmp_path / "c"), write_labels=True, **kw)
    cluster_store(
        store,
        clusterer="placeholder",
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
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer="placeholder"
    )
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
        app,
        [
            "cluster",
            store,
            "--out-dir",
            str(tmp_path / "c"),
            "--seed",
            "3",
            "--clusterer",
            "placeholder",
            "--clusterer-input",
            "rna=/data/rna.h5ad",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mock:" in result.output
    record = json.loads((tmp_path / "c" / "s3d.mock.clusters.json").read_text())
    assert record["clustering_model"]["plugin"] == "placeholder"
    assert record["clustering_model"]["inputs"] == {"rna": "/data/rna.h5ad"}
    assert (tmp_path / "c" / "s3d.mock.clusters.glb").exists()
    bad = CliRunner().invoke(app, ["cluster", store, "--clusterer-input", "rna"])
    assert bad.exit_code != 0


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
