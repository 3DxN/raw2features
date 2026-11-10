"""``raw2features cluster``: clusterer plugins -> JSON, labels, glTF."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import warnings
from collections import Counter

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


def _grids_digest(store: str) -> str:
    """sha256 over the sorted ``grid_hash`` of every grid in ``store``."""
    from raw2features.core.store import GRIDS, grid_keys, open_root

    root = open_root(store)
    hashes = sorted(
        dict(root[GRIDS][k].attrs)["raw2features"]["grid_hash"] for k in grid_keys(root)
    )
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


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
        emb["sha256"], PLACEHOLDER_TRIMODAL_SHA256, _grids_digest(store)
    )
    assert record["label_image"]["name"].split("_")[-1] == _grids_digest(store)[-8:]
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
    assert record["label_image"]["name"].split("_")[2] == "abababab"

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
        overwrite=True,
        gltf=False,
    )


def test_label_listing_is_read_from_disk_not_stale_consolidated_metadata(
    embedded, tmp_path
):
    # A label listed after the root was last consolidated must survive a new label.
    import zarr

    source, store = embedded
    image = zarr.open_group(source, mode="r+")
    labels = image.require_group("labels")
    labels.require_group("old")
    labels.attrs["labels"] = ["old"]
    zarr.consolidate_metadata(image.store)
    labels.require_group("manual")
    labels.attrs["labels"] = ["old", "manual"]  # the consolidated copy still says old

    (summary,) = cluster_store(
        store,
        out_dir=str(tmp_path / "c"),
        clusterer="placeholder",
        write_labels=True,
        gltf=False,
    )
    name = _load(summary)["label_image"]["name"]
    on_disk = zarr.open_group(f"{source}/labels", mode="r", use_consolidated=False)
    assert dict(on_disk.attrs)["labels"] == ["old", "manual", name]
    consolidated = zarr.open_group(source, mode="r")  # refreshed afterwards
    assert dict(consolidated["labels"].attrs)["labels"] == ["old", "manual", name]


def test_properties_map_glyph_label_values_to_clusters(embedded, tmp_path):
    _source, store = embedded
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer="placeholder"
    )
    record = _load(summary)
    assert summary["properties"] == str(
        tmp_path / "c" / "s3d.mock.clusters.properties.json"
    )
    assert record["properties"] == "s3d.mock.clusters.properties.json"
    assert record["mesh_index"] is None
    with open(summary["properties"], encoding="utf-8") as fh:
        properties = json.load(fh)
    doc = _read_glb(summary["gltf"])
    assert properties == [
        {"label-value": i + 1, "cluster": record["clusters"][fid], "feature_id": fid}
        for i, fid in enumerate(record["feature_ids"])
    ]
    assert list(properties[0]) == ["label-value", "cluster", "feature_id"]
    assert [n["extras"]["label-value"] for n in doc["nodes"]] == [
        p["label-value"] for p in properties
    ]


def test_write_meshes_lists_scene_and_properties_in_source(embedded, tmp_path):
    import zarr

    source, store = embedded
    image = zarr.open_group(source, mode="r+")
    image.require_group("meshes").attrs["meshes"] = [
        {"name": "nuclei", "path": "nuclei.glb"}
    ]
    kw = {"out_dir": str(tmp_path / "c"), "clusterer": "placeholder"}
    (summary,) = cluster_store(store, write_meshes=True, **kw)
    record = _load(summary)
    name = record["mesh_index"]["name"]
    assert name == record["label_image"]["name"]  # one run name for both
    meshes = os.path.join(source, "meshes")
    assert summary["meshes"] == record["mesh_index"]["path"] == f"{meshes}/{name}.glb"
    listing = dict(zarr.open_group(meshes, mode="r", use_consolidated=False).attrs)
    assert listing["meshes"] == [
        {"name": "nuclei", "path": "nuclei.glb"},
        {
            "name": name,
            "path": f"{name}.glb",
            "properties": f"{name}.properties.json",
            "embeddings": {
                "path": f"{name}.embeddings.zarr",
                "model": "mock",
                "sha256": record["embedding_model"]["sha256"],
                "sha256_kind": record["embedding_model"]["sha256_kind"],
            },
        },
    ]
    # Row label-value - 1 is that glyph's vector, as stored under its feature ID.
    assert record["mesh_index"]["embeddings"] == f"{meshes}/{name}.embeddings.zarr"
    vectors = zarr.open_array(f"{meshes}/{name}.embeddings.zarr", mode="r")
    assert vectors.dtype == np.float32 and vectors.compressors == ()
    assert (
        dict(vectors.attrs)["raw2features"]["sha256"]
        == record["embedding_model"]["sha256"]
    )
    from raw2features.core.store import GRIDS, open_root

    root = open_root(store)
    expected = np.stack(
        [
            np.asarray(root[GRIDS][key]["features"]["mock"][int(row)])
            for key, row in (fid.rsplit("/", 1) for fid in record["feature_ids"])
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(vectors), expected)
    with open(f"{meshes}/{name}.glb", "rb") as a, open(summary["gltf"], "rb") as b:
        assert a.read() == b.read()
    with open(f"{meshes}/{name}.properties.json", encoding="utf-8") as fh:
        assert json.load(fh)[0]["label-value"] == 1

    # A second run of the same store collides before anything is written ...
    with pytest.raises(FileExistsError, match="mesh"):
        cluster_store(store, write_meshes=True, write_labels=True, **kw)
    assert "labels" not in zarr.open_group(source, mode="r", use_consolidated=False)
    # ... and overwrite replaces the entry in place.
    cluster_store(store, write_meshes=True, overwrite=True, **kw)
    listing = dict(zarr.open_group(meshes, mode="r", use_consolidated=False).attrs)
    assert [e["name"] for e in listing["meshes"]] == ["nuclei", name]

    with pytest.raises(ValueError, match="glTF"):
        cluster_store(store, write_meshes=True, gltf=False, **kw)


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


def _accessor(path: str, doc: dict, index: int) -> np.ndarray:
    """An accessor's data from the .glb BIN chunk (tightly packed views)."""
    data = open(path, "rb").read()
    js_len = struct.unpack_from("<I", data, 12)[0]
    binary = data[20 + js_len + 8 :]
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3}[acc["type"]]
    raw = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    return np.frombuffer(raw, dtype=dtype).reshape(acc["count"], width)


def test_gltf_marker_glyph_is_centre_ellipsoid_and_diagonal_lines(embedded, tmp_path):
    _source, store = embedded
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer="placeholder", gltf_glyph="marker"
    )
    record = _load(summary)
    path = summary["gltf"]
    doc = _read_glb(path)
    ids = record["feature_ids"]
    assert [m["name"] for m in doc["meshes"]] == ids
    (side,) = {f["size_zyx"][1] for f in record["footprints"].values()}
    for node in doc["nodes"]:  # translations are still the footprint centres
        z, y0, x0 = record["footprints"][node["name"]]["origin_zyx"]
        assert node["translation"] == [x0 + (side - 1) / 2, y0 + (side - 1) / 2, z]

    # One shared geometry: a closed ellipsoid and two diagonal prisms, all triangles.
    (prims,) = {json.dumps(m["primitives"]) for m in doc["meshes"]}
    sphere, cross = json.loads(prims)
    assert (sphere["mode"], cross["mode"]) == (4, 4)
    r = 0.02 * side
    verts = _accessor(path, doc, sphere["attributes"]["POSITION"])
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), r, rtol=1e-5)
    faces = _accessor(path, doc, sphere["indices"]).reshape(-1, 3)
    a, b, c = (verts[faces[:, k]] for k in range(3))
    outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c)
    assert (outward > 0).all()  # counter-clockwise seen from outside, none degenerate
    uses = Counter(
        tuple(sorted(e))
        for f in faces.tolist()
        for e in zip(f, f[1:] + f[:1], strict=True)
    )
    assert set(uses.values()) == {2}  # closed: every edge shared by two faces
    normals = _accessor(path, doc, sphere["attributes"]["NORMAL"])
    np.testing.assert_allclose(normals, verts / r, atol=1e-5)

    # Prisms along the diagonals, by default 35% of the way to each corner.
    _check_cross(path, doc, cross, side, arm=0.35)
    assert doc["materials"] == [{"name": "marker"}]
    assert "ellipsoid" in doc["asset"]["extras"]["geometry"]


def _check_cross(path: str, doc: dict, cross: dict, side: float, arm: float) -> None:
    """Two closed, outward-wound triangular prisms of circumradius 0.005 x ``side``,
    along the footprint diagonals, ``arm`` of the way from the centre to a corner."""
    verts = _accessor(path, doc, cross["attributes"]["POSITION"]).astype(np.float64)
    normals = _accessor(path, doc, cross["attributes"]["NORMAL"])
    faces = _accessor(path, doc, cross["indices"]).reshape(-1, 3)
    assert len(faces) == 16  # 2 prisms x (3 sides x 2 + 2 caps)
    tris = verts[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    # Each prism is centred on the origin, so outward faces point away from it.
    assert (np.einsum("ij,ij->i", face_normals, tris.mean(1)) > 0).all()
    unit = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    np.testing.assert_allclose(
        normals[faces], np.repeat(unit[:, None], 3, 1), atol=1e-5
    )
    half = 24  # vertices per prism
    for k, direction in enumerate([(1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)]):
        d = np.asarray(direction) / np.sqrt(2.0)
        part = verts[k * half : (k + 1) * half]
        t = part @ d
        np.testing.assert_allclose(np.abs(t), arm * side / np.sqrt(2.0), rtol=1e-5)
        dist = np.linalg.norm(part - t[:, None] * d, axis=1)
        np.testing.assert_allclose(dist, 0.005 * side, rtol=1e-4)
        corners = {tuple(np.round(v, 3)) for v in part.tolist()}
        assert len(corners) == 6  # closed: every face corner is a prism corner
        edges = Counter(
            tuple(sorted((tuple(np.round(a, 3)), tuple(np.round(b, 3)))))
            for tri in tris[8 * k : 8 * (k + 1)]
            for a, b in zip(tri, np.roll(tri, -1, 0), strict=True)
        )
        assert set(edges.values()) == {2}  # watertight


@pytest.mark.parametrize("glyph", ["quad", "marker"])
def test_each_glyph_is_one_mesh_addressable_by_feature_id(embedded, tmp_path, glyph):
    # A viewer colours glyph i individually through node i / mesh i, found by feature
    # ID: every primitive of the glyph is in its mesh, and no other node uses it.
    _source, store = embedded
    (summary,) = cluster_store(
        store, out_dir=str(tmp_path / "c"), clusterer="placeholder", gltf_glyph=glyph
    )
    ids = _load(summary)["feature_ids"]
    doc = _read_glb(summary["gltf"])
    assert len(set(ids)) == len(ids)
    assert doc["scenes"][0]["nodes"] == list(range(len(ids)))
    for i, (fid, node, mesh) in enumerate(
        zip(ids, doc["nodes"], doc["meshes"], strict=True)
    ):
        assert node["mesh"] == i
        assert node["name"] == mesh["name"] == fid
        assert (
            node["extras"]
            == mesh["extras"]
            == {
                "feature_id": fid,
                "label-value": i + 1,
            }
        )
        assert "children" not in node
        assert len(mesh["primitives"]) == (1 if glyph == "quad" else 2)


def test_bad_gltf_options_are_refused_before_clustering(embedded, tmp_path):
    _source, store = embedded
    for kw, match in (
        ({"gltf_glyph": "x"}, "glyph"),
        ({"gltf_glyph": "marker", "gltf_cross_arm": 0.0}, "cross arm"),
        ({"gltf_glyph": "marker", "gltf_cross_arm": 1.5}, "cross arm"),
    ):
        with pytest.raises(ValueError, match=match):
            cluster_store(
                store, out_dir=str(tmp_path / "c"), clusterer="placeholder", **kw
            )
    assert not list((tmp_path / "c").glob("*.json"))


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
    marker = CliRunner().invoke(
        app,
        ["cluster", store, "--out-dir", str(tmp_path / "m"), "--clusterer",
         "placeholder", "--gltf-glyph", "marker", "--gltf-cross-arm", "1"],
    )  # fmt: skip
    assert marker.exit_code == 0, marker.output
    path = str(tmp_path / "m" / "s3d.mock.clusters.glb")
    doc = _read_glb(path)
    assert doc["materials"] == [{"name": "marker"}]
    side = record["footprints"][record["feature_ids"][0]]["size_zyx"][1]
    cross = doc["meshes"][0]["primitives"][1]
    _check_cross(path, doc, cross, side, arm=1.0)  # the full diagonals
    meshes = CliRunner().invoke(
        app,
        ["cluster", store, "--out-dir", str(tmp_path / "w"), "--clusterer",
         "placeholder", "--write-meshes"],
    )  # fmt: skip
    assert meshes.exit_code == 0, meshes.output
    assert "properties:" in meshes.output and "meshes:" in meshes.output
    unknown = CliRunner().invoke(app, ["cluster", store, "--gltf-glyph", "x"])
    assert unknown.exit_code == 1 and "glyph" in unknown.output
    no_arm = CliRunner().invoke(
        app, ["cluster", store, "--gltf-glyph", "marker", "--gltf-cross-arm", "0"]
    )
    assert no_arm.exit_code == 1 and "cross arm" in no_arm.output
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
