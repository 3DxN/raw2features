"""Write cluster results into the source OME-Zarr as an NGFF label image.

The label image ``<image>/labels/<name>/`` mirrors the source multiscales: same axes,
pyramid level count, per-level shapes (channel axis collapsed to 1), chunking and
``coordinateTransformations``, so it overlays the image at every resolution. Each
embedded location is its patch footprint (``footprint``): an axis-aligned quad, one
plane thick, covering exactly the level-0 voxels the embedding was computed from. A
coarser level marks the voxels whose centres fall inside that area (level-0 extent /
downsample per axis, so the apparent size is constant across the hierarchy), or the
one voxel containing it when it is thinner than a level voxel (z). Only each quad's
bounding box is written, so whole-slide volumes are never materialised.

Footprints are in level-0 voxel units for now; see the NOTE in ``footprint`` for
what changes when they move to physical units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

from .footprint import Footprints

SPATIAL = ("z", "y", "x")


def label_name(embedding_sha256: str, trimodal_sha256: str) -> str:
    """``clusters_<last 8 of embedding sha>_<last 8 of trimodal sha>``."""
    return f"clusters_{embedding_sha256[-8:]}_{trimodal_sha256[-8:]}"


def local_source_path(uri: str) -> str:
    """Filesystem path of a local OME-Zarr; remote sources are refused."""
    parts = urlsplit(uri)
    if parts.scheme in ("", "file"):
        return parts.path if parts.scheme == "file" else uri
    raise ValueError(
        f"cannot write labels into a remote source ({parts.scheme}://); pass a "
        "local copy with --source"
    )


@dataclass
class _Level:
    path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    downsample: dict[str, float]  # per spatial axis, level-0 px per level px
    offset: dict[str, float]  # per spatial axis, level px


def _transform(dataset: dict, kind: str, n: int) -> list[float] | None:
    for t in dataset.get("coordinateTransformations") or []:
        if isinstance(t, dict) and t.get("type") == kind:
            return [float(v) for v in t[kind]]
    return None if kind == "scale" else [0.0] * n


def _open_image(root_path: str):
    """``(image_group, prefix, attrs_key, multiscales)`` - NGFF 0.4 and 0.5 (``ome``).

    ``prefix`` is ``"0"`` for a bioformats2raw layout, else ``""``.
    """
    import zarr

    root = zarr.open_group(root_path, mode="r+")
    candidates = [("", root)]
    if "0" in root:
        candidates.append(("0", root["0"]))
    for prefix, group in candidates:
        attrs = dict(group.attrs)
        if "multiscales" in attrs:
            return group, prefix, None, attrs["multiscales"][0]
        ome = attrs.get("ome")
        if isinstance(ome, dict) and "multiscales" in ome:
            return group, prefix, "ome", ome["multiscales"][0]
    raise ValueError(f"{root_path}: no OME-NGFF multiscales metadata found")


def _levels(image, multiscales: dict, axes: list[str]) -> list[_Level]:
    n = len(axes)
    datasets = multiscales["datasets"]
    s0 = _transform(datasets[0], "scale", n)
    t0 = _transform(datasets[0], "translation", n)
    shape0 = image[datasets[0]["path"]].shape
    levels = []
    for d in datasets:
        arr = image[d["path"]]
        sl = _transform(d, "scale", n)
        tl = _transform(d, "translation", n)
        ds, off = {}, {}
        for i, ax in enumerate(axes):
            if ax not in SPATIAL:
                continue
            if s0 is not None and sl is not None and s0[i]:
                ds[ax] = sl[i] / s0[i]
                off[ax] = (t0[i] - tl[i]) / sl[i] if sl[i] else 0.0
            else:  # no usable scale: fall back to the shape ratio
                ds[ax] = shape0[i] / max(arr.shape[i], 1)
                off[ax] = 0.0
        levels.append(_Level(d["path"], tuple(arr.shape), tuple(arr.chunks), ds, off))
    return levels


def write_label_image(
    source_path: str,
    name: str,
    footprints: Footprints,
    clusters: np.ndarray,
    *,
    expected_level0_wh: tuple[int, int] | None = None,
    overwrite: bool = False,
    extra_metadata: dict | None = None,
) -> str:
    """Rasterise every footprint into ``<image>/labels/<name>``; return its path."""
    import zarr

    image, prefix, ome_key, ms = _open_image(source_path)
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in ms["axes"]]
    levels = _levels(image, ms, axes)
    if expected_level0_wh is not None:
        w0 = levels[0].shape[axes.index("x")]
        h0 = levels[0].shape[axes.index("y")]
        if (w0, h0) != tuple(expected_level0_wh):
            raise ValueError(
                f"source level 0 is {w0}x{h0} but the embeddings were extracted "
                f"from {expected_level0_wh[0]}x{expected_level0_wh[1]}"
            )
    zarr_format = image.metadata.zarr_format
    version = (
        (dict(image.attrs).get("ome") or {}).get("version", "0.5")
        if ome_key
        else ms.get("version", "0.4")
    )

    labels = image.require_group("labels")
    labels_attrs = dict(labels.attrs)
    listing = (labels_attrs.get("ome") or {}) if ome_key else labels_attrs
    names = list(listing.get("labels") or [])
    if name in labels:
        if not overwrite:
            raise FileExistsError(
                f"label image {name!r} already exists; pass overwrite to replace it"
            )
        del labels[name]
    group = labels.create_group(name)

    max_value = int(clusters.max()) if clusters.size else 0
    dtype = next(
        t for t in ("uint8", "uint16", "uint32") if max_value <= np.iinfo(t).max
    )
    c_index = axes.index("c") if "c" in axes else None
    arrays = []
    for lv in levels:
        shape = list(lv.shape)
        chunks = list(lv.chunks)
        if c_index is not None:
            shape[c_index] = chunks[c_index] = 1
        kwargs: dict = {}
        if zarr_format == 2:
            kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
        else:
            kwargs["dimension_names"] = axes
        arrays.append(
            group.create_array(
                lv.path,
                shape=tuple(shape),
                chunks=tuple(chunks),
                dtype=dtype,
                fill_value=0,
                **kwargs,
            )
        )

    spatial = [ax for ax in axes if ax in SPATIAL]  # in source axis order
    for lv, arr in zip(levels, arrays, strict=True):
        _rasterise_level(arr, axes, spatial, lv, footprints, clusters)

    multiscales = {
        **{k: v for k, v in ms.items() if k not in ("datasets", "name")},
        "name": name,
        "datasets": ms["datasets"],
    }
    colors = [
        {"label-value": int(v)} for v in sorted({int(c) for c in clusters.tolist()})
    ]
    image_label = {
        "version": version,
        "colors": colors,
        "source": {"image": "../../"},
    }
    if ome_key:
        multiscales.pop("version", None)
        group.attrs["ome"] = {
            "version": version,
            "multiscales": [multiscales],
            "image-label": image_label,
        }
    else:
        group.attrs["multiscales"] = [multiscales]
        group.attrs["image-label"] = image_label
    if extra_metadata:
        group.attrs["raw2features"] = extra_metadata

    if name not in names:
        names.append(name)
    if ome_key:
        labels.attrs["ome"] = {**listing, "version": version, "labels": names}
    else:
        labels.attrs["labels"] = names
    try:
        zarr.consolidate_metadata(zarr.open_group(source_path, mode="r+").store)
    except Exception:  # noqa: BLE001 - consolidation is an optional read accelerator
        pass
    return os.path.join(source_path, prefix, "labels", name)


def _level_boxes(lv: _Level, spatial, footprints: Footprints):
    """Per-location ``[lo, hi)`` voxel ranges in one level, in ``spatial`` order.

    Level voxel ``j`` has its centre at level-0 index ``(j - offset) * downsample``
    and is kept when that lies in the footprint's ``[origin - 0.5, origin + size -
    0.5)``. An extent thinner than one level voxel keeps the voxel containing its
    middle, so a one-plane quad survives z downsampling.
    """
    origin = dict(zip(SPATIAL, footprints.origin_zyx.T.astype(np.float64), strict=True))
    size = dict(zip(SPATIAL, footprints.size_zyx.T.astype(np.float64), strict=True))
    lo, hi = [], []
    for ax in spatial:
        ds, off = lv.downsample[ax], lv.offset[ax]
        a = origin[ax] - 0.5
        b = a + size[ax]
        lo_k = np.ceil(a / ds + off)
        hi_k = np.ceil(b / ds + off)
        thin = lo_k >= hi_k
        mid = np.floor((a + b) / 2.0 / ds + off + 0.5)
        lo_k = np.where(thin, mid, lo_k)
        hi_k = np.where(thin, mid + 1, hi_k)
        lo.append(lo_k)
        hi.append(hi_k)
    return np.stack(lo, 1).astype(np.int64), np.stack(hi, 1).astype(np.int64)


def _rasterise_level(arr, axes, spatial, lv: _Level, footprints, clusters):
    """Stamp every location's quad into one level; overlaps go to the nearest centre.

    Grids with a stride below the patch side, or z planes merged by a coarser level,
    make quads overlap. A contested voxel takes the location it is relatively closest
    to (Euclidean distance in units of each box's half-extent, so equal boxes split
    along their bisector), so every location keeps its own centre; ties go to the
    lower feature index.
    """
    n = len(clusters)
    if n == 0:
        return
    extents = np.asarray([lv.shape[axes.index(ax)] for ax in spatial])
    lo, hi = _level_boxes(lv, spatial, footprints)
    centre = (lo + hi - 1) / 2.0
    half = (hi - lo) / 2.0
    lo, hi = np.maximum(lo, 0), np.minimum(hi, extents)
    # Boxes that intersect have origins less than one (maximal) box apart.
    cell_size = np.maximum((hi - lo).max(axis=0), 1)
    cells = np.floor_divide(lo, cell_size)
    buckets: dict[tuple, list[int]] = {}
    for i, cell in enumerate(map(tuple, cells)):
        buckets.setdefault(cell, []).append(i)
    neighbours = np.stack(
        np.meshgrid(*([(-1, 0, 1)] * len(spatial)), indexing="ij"), -1
    ).reshape(-1, len(spatial))
    for i in range(n):
        if (lo[i] >= hi[i]).any():
            continue  # outside this level
        rivals = [
            j
            for off in neighbours
            for j in buckets.get(tuple(cells[i] + off), ())
            if j != i and (lo[j] < hi[i]).all() and (lo[i] < hi[j]).all()
        ]
        index = tuple(
            slice(int(lo[i, spatial.index(ax)]), int(hi[i, spatial.index(ax)]))
            if ax in spatial
            else 0
            for ax in axes
        )
        if not rivals:
            arr[index] = int(clusters[i])
            continue
        grids = np.meshgrid(
            *(np.arange(a, b) for a, b in zip(lo[i], hi[i], strict=True)),
            indexing="ij",
        )

        def dist(j: int, grids=grids) -> np.ndarray:
            return sum(
                ((g - centre[j, k]) / half[j, k]) ** 2 for k, g in enumerate(grids)
            )

        d_own = dist(i)
        keep = np.ones(d_own.shape, dtype=bool)
        for j in rivals:
            inside = np.all(
                [(g >= lo[j, k]) & (g < hi[j, k]) for k, g in enumerate(grids)], axis=0
            )
            d_j = np.where(inside, dist(j), np.inf)
            keep &= (d_own < d_j) | ((d_own == d_j) & (i < j))
        if not keep.any():
            continue
        block = np.asarray(arr[index])
        block[keep] = int(clusters[i])
        arr[index] = block
