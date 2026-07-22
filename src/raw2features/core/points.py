"""Point-cloud loading and placement for 3D point-sampled extraction.

A point cloud is a table of 3D coordinates (physical µm in the source frame, one
row per point, columns ``x, y, z`` plus an optional ``id``/``label``). It replaces
the tissue-driven grid as the patch selector: raw2features reads one 2D patch at
each point's ``(x, y)`` on its ``z``-plane. This module is the single place that
knows the file format and the µm→pixel/plane mapping, so the sampler and the CLI
stay format-agnostic.

CSV loads with the stdlib (no extra deps); Parquet lazily imports ``pyarrow`` (the
``tables`` extra) and raises an actionable error when it is missing.
"""

from __future__ import annotations

import csv
import hashlib
import os
import warnings
from dataclasses import dataclass

import numpy as np

from raw2features.core.uris import source_uri

_REQUIRED = ("x", "y", "z")
_ID_COLUMNS = ("id", "label", "name")


@dataclass(frozen=True)
class PointCloud:
    """A validated 3D point cloud in physical µm.

    ``xyz_um`` is ``(N, 3)`` float64 ``(x, y, z)`` in micrometers; ``ids`` is an
    ``(N,)`` array of string ids (empty strings when the source carried none);
    ``sha256`` fingerprints the source file for the content hash / provenance.
    """

    xyz_um: np.ndarray
    ids: np.ndarray
    source: str
    sha256: str

    def __len__(self) -> int:
        return int(self.xyz_um.shape[0])


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def point_cloud_fingerprint(path: str) -> str:
    """sha256 of the point-cloud file (its content identity for the config hash).

    Cheaper than a full load: it hashes bytes without parsing, so the CLI can fold the
    cloud's identity into the run hash up front while the sampler does the real load.
    """
    local = str(path).removeprefix("file://")
    if not os.path.isfile(local):
        raise FileNotFoundError(f"point cloud not found: {source_uri(path)}")
    return _file_sha256(local)


def load_point_cloud(path: str) -> PointCloud:
    """Load a CSV or Parquet point cloud into a :class:`PointCloud` (µm).

    The extension selects the reader (``.parquet``/``.pq`` -> pyarrow; anything else
    -> CSV). Required columns are ``x``, ``y``, ``z`` (case-insensitive); an optional
    id column (``id``/``label``/``name``) is carried through. Rows with a
    non-finite/missing coordinate are a hard error (a corrupt cloud must not embed
    silently at the wrong place).
    """
    local = str(path).removeprefix("file://")
    if not os.path.isfile(local):
        raise FileNotFoundError(f"point cloud not found: {source_uri(path)}")
    ext = os.path.splitext(local)[1].casefold()
    if ext in (".parquet", ".pq"):
        cols, ids = _read_parquet(local)
    else:
        cols, ids = _read_csv(local)
    missing = [c for c in _REQUIRED if c not in cols]
    if missing:
        raise ValueError(
            f"{source_uri(path)}: point cloud is missing column(s) {missing}; "
            f"expected x, y, z (found {sorted(cols)})."
        )
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    if xyz.size and not np.isfinite(xyz).all():
        raise ValueError(
            f"{source_uri(path)}: point cloud has non-finite x/y/z coordinates."
        )
    if ids is None:
        ids = np.array([""] * xyz.shape[0], dtype=object)
    return PointCloud(
        xyz_um=xyz, ids=ids, source=source_uri(path), sha256=_file_sha256(local)
    )


def _read_csv(path: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        field_map = {
            (name or "").strip().casefold(): name for name in reader.fieldnames or []
        }
        # Read only the required columns that are actually present; load_point_cloud
        # reports any missing one with a clear, aggregated error (not a raw KeyError).
        present = [c for c in _REQUIRED if c in field_map]
        id_key = next((k for k in _ID_COLUMNS if k in field_map), None)
        raw: dict[str, list[float]] = {c: [] for c in present}
        ids: list[str] = []
        for row in reader:
            for c in present:
                raw[c].append(float(row[field_map[c]]))
            if id_key is not None:
                ids.append(str(row[field_map[id_key]]))
    cols = {c: np.asarray(v, dtype=np.float64) for c, v in raw.items()}
    return cols, (np.asarray(ids, dtype=object) if id_key is not None else None)


def _read_parquet(path: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading a Parquet point cloud needs the 'tables' extra: "
            'pip install "raw2features[tables]" (or convert the cloud to CSV).'
        ) from exc
    table = pq.read_table(path)
    lower = {name.casefold(): name for name in table.column_names}
    cols = {
        c: np.asarray(table.column(lower[c]).to_numpy(zero_copy_only=False))
        for c in _REQUIRED
        if c in lower
    }
    id_key = next((k for k in _ID_COLUMNS if k in lower), None)
    ids = (
        np.asarray(table.column(lower[id_key]).to_pylist(), dtype=object)
        if id_key is not None
        else None
    )
    return cols, ids


def map_points_to_grid(
    points: PointCloud, reader, level0_patch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place µm points onto the source: patch top-left (level-0 px) + z-plane index.

    The point is the patch **center**: ``x_px = round(x_um / scale_um['x'])`` (and y
    likewise) gives the level-0 center, and the top-left stored in ``coords`` is
    ``center - level0_patch // 2``. ``z`` maps to a plane via
    ``reader.z_index_for_um``. Points whose center falls outside the level-0 extent
    are dropped (and warned), so an off-slide cloud does not embed padded background.

    Returns ``(coords, z_index, kept_points_um, kept_ids)`` where ``coords`` is
    ``(M, 2)`` int32 top-left and ``kept_*`` are the surviving points 1:1 with it.
    """
    scale = reader.scale_um
    if not scale or "x" not in scale or "y" not in scale:
        raise ValueError(
            "source physical pixel size is unknown; point coordinates are in µm and "
            "cannot be placed. Pass --source-mpp (and --source-z-spacing) or fix the "
            "source axis units."
        )
    xyz = points.xyz_um
    n = xyz.shape[0]
    if n == 0:
        empty_i = np.zeros((0, 2), dtype=np.int32)
        return empty_i, np.zeros((0,), dtype=np.int32), xyz.reshape(0, 3), points.ids

    cx = np.round(xyz[:, 0] / float(scale["x"])).astype(np.int64)
    cy = np.round(xyz[:, 1] / float(scale["y"])).astype(np.int64)
    z_index = np.array(
        [reader.z_index_for_um(float(z)) for z in xyz[:, 2]], dtype=np.int64
    )

    w0, h0 = reader.level_dimensions[0]
    in_bounds = (cx >= 0) & (cx < w0) & (cy >= 0) & (cy < h0)
    dropped = int((~in_bounds).sum())
    if dropped:
        warnings.warn(
            f"{dropped} of {n} points fall outside the slide's level-0 extent "
            f"({w0}x{h0} px) and are dropped.",
            stacklevel=2,
        )

    half = level0_patch // 2
    coords = np.stack([cx - half, cy - half], axis=1).astype(np.int32)
    coords = coords[in_bounds]
    z_index = z_index[in_bounds].astype(np.int32)
    kept_points = xyz[in_bounds]
    kept_ids = points.ids[in_bounds]
    return coords, z_index, kept_points, kept_ids
