"""Gather one embedding model's locations across every grid (z plane) of a store."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raw2features.core.store import GRIDS, grid_keys, open_root

from .footprint import Footprints, concat, grid_footprints


@dataclass
class ModelLocations:
    """Every embedded location of one model, in a stable (grid key, row) order.

    ``feature_ids[i]`` is ``"<grid_key>/<row>"`` and addresses
    ``grids/<grid_key>/features/<model>[row]``; ``centers_zyx[i]`` is the level-0
    ``(z, y, x)`` voxel at the patch centre (z = the grid's ``z_index``, 0 for 2D);
    ``footprints`` holds each patch's exact level-0 area (see ``footprint``).
    """

    model: str
    sha256: str
    sha256_kind: str  # "weights_sha256" | "output_fingerprint"
    output_fingerprint: str | None
    slide_id: str
    source_uri: str
    level_dimensions: list[list[int]]
    scale_um: dict[str, float | None]
    feature_ids: list[str] = field(default_factory=list)
    centers_zyx: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int64)
    )
    footprints: Footprints = field(default_factory=lambda: concat([]))
    # Per-grid feature blocks, in feature_ids order (kept per grid for memory).
    features: list[tuple[str, object]] = field(default_factory=list)


def store_models(store: str) -> list[str]:
    """Sorted names of every patch model with features in any grid of ``store``."""
    root = open_root(store)
    names: set[str] = set()
    for key in grid_keys(root):
        g = root[GRIDS][key]
        if "features" in g:
            names.update(g["features"].keys())
    return sorted(names)


def _model_identity(entry: dict) -> tuple[str, str, str | None]:
    fingerprint = (entry.get("output_fingerprint") or {}).get("digest")
    if entry.get("weights_sha256"):
        return str(entry["weights_sha256"]), "weights_sha256", fingerprint
    if fingerprint:
        return str(fingerprint), "output_fingerprint", fingerprint
    raise ValueError("model header records neither weights_sha256 nor a fingerprint")


def collect_locations(store: str, model: str) -> ModelLocations:
    """Collect ``model``'s locations over all grids; its identity must agree."""
    root = open_root(store)
    out: ModelLocations | None = None
    ids: list[str] = []
    centers: list[np.ndarray] = []
    boxes: list[Footprints] = []
    for key in grid_keys(root):
        g = root[GRIDS][key]
        if "features" not in g or model not in g["features"]:
            continue
        header = dict(g.attrs)["raw2features"]
        entry = header["models"][model]
        sha, kind, fingerprint = _model_identity(entry)
        source, patching = header["source"], header["patching"]
        if out is None:
            scale = dict(source.get("scale_um") or {})
            scale.setdefault("z", source.get("z_scale_um"))
            out = ModelLocations(
                model=model,
                sha256=sha,
                sha256_kind=kind,
                output_fingerprint=fingerprint,
                slide_id=source["slide_id"],
                source_uri=source["uri"],
                level_dimensions=[list(d) for d in source["level_dimensions"]],
                scale_um=scale,
            )
        elif sha != out.sha256:
            raise ValueError(
                f"{model}: grid {key!r} was embedded with different weights "
                f"({sha[-8:]} vs {out.sha256[-8:]}); cluster each separately"
            )
        coords = np.asarray(g["coords"], dtype=np.int64).reshape(-1, 2)
        side = int(patching["level0_patch"])
        half = side // 2
        z = int(patching.get("z_index") or 0)
        boxes.append(grid_footprints(coords, side, z))
        block = np.empty((coords.shape[0], 3), dtype=np.int64)
        block[:, 0] = z
        block[:, 1] = coords[:, 1] + half
        block[:, 2] = coords[:, 0] + half
        centers.append(block)
        ids.extend(f"{key}/{row}" for row in range(coords.shape[0]))
        out.features.append((key, g["features"][model]))
    if out is None:
        raise ValueError(f"model {model!r} has no features in {store!r}")
    out.feature_ids = ids
    out.footprints = concat(boxes)
    out.centers_zyx = (
        np.concatenate(centers) if centers else np.zeros((0, 3), dtype=np.int64)
    )
    return out
