"""Cluster every embedded location of a store and write JSON / labels / glTF outputs."""

from __future__ import annotations

import json
import os

import numpy as np

from .collect import collect_locations, store_models
from .labels import label_name, local_source_path, write_label_image
from .trimodal import predict_clusters, resolve_trimodal_sha256

CLUSTERS_FORMAT = "raw2features-clusters/0.2"


def cluster_store(
    store: str,
    *,
    models: list[str] | None = None,
    out_dir: str | None = None,
    trimodal_model: str | None = None,
    seed: int = 0,
    write_labels: bool = False,
    source: str | None = None,
    labels_overwrite: bool = False,
    gltf: bool = True,
) -> list[dict]:
    """Cluster each model's locations; return one summary per model."""
    from .gltf import write_glb

    store = os.path.normpath(store)
    out_dir = out_dir or os.path.dirname(os.path.abspath(store))
    os.makedirs(out_dir, exist_ok=True)
    trimodal_sha, placeholder = resolve_trimodal_sha256(trimodal_model)
    rng = np.random.default_rng(seed)
    summaries = []
    for model in models or store_models(store):
        loc = collect_locations(store, model)
        clusters = np.concatenate(
            [predict_clusters(np.asarray(f), rng) for _key, f in loc.features]
            or [np.zeros(0, dtype=np.int32)]
        ).astype(np.int64)
        name = label_name(loc.sha256, trimodal_sha)
        stem = os.path.join(out_dir, f"{loc.slide_id}.{model}.clusters")
        identity = {
            "embedding_model": {
                "name": model,
                "sha256": loc.sha256,
                "sha256_kind": loc.sha256_kind,
                "output_fingerprint": loc.output_fingerprint,
            },
            "trimodal_model": {
                "sha256": trimodal_sha,
                "path": os.path.abspath(trimodal_model) if trimodal_model else None,
                "placeholder": placeholder,
            },
        }

        label_path = None
        if write_labels:
            label_path = write_label_image(
                local_source_path(source or loc.source_uri),
                name,
                loc.footprints,
                clusters,
                expected_level0_wh=tuple(loc.level_dimensions[0]),
                overwrite=labels_overwrite,
                extra_metadata={"format": CLUSTERS_FORMAT, **identity},
            )

        glb_path = None
        if gltf:
            glb_path = write_glb(
                f"{stem}.glb",
                loc.feature_ids,
                loc.footprints,
                extras={
                    "scale_um": loc.scale_um,
                    "embedding_sha256": loc.sha256,
                    "trimodal_sha256": trimodal_sha,
                },
            )

        record = {
            "format": CLUSTERS_FORMAT,
            "slide_id": loc.slide_id,
            "source_uri": loc.source_uri,
            "store_uri": f"file://{os.path.abspath(store)}",
            **identity,
            "seed": seed,
            # Patch footprints in level-0 voxels (pixel units for now; see the NOTE
            # in clustering.footprint for the move to physical units).
            "footprint_units": "level0_voxels",
            "label_image": (
                {"name": name, "path": label_path} if label_path else {"name": name}
            ),
            "gltf": os.path.basename(glb_path) if glb_path else None,
            # Order of feature_ids == glTF mesh/node order.
            "feature_ids": loc.feature_ids,
            "clusters": dict(
                zip(loc.feature_ids, (int(c) for c in clusters), strict=True)
            ),
            "locations_zyx": dict(
                zip(loc.feature_ids, loc.centers_zyx.tolist(), strict=True)
            ),
            "footprints": {
                fid: {"origin_zyx": origin, "size_zyx": size}
                for fid, origin, size in zip(
                    loc.feature_ids,
                    loc.footprints.origin_zyx.tolist(),
                    loc.footprints.size_zyx.tolist(),
                    strict=True,
                )
            },
        }
        json_path = f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
        summaries.append(
            {
                "model": model,
                "n": len(loc.feature_ids),
                "json": json_path,
                "gltf": glb_path,
                "labels": label_path,
            }
        )
    return summaries
