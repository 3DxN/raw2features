"""Cluster every embedded location of a store; write JSON, glTF, properties, labels
and the mesh index."""

from __future__ import annotations

import json
import os

import numpy as np

from raw2features.core import plugins

from . import trimodal  # noqa: F401 - registers the built-in clusterers
from .base import ClusterContext, Clusterer, check_clusters
from .collect import collect_locations, store_models
from .labels import label_name, local_source_path, write_label_image
from .meshes import check_mesh_slot, write_mesh_index

CLUSTERS_FORMAT = "raw2features-clusters/0.2"


def cluster_store(
    store: str,
    *,
    models: list[str] | None = None,
    out_dir: str | None = None,
    clusterer: str | Clusterer = "trimodal",
    clusterer_model: str | None = None,
    clusterer_inputs: dict[str, str] | None = None,
    seed: int = 0,
    write_labels: bool = False,
    source: str | None = None,
    overwrite: bool = False,
    gltf: bool = True,
    write_meshes: bool = False,
    gltf_glyph: str = "quad",
    gltf_cross_arm: float | None = None,
) -> list[dict]:
    """Cluster each model's locations; return one summary per model.

    ``clusterer`` is a registered ``raw2features.clusterers`` name (constructed with
    ``clusterer_model``, ``None`` = its default model file) or an instance.
    ``gltf_glyph`` is the glTF primitive per location (``gltf.GLYPHS``) and
    ``gltf_cross_arm`` the ``marker`` crosshair arm (``None`` = ``gltf.CROSS_ARM``).
    ``write_labels`` / ``write_meshes`` add the label image / the glTF scene and its
    properties to the source; ``overwrite`` replaces an existing one of that name.
    """
    from .gltf import CROSS_ARM, check_glyph, write_glb

    cross_arm = CROSS_ARM if gltf_cross_arm is None else gltf_cross_arm
    if gltf:
        check_glyph(gltf_glyph, cross_arm)  # before clustering or writing anything
    if write_meshes and not gltf:
        raise ValueError("write_meshes needs the glTF scene; do not disable gltf")
    store = os.path.normpath(store)
    out_dir = out_dir or os.path.dirname(os.path.abspath(store))
    os.makedirs(out_dir, exist_ok=True)
    engine = (
        clusterer
        if isinstance(clusterer, Clusterer)
        else plugins.get("clusterers", clusterer)(clusterer_model)
    )
    engine.load()
    ident = engine.identity()
    summaries = []
    for model in models or store_models(store):
        loc = collect_locations(store, model)
        features = np.concatenate([np.asarray(f) for _key, f in loc.features])
        context = ClusterContext(
            slide_id=loc.slide_id,
            store=store,
            source_uri=loc.source_uri,
            embedding_model=model,
            embedding_sha256=loc.sha256,
            feature_ids=loc.feature_ids,
            footprints=loc.footprints,
            scale_um=loc.scale_um,
            seed=seed,
            inputs=dict(clusterer_inputs or {}),
        )
        clusters = check_clusters(
            engine.predict(features, context),
            len(loc.feature_ids),
            plugin=ident.plugin,
        )
        name = label_name(loc.sha256, ident.model_sha256, loc.grids_digest)
        stem = os.path.join(out_dir, f"{loc.slide_id}.{model}.clusters")
        identity = {
            "embedding_model": {
                "name": model,
                "sha256": loc.sha256,
                "sha256_kind": loc.sha256_kind,
                "output_fingerprint": loc.output_fingerprint,
            },
            "clustering_model": {
                "plugin": ident.plugin,
                "version": ident.version,
                "sha256": ident.model_sha256,
                "path": ident.model_path,
                "placeholder": ident.placeholder,
                "inputs": dict(clusterer_inputs or {}),
            },
        }

        source_path = (
            local_source_path(source or loc.source_uri)
            if write_labels or write_meshes
            else None
        )
        if write_meshes:  # a collision must stop the run before labels are written
            check_mesh_slot(source_path, name, overwrite=overwrite)
        label_path = None
        if write_labels:
            label_path = write_label_image(
                source_path,
                name,
                loc.footprints,
                clusters,
                expected_level0_wh=tuple(loc.level_dimensions[0]),
                overwrite=overwrite,
                extra_metadata={"format": CLUSTERS_FORMAT, **identity},
            )

        glb_path = properties_path = mesh_path = None
        if gltf:
            glb_path = write_glb(
                f"{stem}.glb",
                loc.feature_ids,
                loc.footprints,
                glyph=gltf_glyph,
                cross_arm=cross_arm,
                extras={
                    "scale_um": loc.scale_um,
                    "embedding_sha256": loc.sha256,
                    "clustering_sha256": ident.model_sha256,
                },
            )
            # Glyph i has extras["label-value"] == i + 1; cluster first, so viewers
            # that default to the first property colour by it.
            properties_path = f"{stem}.properties.json"
            with open(properties_path, "w", encoding="utf-8") as fh:
                json.dump(
                    [
                        {"label-value": i + 1, "cluster": int(c), "feature_id": fid}
                        for i, (fid, c) in enumerate(
                            zip(loc.feature_ids, clusters, strict=True)
                        )
                    ],
                    fh,
                    indent=1,
                )
            if write_meshes:
                mesh_path = write_mesh_index(
                    source_path,
                    name,
                    glb_path,
                    properties_path,
                    features,
                    identity["embedding_model"],
                    overwrite=overwrite,
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
            "properties": (
                os.path.basename(properties_path) if properties_path else None
            ),
            "mesh_index": (
                {
                    "name": name,
                    "path": mesh_path,
                    "embeddings": os.path.join(
                        os.path.dirname(mesh_path), f"{name}.embeddings.zarr"
                    ),
                }
                if mesh_path
                else None
            ),
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
                "properties": properties_path,
                "labels": label_path,
                "meshes": mesh_path,
            }
        )
    return summaries
