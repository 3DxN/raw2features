"""``raw2features cluster`` - cluster stored embeddings with a clusterer plugin."""

from __future__ import annotations

import typer

from raw2features.core.uris import redact_uri_credentials


def cluster(
    store: str = typer.Argument(..., help="A *.embeddings.zarr store."),
    model: list[str] = typer.Option(
        [], "--model", "-m", help="Embedding model(s) to cluster (default: all)."
    ),
    out_dir: str | None = typer.Option(
        None, "--out-dir", help="Where to write JSON/glTF (default: next to the store)."
    ),
    clusterer: str = typer.Option(
        "trimodal",
        "--clusterer",
        help="Clusterer plugin (raw2features list clusterers).",
    ),
    clusterer_model: str | None = typer.Option(
        None,
        "--clusterer-model",
        "--trimodal-model",
        help="Model file of the clusterer; its sha256 is recorded. Default: the "
        "plugin's default (model.pt for trimodal).",
    ),
    clusterer_input: list[str] = typer.Option(
        [],
        "--clusterer-input",
        help="Extra input for the clusterer as NAME=PATH (repeatable), e.g. another "
        "modality.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the clusterer."),
    write_labels: bool = typer.Option(
        False,
        "--write-labels",
        help="Write the clusters as an OME-Zarr label image (one patch-footprint "
        "quad per location) into the source image under "
        "labels/clusters_<emb8>_<model8>_<grids8>. Modifies the source.",
    ),
    write_meshes: bool = typer.Option(
        False,
        "--write-meshes",
        help="Copy the glTF scene and its properties into the source image under "
        "meshes/ and list them in meshes/zarr.json, for viewers that discover meshes "
        "there. Modifies the source.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Local source OME-Zarr for --write-labels / --write-meshes (default: "
        "the store's recorded source URI).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "--labels-overwrite",
        help="Replace an existing label image / mesh entry of that name.",
    ),
    gltf: bool = typer.Option(
        True, "--gltf/--no-gltf", help="Write a .glb scene (one glyph per location)."
    ),
    gltf_glyph: str = typer.Option(
        "quad",
        "--gltf-glyph",
        help="glTF primitive per location: 'quad' (the patch footprint) or 'marker' "
        "(a small ellipsoid at the patch centre with a crosshair of thin prisms "
        "along the patch diagonals).",
    ),
    gltf_cross_arm: float | None = typer.Option(
        None,
        "--gltf-cross-arm",
        help="Crosshair arm of --gltf-glyph marker, as a fraction (0, 1] of the way "
        "from the patch centre to its corner; 1 draws the full diagonals. "
        "Default: 0.35.",
    ),
) -> None:
    """Assign each embedded location a cluster; write JSON, glTF and labels."""
    from raw2features.clustering import cluster_store

    inputs = {}
    for item in clusterer_input:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise typer.BadParameter(
                f"expected NAME=PATH, got {item!r}", param_hint="--clusterer-input"
            )
        inputs[key] = value
    try:
        summaries = cluster_store(
            store,
            models=list(model) or None,
            out_dir=out_dir,
            clusterer=clusterer,
            clusterer_model=clusterer_model,
            clusterer_inputs=inputs,
            seed=seed,
            write_labels=write_labels,
            source=source,
            overwrite=overwrite,
            gltf=gltf,
            write_meshes=write_meshes,
            gltf_glyph=gltf_glyph,
            gltf_cross_arm=gltf_cross_arm,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        ValueError,
    ) as exc:
        typer.echo(redact_uri_credentials(f"Error: {exc}"), err=True)
        raise typer.Exit(1) from exc
    for s in summaries:
        typer.echo(f"{s['model']}: {s['n']} locations")
        for key in ("json", "gltf", "properties", "labels", "meshes"):
            if s[key]:
                typer.echo(f"  {key}: {s[key]}")
