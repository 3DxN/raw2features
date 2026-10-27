"""``raw2features cluster`` - cluster stored embeddings with the trimodal model."""

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
    trimodal_model: str | None = typer.Option(
        None,
        "--trimodal-model",
        help="Trimodal model file; its sha256 is recorded. Default: a placeholder "
        "identity (inference is currently a placeholder too).",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the placeholder inference."),
    write_labels: bool = typer.Option(
        False,
        "--write-labels",
        help="Write the clusters as an OME-Zarr label image (one patch-footprint "
        "quad per location) into the source image under "
        "labels/clusters_<emb8>_<tri8>. Modifies the source.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Local source OME-Zarr for --write-labels (default: the store's "
        "recorded source URI).",
    ),
    labels_overwrite: bool = typer.Option(
        False, "--labels-overwrite", help="Replace an existing label of that name."
    ),
    gltf: bool = typer.Option(
        True, "--gltf/--no-gltf", help="Write a .glb scene (one quad per location)."
    ),
) -> None:
    """Assign each embedded location a cluster; write JSON, glTF and labels."""
    from raw2features.clustering import cluster_store

    try:
        summaries = cluster_store(
            store,
            models=list(model) or None,
            out_dir=out_dir,
            trimodal_model=trimodal_model,
            seed=seed,
            write_labels=write_labels,
            source=source,
            labels_overwrite=labels_overwrite,
            gltf=gltf,
        )
    except (FileExistsError, KeyError, ValueError) as exc:
        typer.echo(redact_uri_credentials(f"Error: {exc}"), err=True)
        raise typer.Exit(1) from exc
    for s in summaries:
        typer.echo(f"{s['model']}: {s['n']} locations")
        for key in ("json", "gltf", "labels"):
            if s[key]:
                typer.echo(f"  {key}: {s[key]}")
