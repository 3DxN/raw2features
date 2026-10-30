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
            labels_overwrite=labels_overwrite,
            gltf=gltf,
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
        for key in ("json", "gltf", "labels"):
            if s[key]:
                typer.echo(f"  {key}: {s[key]}")
