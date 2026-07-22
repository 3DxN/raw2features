"""``raw2features sample`` - write a tiny synthetic OME-Zarr slide to try offline."""

from __future__ import annotations

import typer


def sample(
    out_path: str = typer.Argument(
        "sample.ome.zarr", help="Where to write the synthetic OME-Zarr slide."
    ),
    size: int = typer.Option(1024, "--size", help="Level-0 side in pixels (square)."),
    mpp: float = typer.Option(0.5, "--mpp", help="Level-0 microns/pixel."),
    z: int = typer.Option(
        1, "--z", help="Number of z-planes (>1 writes a z-stack for 3D point sampling)."
    ),
    z_spacing: float = typer.Option(
        1.0, "--z-spacing", help="µm between z-planes (only meaningful with --z > 1)."
    ),
) -> None:
    """Write a small synthetic OME-NGFF slide so you can try the pipeline offline.

    No download, token, or GPU needed:

        raw2features sample sample.ome.zarr
        raw2features embed  sample.ome.zarr out/ -m resnet50 --device auto

    A z-stack (``--z 8``) plus a point cloud tries the 3D point-sampling path:

        raw2features sample sample.ome.zarr --z 8
        raw2features embed  sample.ome.zarr out/ -m resnet50 --points points.csv
    """
    from raw2features.data import write_sample_slide

    path = write_sample_slide(out_path, mpp0=mpp, size=size, z=z, z_spacing=z_spacing)
    typer.echo(f"wrote synthetic OME-Zarr slide: {path}")
    typer.echo(f"try:  raw2features embed {path} out/ -m resnet50 --device auto")
