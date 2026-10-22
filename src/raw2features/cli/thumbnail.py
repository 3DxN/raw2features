"""``raw2features thumbnail`` - write a slide thumbnail (+ optional QC overlay).

Standalone so thumbnails can be batched separately, before or after the heavy
embed run. ``--overlay`` reruns the (cheap) segmenter + patcher to show the
tissue mask and the patches that tiling would keep for the given grid params.
"""

from __future__ import annotations

import typer

from raw2features.core import plugins
from raw2features.viz import DEFAULT_THUMBNAIL_MPP, write_thumbnails

from . import _extended as ext


def thumbnail(
    slide: str = typer.Argument(..., help="Path to an OME-Zarr store."),
    out_dir: str = typer.Argument(..., help="Output directory for the thumbnail."),
    reader: str = typer.Option("omezarr", "--reader"),
    thumbnail_mpp: float = typer.Option(
        DEFAULT_THUMBNAIL_MPP,
        "--thumbnail-mpp",
        help="Thumbnail MPP (default = seg MPP).",
    ),
    max_px: int | None = typer.Option(
        None, "--max-px", help="Cap longest side (overrides --thumbnail-mpp)."
    ),
    overlay: bool = typer.Option(
        False, "--overlay", help="Also tint the tissue mask + outline kept patches."
    ),
    segmenter: str = typer.Option("otsu", "--segmenter", help="Overlay segmenter."),
    mpp: float = typer.Option(1.0, "--mpp", help="Target MPP for the grid (overlay)."),
    patch_size: int = typer.Option(224, "--patch-size", help="Patch px (overlay)."),
    step: int | None = typer.Option(None, "--step", help="Stride (overlay)."),
    tissue_threshold: float = typer.Option(0.1, "--tissue-threshold"),
    z_planes: str | None = ext.z_planes_option(),
    patch_um: float | None = ext.patch_um_option(),
    grid_offset_x: float = ext.grid_offset_x_option(),
    grid_offset_y: float = ext.grid_offset_y_option(),
    pseudo_he: bool = ext.pseudo_he_option(),
    he_nuclear_channel: str | None = ext.he_nuclear_channel_option(),
    he_cyto_channel: str | None = ext.he_cyto_channel_option(),
    he_nuclear_norm: float = ext.he_nuclear_norm_option(),
    he_cyto_norm: float = ext.he_cyto_norm_option(),
    he_nuclear_k: float = ext.he_nuclear_k_option(),
    he_cyto_k: float = ext.he_cyto_k_option(),
    he_gamma: float = ext.he_gamma_option(),
    he_nuclear_bg: str = ext.he_nuclear_bg_option(),
    he_cyto_bg: str = ext.he_cyto_bg_option(),
    he_bg_threshold: float = ext.he_bg_threshold_option(),
    he_nuclear_color: str = ext.he_nuclear_color_option(),
    he_cyto_color: str = ext.he_cyto_color_option(),
) -> None:
    """Render a thumbnail; with --overlay, show the tissue mask + kept patches.

    With --z-planes, one thumbnail (and overlay) is written per plane.
    """
    from raw2features.pipeline.runner import parse_z_planes, slide_id_from_path

    if patch_um is not None:
        mpp = patch_um / patch_size
    offset_x, offset_y = ext.validate_grid_offset(grid_offset_x, grid_offset_y)
    z_planes = ext.validate_z_planes(z_planes)
    params = ext.build_pseudo_he(
        pseudo_he,
        he_nuclear_channel,
        he_cyto_channel,
        he_nuclear_norm,
        he_cyto_norm,
        he_nuclear_k,
        he_cyto_k,
        he_gamma,
        he_nuclear_bg,
        he_cyto_bg,
        he_bg_threshold,
        he_nuclear_color,
        he_cyto_color,
    )

    reader_cls = plugins.get("readers", reader)
    slide_id = slide_id_from_path(slide)

    with reader_cls(slide) as base:
        rdr = base
        if params is not None:
            from raw2features.readers.pseudo_he import PseudoHEView

            rdr = PseudoHEView(base, params)
        planes = (
            [None] if z_planes is None else parse_z_planes(z_planes, base.z_count)
        )
        for z in planes:
            if z is not None:
                base.set_plane(z)
            tissue = None
            coords = None
            level0_patch = None
            if overlay:
                from raw2features.patcher.grid import GridPatcher

                seg = plugins.get("segmenters", segmenter)()
                tissue = seg.segment(rdr)
                patcher = GridPatcher(
                    target_mpp=mpp,
                    patch_px=patch_size,
                    step_out_px=step,
                    offset_x_frac=offset_x,
                    offset_y_frac=offset_y,
                )
                grid = patcher.build_grid(rdr)
                coords, _grid_index, _grid_tissue = patcher.tile(
                    grid, tissue, tissue_threshold
                )
                level0_patch = grid.level0_patch
            meta = write_thumbnails(
                rdr,
                out_dir,
                slide_id,
                mpp=thumbnail_mpp,
                max_px=max_px,
                tissue=tissue,
                coords=coords,
                level0_patch=level0_patch,
                overlay=overlay,
                plain_name=None if z is None else f"{slide_id}.z{z}.thumbnail.png",
                overlay_name=(
                    None if z is None else f"{slide_id}.z{z}.thumbnail.overlay.png"
                ),
            )
            if z is not None:
                typer.echo(f"z: {z}")
            for key, value in meta.items():
                typer.echo(f"{key}: {value}")
