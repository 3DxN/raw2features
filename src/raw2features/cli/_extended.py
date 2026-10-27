"""Shared CLI options: z-stacks, physical patch size + grid offset, pseudo-H&E, and
local model weights. Declared once here so ``embed``, ``embed-many``, ``verify`` and
``thumbnail`` expose (and hash) them identically."""

from __future__ import annotations

import math

import typer

from raw2features.core.falsecolor import EOSIN_RGB, HEMATOXYLIN_RGB, PseudoHEParams

DEFAULT_PHYSICAL_PATCH_PX = 224


# -- option factories (Typer reads the defaults as option declarations) ----------
def z_planes_option():
    return typer.Option(
        None,
        "--z-planes",
        help="Process a z-stack plane by plane: 'all', a list '0,3,5', or a slice "
        "'start:stop[:step]'. Each plane is segmented, tiled and embedded as its own "
        "grid (grids/<key>_z<k>). Default: 2D (plane 0 only).",
    )


def patch_um_option():
    return typer.Option(
        None,
        "--patch-um",
        help="Physical patch side in µm. Each patch is read at exactly this size and "
        "resampled to --patch-size px (default 224), i.e. --mpp = patch_um / "
        "patch_size. Mutually exclusive with --mpp and --config.",
    )


def grid_offset_x_option():
    return typer.Option(
        0.0,
        "--grid-offset-x",
        help="Grid origin x offset as a fraction [0, 1) of the patch side.",
    )


def grid_offset_y_option():
    return typer.Option(
        0.0,
        "--grid-offset-y",
        help="Grid origin y offset as a fraction [0, 1) of the patch side.",
    )


def pseudo_he_option():
    return typer.Option(
        False,
        "--pseudo-he/--no-pseudo-he",
        help="Render two fluorescent channels as pseudo-H&E (Beer-Lambert false "
        "colouring) and run the whole brightfield pipeline - segmentation, "
        "thumbnails, RGB encoders - on that virtual H&E.",
    )


def he_nuclear_channel_option():
    return typer.Option(
        None,
        "--he-nuclear-channel",
        help="Nuclear (hematoxylin-like) channel: name (e.g. DAPI) or index.",
    )


def he_cyto_channel_option():
    return typer.Option(
        None,
        "--he-cyto-channel",
        help="Cytoplasm/stroma (eosin-like) channel: name or index.",
    )


def he_nuclear_norm_option():
    return typer.Option(
        8500.0, "--he-nuclear-norm", help="Nuclear normalisation factor."
    )


def he_cyto_norm_option():
    return typer.Option(
        2000.0, "--he-cyto-norm", help="Cytoplasm normalisation factor."
    )


def he_nuclear_k_option():
    return typer.Option(0.08, "--he-nuclear-k", help="Nuclear absorption constant k.")


def he_cyto_k_option():
    return typer.Option(0.012, "--he-cyto-k", help="Cytoplasm absorption constant k.")


def he_gamma_option():
    return typer.Option(
        0.85, "--he-gamma", help="Gamma applied after background removal."
    )


def he_nuclear_bg_option():
    return typer.Option(
        "auto",
        "--he-nuclear-bg",
        help="Nuclear background (source intensity units) or 'auto' (per z plane: "
        "95th percentile of pixels above --he-bg-threshold, divided by 5).",
    )


def he_cyto_bg_option():
    return typer.Option(
        "auto", "--he-cyto-bg", help="Cytoplasm background or 'auto' (as nuclear)."
    )


def he_bg_threshold_option():
    return typer.Option(
        50.0, "--he-bg-threshold", help="Foreground threshold for 'auto' background."
    )


def he_nuclear_color_option():
    return typer.Option(
        ",".join(f"{v:g}" for v in HEMATOXYLIN_RGB),
        "--he-nuclear-color",
        help="Nuclear dye RGB absorption vector 'r,g,b'.",
    )


def he_cyto_color_option():
    return typer.Option(
        ",".join(f"{v:g}" for v in EOSIN_RGB),
        "--he-cyto-color",
        help="Cytoplasm dye RGB absorption vector 'r,g,b'.",
    )


def weights_option():
    return typer.Option(
        [],
        "--weights",
        help="Load a model's weights from a local file or directory instead of the "
        "Hugging Face Hub: MODEL=PATH (repeatable), e.g. uni2_h=/models/UNI2-h. The "
        "file must match the model's pinned sha256.",
    )


# -- parsing / validation ---------------------------------------------------------
def _parse_rgb(text: str, hint: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError:
        values = ()
    if len(values) != 3 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise typer.BadParameter(
            "must be three non-negative numbers 'r,g,b'", param_hint=hint
        )
    return values  # type: ignore[return-value]


def _parse_background(text: str, hint: str) -> float | str:
    if text.strip().lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError:
        raise typer.BadParameter(
            "must be a number or 'auto'", param_hint=hint
        ) from None
    if not math.isfinite(value) or value < 0:
        raise typer.BadParameter("must be non-negative", param_hint=hint)
    return value


def build_pseudo_he(
    enabled: bool,
    nuclear_channel: str | None,
    cyto_channel: str | None,
    nuclear_norm: float,
    cyto_norm: float,
    nuclear_k: float,
    cyto_k: float,
    gamma: float,
    nuclear_bg: str,
    cyto_bg: str,
    bg_threshold: float,
    nuclear_color: str,
    cyto_color: str,
) -> PseudoHEParams | None:
    """The validated pseudo-H&E parameters, or ``None`` when ``--pseudo-he`` is off."""
    if not enabled:
        if nuclear_channel is not None or cyto_channel is not None:
            raise typer.BadParameter(
                "channel selection requires --pseudo-he", param_hint="--he-*-channel"
            )
        return None
    if nuclear_channel is None or cyto_channel is None:
        raise typer.BadParameter(
            "--pseudo-he needs both --he-nuclear-channel and --he-cyto-channel",
            param_hint="--pseudo-he",
        )
    try:
        return PseudoHEParams(
            nuclear_channel=nuclear_channel,
            cyto_channel=cyto_channel,
            nuclear_norm=nuclear_norm,
            cyto_norm=cyto_norm,
            nuclear_k=nuclear_k,
            cyto_k=cyto_k,
            gamma=gamma,
            nuclear_bg=_parse_background(nuclear_bg, "--he-nuclear-bg"),
            cyto_bg=_parse_background(cyto_bg, "--he-cyto-bg"),
            bg_threshold=bg_threshold,
            nuclear_color=_parse_rgb(nuclear_color, "--he-nuclear-color"),
            cyto_color=_parse_rgb(cyto_color, "--he-cyto-color"),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--pseudo-he") from exc


def resolve_physical_patch(
    patch_um: float | None,
    mpp: float | None,
    patch_size: int | None,
    *,
    config: str | None = None,
    snap_to_level: bool = False,
) -> tuple[float | None, int | None]:
    """Map ``--patch-um`` onto the ``(mpp, patch_size)`` override pair.

    Returns the inputs unchanged when ``--patch-um`` is not given.
    """
    if patch_um is None:
        return mpp, patch_size
    if not math.isfinite(patch_um) or patch_um <= 0:
        raise typer.BadParameter(
            "must be finite and greater than zero", param_hint="--patch-um"
        )
    if mpp is not None:
        raise typer.BadParameter(
            "cannot be combined with --mpp", param_hint="--patch-um"
        )
    if config is not None:
        raise typer.BadParameter(
            "cannot be combined with --config", param_hint="--patch-um"
        )
    if snap_to_level:
        raise typer.BadParameter(
            "cannot be combined with --snap-to-level (it would change the physical "
            "patch size)",
            param_hint="--patch-um",
        )
    px = patch_size if patch_size is not None else DEFAULT_PHYSICAL_PATCH_PX
    return patch_um / px, px


def validate_grid_offset(x: float, y: float) -> tuple[float, float]:
    for value, hint in ((x, "--grid-offset-x"), (y, "--grid-offset-y")):
        if not (math.isfinite(value) and 0.0 <= value < 1.0):
            raise typer.BadParameter("must be a fraction in [0, 1)", param_hint=hint)
    return float(x), float(y)


def validate_z_planes(spec: str | None) -> str | None:
    if spec is None:
        return None
    from raw2features.pipeline.runner import parse_z_planes

    try:
        parse_z_planes(spec, None)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--z-planes") from exc
    return spec


def parse_local_weights(entries: list[str], models: list[str]) -> dict[str, str]:
    """Parse repeatable ``MODEL=PATH`` into a mapping; each model must support it."""
    from raw2features.embedders.model_registry import get_spec

    out: dict[str, str] = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            raise typer.BadParameter("expected MODEL=PATH", param_hint="--weights")
        if name not in models:
            raise typer.BadParameter(
                f"model {name!r} is not among the requested models {models}",
                param_hint="--weights",
            )
        if name in out:
            raise typer.BadParameter(f"{name!r} given twice", param_hint="--weights")
        spec = get_spec(name)
        bare_checkpoint = spec.checkpoint and not spec.source.startswith("hf-hub:")
        if spec.family != "timm" or not (spec.local_arch or bare_checkpoint):
            raise typer.BadParameter(
                f"model {name!r} does not support local weights",
                param_hint="--weights",
            )
        out[name] = path
    return out
