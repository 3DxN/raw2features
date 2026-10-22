"""Fluorescence -> pseudo-H&E false colouring (Beer-Lambert virtual staining).

Two fluorescent channels - a nuclear stain (e.g. DAPI/Hoechst) and a
cytoplasm/stroma stain (e.g. eosin fluorescence) - are mapped to an RGB image that
resembles a hematoxylin & eosin brightfield slide. Each channel is treated as the
concentration of an absorbing dye: after background subtraction and a gamma
compression it is scaled by a per-channel normalisation, weighted by the dye's RGB
absorption vector and a channel constant ``k``, and transmitted light is
``255 * exp(-(optical density))``, per RGB channel.

The method is that of Giacomelli et al., "Virtual Hematoxylin and Eosin
Transillumination Microscopy Using Epi-Fluorescence Imaging", PLOS ONE (2016),
doi:10.1371/journal.pone.0159337, as parameterised by Serafin et al., "FalseColor-
Python: A rapid intensity-leveling and digital-staining package for fluorescence-
based slide-free digital pathology", PLOS ONE (2020),
doi:10.1371/journal.pone.0233198 (reference implementation:
https://github.com/serrob23/falsecolor). This module is an independent
re-implementation of the published equations; the default constants are the
published H&E settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Published H&E absorption vectors (R, G, B) for the nuclear and cytoplasm dyes.
HEMATOXYLIN_RGB: tuple[float, float, float] = (0.17, 0.27, 0.105)
EOSIN_RGB: tuple[float, float, float] = (0.05, 1.0, 0.54)


@dataclass(frozen=True)
class PseudoHEParams:
    """Configuration of the two-channel fluorescence -> pseudo-H&E transform.

    ``nuclear_channel`` / ``cyto_channel`` name a source channel (matched against
    the source's channel names) or give its positional index. ``*_bg`` is the
    background level subtracted before the transform, in source intensity units, or
    ``"auto"`` to estimate it per z plane (see :func:`auto_background`).
    """

    nuclear_channel: str
    cyto_channel: str
    nuclear_norm: float = 8500.0
    cyto_norm: float = 2000.0
    nuclear_k: float = 0.08
    cyto_k: float = 0.012
    gamma: float = 0.85
    nuclear_bg: float | str = "auto"
    cyto_bg: float | str = "auto"
    bg_threshold: float = 50.0
    nuclear_color: tuple[float, float, float] = HEMATOXYLIN_RGB
    cyto_color: tuple[float, float, float] = EOSIN_RGB

    def __post_init__(self) -> None:
        for name in ("nuclear_norm", "cyto_norm", "gamma"):
            if not getattr(self, name) > 0:
                raise ValueError(f"pseudo-H&E {name} must be positive")
        for name in ("nuclear_k", "cyto_k", "bg_threshold"):
            if getattr(self, name) < 0:
                raise ValueError(f"pseudo-H&E {name} must be non-negative")
        for name in ("nuclear_bg", "cyto_bg"):
            value = getattr(self, name)
            if isinstance(value, str) and value != "auto":
                raise ValueError(f"pseudo-H&E {name} must be a number or 'auto'")
            if not isinstance(value, str) and value < 0:
                raise ValueError(f"pseudo-H&E {name} must be non-negative")
        for name in ("nuclear_color", "cyto_color"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"pseudo-H&E {name} must be an (r, g, b) triple")
        if self.nuclear_channel == self.cyto_channel:
            raise ValueError("pseudo-H&E nuclear and cytoplasm channels must differ")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["nuclear_color"] = list(self.nuclear_color)
        out["cyto_color"] = list(self.cyto_color)
        return out


def auto_background(image: np.ndarray, threshold: float) -> float:
    """Estimate a channel's background as 1/5 of its bright-foreground level.

    The foreground level is the 95th percentile of pixels above ``threshold``; an
    image with no such pixels has background 0.
    """
    fg = np.asarray(image)[np.asarray(image) > threshold]
    if fg.size == 0:
        return 0.0
    return float(np.percentile(fg, 95)) / 5.0


def preprocess(
    image: np.ndarray, background: float, norm: float, gamma: float
) -> np.ndarray:
    """Background-subtract, gamma-compress, and normalise one channel (float32)."""
    x = np.clip(np.asarray(image, dtype=np.float32) - np.float32(background), 0, None)
    return np.power(x, np.float32(gamma)) * np.float32(255.0 / norm)


def to_pseudo_he(
    nuclear: np.ndarray,
    cyto: np.ndarray,
    params: PseudoHEParams,
    nuclear_bg: float,
    cyto_bg: float,
) -> np.ndarray:
    """Map ``(h, w)`` nuclear and cytoplasm intensities to ``(h, w, 3)`` uint8 RGB."""
    n = preprocess(nuclear, nuclear_bg, params.nuclear_norm, params.gamma)
    e = preprocess(cyto, cyto_bg, params.cyto_norm, params.gamma)
    beta_n = np.asarray(params.nuclear_color, dtype=np.float32) * params.nuclear_k
    beta_e = np.asarray(params.cyto_color, dtype=np.float32) * params.cyto_k
    od = n[..., None] * beta_n + e[..., None] * beta_e
    rgb = np.float32(255.0) * np.exp(-od)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
