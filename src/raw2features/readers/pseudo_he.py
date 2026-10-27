"""``PseudoHEView`` - serve a two-channel fluorescence source as pseudo-H&E RGB.

Wraps any multi-channel :class:`WSISource` and replaces its ``read_region`` with
the Beer-Lambert false colouring of :mod:`raw2features.core.falsecolor`. The view
presents itself as a plain RGB brightfield source (no channel axis, no channel
names), so segmentation, thumbnails, stain normalisation, the block-read fast path
and every RGB patch encoder consume virtual H&E unchanged. The transform is applied
per pixel to native-resolution reads, before any exact-MPP resample.
"""

from __future__ import annotations

import threading

import numpy as np

from raw2features.core.falsecolor import PseudoHEParams, auto_background, to_pseudo_he
from raw2features.core.geometry import Point, Region, Size

from .base import WSISource

# Longest side of the pyramid level the "auto" background is estimated from.
_BACKGROUND_LEVEL_MAX_SIDE = 2048


def resolve_channel(spec: str, names: list[str] | None, count: int) -> int:
    """Resolve a channel name or positional index against a source panel."""
    if names:
        if spec in names:
            return names.index(spec)
        folded = [n.casefold() for n in names]
        if spec.casefold() in folded:
            return folded.index(spec.casefold())
    if spec.strip().isdigit():
        index = int(spec)
        if index < count:
            return index
        raise ValueError(f"channel index {index} out of range ({count} channels)")
    raise ValueError(
        f"channel {spec!r} not found in source channels {names or '(unnamed)'}"
    )


class PseudoHEView(WSISource):
    """Read-through view rendering two fluorescent channels as pseudo-H&E."""

    name = "pseudo_he"

    def __init__(self, inner: WSISource, params: PseudoHEParams) -> None:
        super().__init__(inner.path)
        if not inner.has_channel_axis or inner.channel_count < 2:
            raise ValueError(
                "pseudo-H&E needs a multi-channel fluorescence source with at least "
                "two channels"
            )
        self.inner = inner
        self.params = params
        # Provenance names the underlying reader; the view is recorded separately
        # under source.pseudo_he.
        self.name = inner.name
        names = inner.channel_names
        count = inner.channel_count
        self.nuclear_index = resolve_channel(params.nuclear_channel, names, count)
        self.cyto_index = resolve_channel(params.cyto_channel, names, count)
        if self.nuclear_index == self.cyto_index:
            raise ValueError("pseudo-H&E nuclear and cytoplasm channels must differ")
        self._backgrounds: dict[int, tuple[float, float]] = {}
        self._lock = threading.Lock()

    # -- lifecycle / geometry: delegated --------------------------------------
    def open(self) -> PseudoHEView:
        return self

    def close(self) -> None:
        self.inner.close()

    @property
    def mpp(self) -> float | None:
        return self.inner.mpp

    @property
    def level_dimensions(self) -> list[Size]:
        return self.inner.level_dimensions

    def level_downsamples(self) -> list[float]:
        return self.inner.level_downsamples()

    def read_level_mapping(self, level: int) -> tuple[float, float, float, float]:
        return self.inner.read_level_mapping(level)

    @property
    def z_count(self) -> int:
        return self.inner.z_count

    @property
    def z_index(self) -> int:
        return self.inner.z_index

    @property
    def z_scale_um(self) -> float | None:
        return self.inner.z_scale_um

    @property
    def z_um(self) -> float | None:
        return self.inner.z_um

    def set_plane(self, z: int) -> None:
        self.inner.set_plane(z)

    def __getattr__(self, item: str):
        # Source self-description (axes, scale_um, translation, ...) passes through.
        if item.startswith("__") or item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    # -- pixels -----------------------------------------------------------------
    def backgrounds(self) -> tuple[float, float]:
        """``(nuclear_bg, cyto_bg)`` for the current z plane, estimating "auto"."""
        z = self.inner.z_index
        with self._lock:
            cached = self._backgrounds.get(z)
            if cached is not None:
                return cached
            p = self.params
            needs_auto = p.nuclear_bg == "auto" or p.cyto_bg == "auto"
            planes = self._background_planes() if needs_auto else None
            nuc = (
                auto_background(planes[0], p.bg_threshold)
                if p.nuclear_bg == "auto"
                else float(p.nuclear_bg)
            )
            cyto = (
                auto_background(planes[1], p.bg_threshold)
                if p.cyto_bg == "auto"
                else float(p.cyto_bg)
            )
            self._backgrounds[z] = (nuc, cyto)
            return nuc, cyto

    def _background_planes(self) -> tuple[np.ndarray, np.ndarray]:
        dims = self.inner.level_dimensions
        level = len(dims) - 1
        for i, d in enumerate(dims):
            if max(d.width, d.height) <= _BACKGROUND_LEVEL_MAX_SIDE:
                level = i
                break
        d = dims[level]
        block = self.inner.read_region_channels(
            Region(level, Point(0, 0), Size(d.width, d.height))
        )
        return block[..., self.nuclear_index], block[..., self.cyto_index]

    def read_region(self, region: Region) -> np.ndarray:
        block = self.inner.read_region_channels(region)
        nuc_bg, cyto_bg = self.backgrounds()
        return to_pseudo_he(
            block[..., self.nuclear_index],
            block[..., self.cyto_index],
            self.params,
            nuc_bg,
            cyto_bg,
        )

    def provenance(self) -> dict:
        """Header record: parameters, resolved channels and per-z backgrounds."""
        names = self.inner.channel_names
        return {
            "method": "beer_lambert_falsecolor",
            "params": self.params.to_dict(),
            "nuclear_index": self.nuclear_index,
            "cyto_index": self.cyto_index,
            "nuclear_name": names[self.nuclear_index] if names else None,
            "cyto_name": names[self.cyto_index] if names else None,
            "backgrounds": {
                str(z): list(bg) for z, bg in sorted(self._backgrounds.items())
            },
        }
