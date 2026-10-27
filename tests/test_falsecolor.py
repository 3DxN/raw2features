"""Fluorescence -> pseudo-H&E transform and the ``PseudoHEView`` reader wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.core.falsecolor import (
    EOSIN_RGB,
    HEMATOXYLIN_RGB,
    PseudoHEParams,
    auto_background,
    preprocess,
    to_pseudo_he,
)
from raw2features.core.geometry import Point, Region, Size

PARAMS = PseudoHEParams(nuclear_channel="DAPI", cyto_channel="Eosin")


def test_preprocess_subtracts_background_then_gamma_and_normalises():
    out = preprocess(np.array([10.0, 110.0]), background=10.0, norm=255.0, gamma=0.5)
    np.testing.assert_allclose(out, [0.0, 10.0], rtol=1e-6)


def test_zero_signal_is_white():
    z = np.zeros((4, 5), dtype=np.uint16)
    rgb = to_pseudo_he(z, z, PARAMS, 0.0, 0.0)
    assert rgb.shape == (4, 5, 3) and rgb.dtype == np.uint8
    assert (rgb == 255).all()


def test_beer_lambert_matches_hand_computed_value():
    nuc, cyto = np.array([[1000.0]]), np.array([[500.0]])
    rgb = to_pseudo_he(nuc, cyto, PARAMS, 100.0, 50.0)
    n = (900.0**0.85) * 255.0 / 8500.0
    e = (450.0**0.85) * 255.0 / 2000.0
    want = [
        255.0 * np.exp(-(n * 0.08 * h + e * 0.012 * c))
        for h, c in zip(HEMATOXYLIN_RGB, EOSIN_RGB, strict=True)
    ]
    np.testing.assert_allclose(rgb[0, 0], np.rint(want), atol=1)


def test_nuclear_signal_is_purple_and_cyto_signal_is_pink():
    hi, lo = np.full((1, 1), 20000.0), np.zeros((1, 1))
    r, g, b = to_pseudo_he(hi, lo, PARAMS, 0.0, 0.0)[0, 0].astype(int)
    assert g < r and b > g  # hematoxylin: blue/purple, green most absorbed
    r, g, b = to_pseudo_he(lo, hi, PARAMS, 0.0, 0.0)[0, 0].astype(int)
    assert r > b > g  # eosin: pink


def test_auto_background_is_a_fifth_of_bright_foreground():
    img = np.concatenate([np.zeros(100), np.full(100, 1000.0)])
    assert auto_background(img, threshold=50) == pytest.approx(200.0)
    assert auto_background(np.zeros(10), threshold=50) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nuclear_norm": 0},
        {"gamma": -1},
        {"nuclear_bg": "median"},
        {"cyto_color": (1.0, 2.0)},
        {"cyto_channel": "DAPI"},
    ],
)
def test_params_reject_invalid_values(kwargs):
    base = {"nuclear_channel": "DAPI", "cyto_channel": "Eosin"}
    with pytest.raises(ValueError):
        PseudoHEParams(**{**base, **kwargs})


def test_resolve_channel_by_name_case_and_index():
    from raw2features.readers.pseudo_he import resolve_channel

    names = ["DAPI", "Eosin", "CD3"]
    assert resolve_channel("Eosin", names, 3) == 1
    assert resolve_channel("dapi", names, 3) == 0
    assert resolve_channel("2", names, 3) == 2
    assert resolve_channel("1", None, 2) == 1
    with pytest.raises(ValueError):
        resolve_channel("Hoechst", names, 3)
    with pytest.raises(ValueError):
        resolve_channel("5", names, 3)


@pytest.fixture
def fluo_stack(tmp_path) -> str:
    pytest.importorskip("zarr")
    from raw2features.data import write_sample_fluorescence_stack

    return write_sample_fluorescence_stack(
        str(tmp_path / "f.ome.zarr"), z=3, size=256, levels=2
    )


def test_pseudo_he_view_presents_rgb_brightfield(fluo_stack):
    from raw2features.readers.omezarr import OmeZarrReader
    from raw2features.readers.pseudo_he import PseudoHEView

    with OmeZarrReader(fluo_stack) as inner:
        view = PseudoHEView(inner, PARAMS)
        assert view.channel_names is None and not view.has_channel_axis
        assert view.name == inner.name
        assert view.mpp == inner.mpp and view.z_count == 3
        rgb = view.read_region(Region(0, Point(0, 0), Size(256, 256)))
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        assert (rgb[0, 0] >= 250).all()  # dark fluorescence background -> white
        assert rgb[128, 128].min() < 230  # tissue is stained
        prov = view.provenance()
        assert prov["nuclear_index"] == 0 and prov["cyto_index"] == 1
        assert "0" in prov["backgrounds"]


def test_pseudo_he_view_rejects_unknown_channel(synthetic_ngff):
    from raw2features.readers.omezarr import OmeZarrReader
    from raw2features.readers.pseudo_he import PseudoHEView

    # An RGB brightfield source carries no DAPI/Eosin channels to bind.
    with OmeZarrReader(synthetic_ngff) as inner, pytest.raises(ValueError):
        PseudoHEView(inner, PARAMS)
