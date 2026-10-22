"""Offline model loading from a local weights file (``--weights MODEL=PATH``)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

pytest.importorskip("torch")
import torch
import typer

from raw2features.embedders.fingerprint import patch_output_fingerprint
from raw2features.embedders.model_registry import build_embedder, get_spec
from raw2features.embedders.timm_embedder import TimmEmbedder


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_timm(monkeypatch):
    events: list[tuple] = []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

    timm = pytest.importorskip("timm")

    def create_model(name, **kwargs):
        events.append(("create", name, kwargs.get("pretrained")))
        return Model()

    monkeypatch.setattr(timm, "create_model", create_model)

    def no_hub(*args, **kwargs):
        raise AssertionError("the Hugging Face Hub must not be contacted")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    return events


def test_uni2_loads_local_file_without_hub(tmp_path, fake_timm, monkeypatch):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"stand-in uni2 weights")
    embedder = build_embedder("uni2_h")
    embedder.spec = replace(embedder.spec, weights_sha256=_digest(weights))
    loaded: list[str] = []
    monkeypatch.setattr(
        TimmEmbedder,
        "_load_state_file",
        lambda self, model, path, ckpt: loaded.append(path),
    )

    embedder.local_weights = str(tmp_path)  # a directory holding weights_filename
    embedder.load(device="cpu", dtype=torch.float32)

    assert fake_timm == [("create", "vit_giant_patch14_224", False)]
    assert loaded == [str(weights)]


def test_local_file_with_wrong_hash_is_refused_before_deserialising(
    tmp_path, fake_timm, monkeypatch
):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"not the pinned uni2 weights")

    def no_load(*args, **kwargs):
        raise AssertionError("torch.load must not run on unverified bytes")

    monkeypatch.setattr(torch, "load", no_load)
    embedder = build_embedder("uni2_h")
    embedder.local_weights = str(weights)
    with pytest.raises(ValueError, match="sha256"):
        embedder.load(device="cpu", dtype=torch.float32)


def test_missing_local_file_is_reported(tmp_path, fake_timm):
    embedder = build_embedder("uni2_h")
    embedder.local_weights = str(tmp_path / "absent.bin")
    with pytest.raises(FileNotFoundError):
        embedder.load(device="cpu", dtype=torch.float32)


def test_local_route_keeps_the_output_fingerprint():
    # The fingerprint is a function of the registry spec alone; the local route only
    # changes where the identical pinned bytes come from.
    spec = get_spec("uni2_h")
    assert spec.local_arch == "vit_giant_patch14_224"
    assert patch_output_fingerprint(spec, "fp32") == patch_output_fingerprint(
        replace(spec, local_arch=None), "fp32"
    )


def test_cli_weights_parsing():
    from raw2features.cli._extended import parse_local_weights

    assert parse_local_weights(["uni2_h=/m/uni2"], ["uni2_h"]) == {
        "uni2_h": "/m/uni2"
    }
    for entries, models in (
        (["uni2_h"], ["uni2_h"]),  # no PATH
        (["uni2_h=/m"], ["resnet50"]),  # model not requested
        (["resnet50=/m"], ["resnet50"]),  # no local loading support
    ):
        with pytest.raises(typer.BadParameter):
            parse_local_weights(entries, models)
