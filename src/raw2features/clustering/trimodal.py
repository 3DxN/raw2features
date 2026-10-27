"""Trimodal clustering of patch embeddings: one integer cluster per embedded location.

The trimodal model is not integrated yet. :func:`predict_clusters` is a deterministic
placeholder so the downstream outputs (JSON, OME-Zarr labels, glTF) can be built and
validated now; its signature is the seam the real inference will plug into.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Cluster values the placeholder draws from.
PLACEHOLDER_CLUSTERS: tuple[int, int] = (7, 23)

# Stand-in identity for the not-yet-available trimodal model, used when no
# ``--trimodal-model`` file is given. Replace together with predict_clusters.
PLACEHOLDER_TRIMODAL_SHA256 = hashlib.sha256(
    b"raw2features trimodal clustering placeholder v0"
).hexdigest()


def resolve_trimodal_sha256(path: str | None) -> tuple[str, bool]:
    """``(sha256, is_placeholder)`` for the trimodal model file, or the placeholder."""
    if path is None:
        return PLACEHOLDER_TRIMODAL_SHA256, True
    from raw2features.embedders._hub import sha256_file

    return sha256_file(path), False


def predict_clusters(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign one integer cluster to each row of ``features`` (``(N, D)``).

    PLACEHOLDER: draws uniformly from ``PLACEHOLDER_CLUSTERS`` (7 or 23). This will be
    replaced by a functioning trimodal-model inference mechanism; callers must not rely
    on the values beyond their being integers, one per feature row.
    """
    n = int(np.asarray(features).shape[0])
    return rng.choice(np.asarray(PLACEHOLDER_CLUSTERS, dtype=np.int32), size=n)
