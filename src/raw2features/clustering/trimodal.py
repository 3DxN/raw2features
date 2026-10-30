"""Built-in clusterers: ``trimodal`` (the default) and ``placeholder``.

The trimodal model is not integrated yet: :class:`TrimodalClusterer` already has its
final shape - model file (``model.pt`` by default), sha256 identity, lazy dependency
loading - but its :meth:`~TrimodalClusterer.predict` still draws placeholder values so
the downstream outputs (JSON, OME-Zarr labels, glTF) can be built and validated now.

The class can move into its own package unchanged; it then registers itself under the
``raw2features.clusterers`` entry-point group (see ``clustering.base``) and the entry
point in raw2features' ``pyproject.toml`` is dropped.
"""

from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np

from raw2features.core.plugins import register

from .base import ClusterContext, Clusterer, ClustererIdentity

# Cluster values the placeholder draws from.
PLACEHOLDER_CLUSTERS: tuple[int, int] = (7, 23)

# Stand-in identity used while no trimodal model file is available. Remove together
# with the placeholder inference.
PLACEHOLDER_TRIMODAL_SHA256 = hashlib.sha256(
    b"raw2features trimodal clustering placeholder v0"
).hexdigest()

DEFAULT_MODEL_FILE = "model.pt"


def _version() -> str:
    from raw2features import __version__

    return __version__


def placeholder_clusters(n: int, seed: int) -> np.ndarray:
    """PLACEHOLDER inference: ``n`` values drawn uniformly from 7 and 23."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(PLACEHOLDER_CLUSTERS, dtype=np.int64), size=n)


@register("clusterers", "placeholder")
class PlaceholderClusterer(Clusterer):
    """Random 7/23 per location, no model - for tests and pipeline dry runs."""

    name = "placeholder"

    def identity(self) -> ClustererIdentity:
        return ClustererIdentity(
            plugin=self.name,
            version=_version(),
            model_sha256=PLACEHOLDER_TRIMODAL_SHA256,
            model_path=None,
            placeholder=True,
        )

    def predict(self, features: np.ndarray, context: ClusterContext) -> np.ndarray:
        return placeholder_clusters(int(np.asarray(features).shape[0]), context.seed)


@register("clusterers", "trimodal")
class TrimodalClusterer(Clusterer):
    """Trimodal-model clustering of patch embeddings.

    ``model_file`` defaults to ``model.pt`` (resolved against the working directory).
    An explicitly given file must exist. While inference is a placeholder, a missing
    *default* file falls back to :data:`PLACEHOLDER_TRIMODAL_SHA256` with a warning;
    once real inference lands, a missing model file must be an error.
    """

    name = "trimodal"
    default_model_file = DEFAULT_MODEL_FILE

    def __init__(self, model_file: str | None = None) -> None:
        super().__init__(model_file)
        self._explicit_model = model_file is not None
        self._sha256: str | None = None
        self._model = None

    def _model_path(self) -> str | None:
        path = os.path.abspath(self.model_file) if self.model_file else None
        if path and os.path.isfile(path):
            return path
        if self._explicit_model:
            raise FileNotFoundError(f"trimodal model file not found: {self.model_file}")
        return None

    def load(self) -> None:
        # ------------------------------------------------------------------------
        # DEPENDENCIES: import the spatial-transcriptomics packages the trimodal
        # model needs HERE, inside load(), never at module level - raw2features must
        # import (and `raw2features list clusterers` must work) without them. Add
        # them to the `trimodal` extra in pyproject.toml, and turn a missing package
        # into an actionable error, e.g.:
        #
        #     try:
        #         import <st_package>  # noqa: F401
        #     except ImportError as exc:
        #         raise ImportError(
        #             "the trimodal clusterer needs <st_package>; "
        #             'pip install "raw2features[trimodal]"'
        #         ) from exc
        # ------------------------------------------------------------------------
        path = self._model_path()
        if path is None:
            warnings.warn(
                f"trimodal model file {self.model_file!r} not found; recording the "
                "placeholder model identity",
                stacklevel=2,
            )
            return
        from raw2features.embedders._hub import sha256_file

        self._sha256 = sha256_file(path)
        # MODEL LOADING: deserialise `path` here into self._model (with
        # torch.load(path, map_location="cpu", weights_only=True) if it is a
        # state dict - never unpickle arbitrary objects), after hashing it above.

    def identity(self) -> ClustererIdentity:
        path = self._model_path()
        if path is not None and self._sha256 is None:
            from raw2features.embedders._hub import sha256_file

            self._sha256 = sha256_file(path)
        return ClustererIdentity(
            plugin=self.name,
            version=_version(),
            model_sha256=self._sha256 or PLACEHOLDER_TRIMODAL_SHA256,
            model_path=path,
            # True until predict runs the real model.
            placeholder=True,
        )

    def predict(self, features: np.ndarray, context: ClusterContext) -> np.ndarray:
        # PLACEHOLDER: replace with inference of self._model on `features` (and
        # whatever else of `context` the model needs - footprints, other models'
        # features via context.other_features(), extra modalities via
        # context.inputs). Then set placeholder=False in identity().
        return placeholder_clusters(int(np.asarray(features).shape[0]), context.seed)
