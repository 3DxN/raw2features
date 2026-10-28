"""``Clusterer`` - the seam a clustering inference engine plugs into.

A clusterer assigns one positive integer cluster to every embedded location of one
embedding model in a store. raw2features owns everything around that call: reading
the store, the location identities (feature IDs), patch footprints, provenance and
all outputs (JSON, OME-Zarr label image, glTF). A plugin only implements
:meth:`Clusterer.predict` and reports its identity.

Plugins are discovered through the ``raw2features.clusterers`` entry-point group, so
an engine can live in its own package::

    [project.entry-points."raw2features.clusterers"]
    trimodal = "my_package.plugin:TrimodalClusterer"

and is selected with ``raw2features cluster --clusterer trimodal``. The contract:

* ``__init__(model_file: str | None = None)`` - ``None`` means the class's
  ``default_model_file``. Construction must be cheap; heavy imports and model loading
  belong in :meth:`load`.
* :meth:`identity` - the model's sha256 (recorded in the JSON and in the label image
  name), the plugin version and whether the result is a placeholder.
* :meth:`predict` - called **once per slide and embedding model** with every location's
  features (all z planes), so per-slide fitting/normalisation is possible. Returns an
  ``(N,)`` integer array, values ``>= 1``: ``0`` is the background of the label image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .footprint import Footprints


@dataclass(frozen=True)
class ClustererIdentity:
    """What the outputs record about the clustering model."""

    plugin: str  # registered clusterer name
    version: str  # plugin package version (provenance)
    model_sha256: str
    model_path: str | None  # absolute path of the model file, when one was used
    placeholder: bool = False  # True while the result is not real inference


@dataclass(frozen=True)
class ClusterContext:
    """Everything a clusterer may use beyond the feature matrix, for one slide/model.

    Row ``i`` of the features passed to :meth:`Clusterer.predict` is ``feature_ids[i]``
    with level-0 voxel footprint ``footprints.origin_zyx[i]`` / ``size_zyx[i]``.
    """

    slide_id: str
    store: str  # path of the *.embeddings.zarr store
    source_uri: str  # the image the embeddings were computed from
    embedding_model: str
    embedding_sha256: str
    feature_ids: list[str]
    footprints: Footprints
    scale_um: Mapping[str, float | None]  # level-0 µm per voxel (x, y, z)
    seed: int = 0
    # Extra inputs by name (e.g. other modalities): ``--clusterer-input NAME=PATH``.
    inputs: Mapping[str, str] = field(default_factory=dict)

    def other_features(self, model: str) -> np.ndarray:
        """Another embedding model's features for the same rows (same store/grids)."""
        from .collect import collect_locations

        other = collect_locations(self.store, model)
        if other.feature_ids != self.feature_ids:
            raise ValueError(
                f"model {model!r} was not embedded at the same locations as "
                f"{self.embedding_model!r}"
            )
        return np.concatenate([np.asarray(f) for _key, f in other.features])


class Clusterer(ABC):
    """Abstract clustering engine (see the module docstring for the contract)."""

    name: str = "clusterer"
    default_model_file: str | None = None

    def __init__(self, model_file: str | None = None) -> None:
        self.model_file = (
            model_file if model_file is not None else self.default_model_file
        )

    def load(self) -> None:  # noqa: B027 - optional hook
        """Import heavy dependencies and load the model; called once before predict."""

    @abstractmethod
    def identity(self) -> ClustererIdentity:
        """The model identity recorded with the results."""

    @abstractmethod
    def predict(self, features: np.ndarray, context: ClusterContext) -> np.ndarray:
        """``(N,)`` positive integer clusters for ``features`` ``(N, D)``."""


def check_clusters(clusters, n: int, *, plugin: str) -> np.ndarray:
    """Validate a :meth:`Clusterer.predict` result; return it as int64."""
    arr = np.asarray(clusters)
    if arr.shape != (n,):
        raise ValueError(
            f"clusterer {plugin!r} returned shape {arr.shape}, need ({n},)"
        )
    if n and not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"clusterer {plugin!r} returned {arr.dtype}, need integers")
    if n and int(arr.min()) < 1:
        raise ValueError(
            f"clusterer {plugin!r} returned cluster {int(arr.min())}; clusters must "
            "be >= 1 (0 is the label-image background)"
        )
    return arr.astype(np.int64)
