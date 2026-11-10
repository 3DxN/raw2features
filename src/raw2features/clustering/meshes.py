"""Register the glTF scene and its per-glyph properties in the source OME-Zarr.

Viewers discover meshes in an image through a ``meshes`` zarr group next to
``labels``, whose attributes list the files (paths relative to ``meshes/``)::

    {"meshes": [{"name": "<name>", "path": "<name>.glb",
                 "properties": "<name>.properties.json",
                 "embeddings": {"path": "<name>.embeddings.zarr", "model": "uni2_h",
                                "sha256": "...", "sha256_kind": "weights_sha256"}}]}

``properties`` is a JSON array in the NGFF ``image-label.properties`` layout, keyed by
``label-value`` - the glyph's integer ID in the ``.glb`` (``extras["label-value"]``).
``embeddings`` is a zarr array ``(glyphs, dim)``, float32 and uncompressed, whose row
``label-value - 1`` is that glyph's embedding vector; ``sha256`` identifies the
embedding model (``sha256_kind``: its pinned weights, or an output fingerprint). Rows
are chunked so a viewer fetches one glyph's vector without reading the rest. This is
not part of NGFF: ``meshes`` is a raw2features / viewer convention, kept out of the
``ome`` attribute namespace.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from .labels import _open_image, reconsolidate

EMBEDDING_CHUNK_BYTES = 4 << 20  # ~4 MiB of rows per chunk


def _entry(name: str) -> dict:
    return {
        "name": name,
        "path": f"{name}.glb",
        "properties": f"{name}.properties.json",
    }


def _embeddings_path(name: str) -> str:
    return f"{name}.embeddings.zarr"


def _listing(image) -> list[dict]:
    if "meshes" not in image:
        return []
    entries = dict(image["meshes"].attrs).get("meshes")
    return (
        [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
    )


def check_mesh_slot(source_path: str, name: str, *, overwrite: bool = False) -> None:
    """Refuse an existing ``name`` (listed, or its files present) unless ``overwrite``.

    Called before anything is written, so a collision leaves the source untouched.
    """
    if overwrite:
        return
    image, prefix, _ome, _ms = _open_image(source_path)
    directory = os.path.join(source_path, prefix, "meshes")
    files = [
        os.path.join(directory, p)
        for p in (*list(_entry(name).values())[1:], _embeddings_path(name))
    ]
    if any(e.get("name") == name for e in _listing(image)) or any(
        os.path.exists(f) for f in files
    ):
        raise FileExistsError(
            f"mesh {name!r} already exists in {directory}; pass overwrite to replace it"
        )


def write_mesh_index(
    source_path: str,
    name: str,
    glb_path: str,
    properties_path: str,
    features: np.ndarray,
    embedding_model: dict,
    *,
    overwrite: bool = False,
) -> str:
    """Copy the scene and its properties into ``<image>/meshes/``, write the glyphs'
    embedding vectors next to them (row ``label-value - 1``) and list all three
    there as ``name``, keeping every other entry; return the ``.glb`` path.

    ``embedding_model`` is the clusters JSON block (``name``, ``sha256``,
    ``sha256_kind``, ``output_fingerprint``) identifying the model of ``features``.
    """
    check_mesh_slot(source_path, name, overwrite=overwrite)
    image, prefix, _ome, _ms = _open_image(source_path)
    group = image.require_group("meshes")
    entry = _entry(name)
    directory = os.path.join(source_path, prefix, "meshes")
    shutil.copyfile(glb_path, os.path.join(directory, entry["path"]))
    shutil.copyfile(properties_path, os.path.join(directory, entry["properties"]))
    vectors = np.ascontiguousarray(features, dtype=np.float32)
    n, dim = vectors.shape
    rows = max(1, min(n, EMBEDDING_CHUNK_BYTES // max(1, 4 * dim)))
    kwargs: dict = {}
    if image.metadata.zarr_format == 2:
        kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
    else:
        kwargs["dimension_names"] = ["glyph", "feature"]
    path = _embeddings_path(name)
    array = group.create_array(
        path,
        shape=(n, dim),
        chunks=(rows, dim),
        dtype="float32",
        compressors=None,  # vectors barely compress; keep any zarr reader able to read
        fill_value=0.0,
        overwrite=True,  # a collision was refused above unless overwrite
        **kwargs,
    )
    array[...] = vectors
    model = {
        "model": embedding_model["name"],
        "sha256": embedding_model["sha256"],
        "sha256_kind": embedding_model["sha256_kind"],
    }
    array.attrs["raw2features"] = {
        **model,
        "output_fingerprint": embedding_model.get("output_fingerprint"),
        "rows": "glyph label-value - 1 (the .glb node order)",
    }
    entry["embeddings"] = {"path": path, **model}
    others = [e for e in _listing(image) if e.get("name") != name]
    group.attrs["meshes"] = [*others, entry]
    reconsolidate(source_path)
    return os.path.join(directory, entry["path"])
