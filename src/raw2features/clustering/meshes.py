"""Register the glTF scene and its per-glyph properties in the source OME-Zarr.

Viewers discover meshes in an image through a ``meshes`` zarr group next to
``labels``, whose attributes list the files (paths relative to ``meshes/``)::

    {"meshes": [{"name": "<name>", "path": "<name>.glb",
                 "properties": "<name>.properties.json"}]}

``properties`` is a JSON array in the NGFF ``image-label.properties`` layout, keyed by
``label-value`` - the glyph's integer ID in the ``.glb`` (``extras["label-value"]``).
This is not part of NGFF: ``meshes`` is a raw2features / viewer convention, kept out
of the ``ome`` attribute namespace.
"""

from __future__ import annotations

import os
import shutil

from .labels import _open_image, reconsolidate


def _entry(name: str) -> dict:
    return {
        "name": name,
        "path": f"{name}.glb",
        "properties": f"{name}.properties.json",
    }


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
    files = [os.path.join(directory, p) for k, p in _entry(name).items() if k != "name"]
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
    *,
    overwrite: bool = False,
) -> str:
    """Copy the scene and its properties into ``<image>/meshes/`` and list them
    there as ``name``, keeping every other entry; return the ``.glb`` path."""
    check_mesh_slot(source_path, name, overwrite=overwrite)
    image, prefix, _ome, _ms = _open_image(source_path)
    group = image.require_group("meshes")
    entry = _entry(name)
    directory = os.path.join(source_path, prefix, "meshes")
    shutil.copyfile(glb_path, os.path.join(directory, entry["path"]))
    shutil.copyfile(properties_path, os.path.join(directory, entry["properties"]))
    others = [e for e in _listing(image) if e.get("name") != name]
    group.attrs["meshes"] = [*others, entry]
    reconsolidate(source_path)
    return os.path.join(directory, entry["path"])
