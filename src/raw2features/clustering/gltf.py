"""Binary glTF 2.0 (``.glb``) scene: one quad mesh per embedded location.

Mesh ``i`` and node ``i`` both carry ``name == feature_ids[i]`` (and
``extras.feature_id``), so the clusters JSON - keyed by the same feature IDs - maps
meshes to cluster values for colour-coded display. Each quad is the patch footprint
(``footprint``): axis-aligned, exactly the level-0 area the embedding was computed
from, in its z plane. Quads of equal size share one geometry; each node's
``translation`` is the footprint centre as level-0 ``(x, y, z)`` (voxel-centre
convention). A double-sided material keeps quads visible from both sides. No
colours are baked in. Written with numpy + ``struct`` only.

Units are level-0 voxels for now - see the NOTE in ``footprint`` for the move to
physical units (glTF's native unit is the metre).
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .footprint import Footprints

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT, _USHORT = 5126, 5123
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def quad(height: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (4,3) float32, faces (2,3) uint16)``: a ``width x height`` quad
    centred on the origin in the z = 0 plane, edges along x/y, wound
    counter-clockwise seen from +z."""
    hx, hy = width / 2.0, height / 2.0
    vertices = np.asarray(
        [(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)],
        dtype=np.float32,
    )
    return vertices, np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.uint16)


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(
    path: str,
    feature_ids: list[str],
    footprints: Footprints,
    *,
    extras: dict | None = None,
) -> str:
    """Write the scene to ``path`` and return it."""
    sizes = footprints.size_zyx[:, 1:]  # (height, width) per location
    shapes, shape_of = np.unique(sizes, axis=0, return_inverse=True)
    blobs: list[tuple[bytes, int]] = []
    accessors: list[dict] = []
    primitives: list[dict] = []
    normal = np.tile(np.float32([0, 0, 1]), (4, 1))
    for height, width in shapes.tolist():
        vertices, faces = quad(height, width)
        base = len(accessors)
        blobs += [
            (vertices.tobytes(), _ARRAY_BUFFER),
            (normal.tobytes(), _ARRAY_BUFFER),
            (faces.reshape(-1).tobytes(), _ELEMENT_ARRAY_BUFFER),
        ]
        accessors += [
            {"bufferView": base, "componentType": _FLOAT, "count": 4,
             "type": "VEC3", "min": vertices.min(0).tolist(),
             "max": vertices.max(0).tolist()},
            {"bufferView": base + 1, "componentType": _FLOAT, "count": 4,
             "type": "VEC3"},
            {"bufferView": base + 2, "componentType": _USHORT,
             "count": int(faces.size), "type": "SCALAR"},
        ]  # fmt: skip
        primitives.append(
            {
                "attributes": {"POSITION": base, "NORMAL": base + 1},
                "indices": base + 2,
                "material": 0,
                "mode": 4,
            }
        )
    views, offset = [], 0
    for blob, target in blobs:
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(blob),
                "target": target,
            }
        )
        offset += len(blob) + (4 - len(blob) % 4) % 4
    binary = b"".join(_pad(b, b"\x00") for b, _target in blobs)

    meshes, nodes = [], []
    centres = footprints.centres_zyx()
    for i, (fid, (z, y, x)) in enumerate(zip(feature_ids, centres, strict=True)):
        meshes.append(
            {
                "name": fid,
                "primitives": [primitives[int(shape_of[i])]],
                "extras": {"feature_id": fid},
            }
        )
        nodes.append({"name": fid, "mesh": i,
                      "translation": [float(x), float(y), float(z)],
                      "extras": {"feature_id": fid}})  # fmt: skip
    doc = {
        "asset": {
            "version": "2.0",
            "generator": "raw2features cluster",
            "extras": {
                "units": "level-0 voxels (voxel centres at integer indices); "
                "node translation = footprint centre (x, y, z); image y points down",
                "geometry": "patch footprint quad, axis-aligned, z = plane index",
                **(extras or {}),
            },
        },
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [{"name": "footprint", "doubleSided": True}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    js = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(js), _CHUNK_JSON) + js)
        fh.write(struct.pack("<II", len(binary), _CHUNK_BIN) + binary)
    return path
