"""Binary glTF 2.0 (``.glb``) scene: one sphere mesh per embedded location.

Mesh ``i`` and node ``i`` both carry ``name == feature_ids[i]`` (and
``extras.feature_id``), so the clusters JSON - keyed by the same feature IDs - maps
meshes to cluster values for colour-coded display. All meshes reference one shared
icosphere geometry; each node's ``translation`` places it at the location's level-0
``(x, y, z)`` voxel. No colours are baked in. Written with numpy + ``struct`` only.
"""

from __future__ import annotations

import json
import struct

import numpy as np

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT, _USHORT = 5126, 5123
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def icosphere(radius: float, subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V,3) float32, faces (F,3) uint16)`` of a sphere of ``radius``."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.asarray(v, dtype=np.float64) for v in verts]

    def midpoint(a: int, b: int, cache: dict[tuple[int, int], int]) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            vertices.append((vertices[a] + vertices[b]) / 2.0)
            cache[key] = len(vertices) - 1
        return cache[key]

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(a, b, cache)
            bc = midpoint(b, c, cache)
            ca = midpoint(c, a, cache)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    v = np.asarray(vertices)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return v.astype(np.float32), np.asarray(faces, dtype=np.uint16)


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(
    path: str,
    feature_ids: list[str],
    centers_zyx: np.ndarray,
    *,
    radius_px: float = 16.0,
    extras: dict | None = None,
) -> str:
    """Write the scene to ``path`` and return it."""
    vertices, faces = icosphere(radius_px)
    normals = (vertices / np.float32(radius_px)).astype(np.float32)
    blobs = [vertices.tobytes(), normals.tobytes(), faces.reshape(-1).tobytes()]
    views, offset = [], 0
    for blob, target in zip(
        blobs, (_ARRAY_BUFFER, _ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER), strict=True
    ):
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(blob),
                "target": target,
            }
        )
        offset += len(blob) + (4 - len(blob) % 4) % 4
    binary = b"".join(_pad(b, b"\x00") for b in blobs)

    accessors = [
        {"bufferView": 0, "componentType": _FLOAT, "count": len(vertices),
         "type": "VEC3", "min": vertices.min(0).tolist(),
         "max": vertices.max(0).tolist()},
        {"bufferView": 1, "componentType": _FLOAT, "count": len(normals),
         "type": "VEC3"},
        {"bufferView": 2, "componentType": _USHORT, "count": int(faces.size),
         "type": "SCALAR"},
    ]  # fmt: skip
    primitive = {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}
    meshes, nodes = [], []
    for i, (fid, (z, y, x)) in enumerate(zip(feature_ids, centers_zyx, strict=True)):
        meshes.append(
            {"name": fid, "primitives": [primitive], "extras": {"feature_id": fid}}
        )
        nodes.append({"name": fid, "mesh": i,
                      "translation": [float(x), float(y), float(z)],
                      "extras": {"feature_id": fid}})  # fmt: skip
    doc = {
        "asset": {
            "version": "2.0",
            "generator": "raw2features cluster",
            "extras": {
                "units": "level-0 voxels; node translation = (x, y, z)",
                "sphere_radius_px": radius_px,
                **(extras or {}),
            },
        },
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
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
