"""Binary glTF 2.0 (``.glb``) scene: one mesh per embedded location.

Mesh ``i`` and node ``i`` both carry ``name == feature_ids[i]`` (and
``extras.feature_id``), so the clusters JSON - keyed by the same feature IDs - maps
meshes to cluster values for colour-coded display. Each node's ``translation`` is the
patch footprint centre (``footprint``) as level-0 ``(x, y, z)`` (voxel-centre
convention); footprints of equal size share one geometry. The glyph drawn there is
one of ``GLYPHS``:

* ``quad`` - the footprint itself: axis-aligned, exactly the level-0 area the
  embedding was computed from, in its z plane, with a double-sided material so it is
  visible from both sides.
* ``marker`` - a small ellipsoid at the centre (semi-axes ``MARKER_RADIUS`` of the
  footprint side along x/y, of the shorter side along z) plus a crosshair along the
  two footprint diagonals, drawn as lines rather than faces. Each arm reaches
  ``cross_arm`` of the way from the centre to its corner (default ``CROSS_ARM``):
  full-length diagonals of neighbouring footprints are collinear and join up into a
  lattice that hides which patch is which.

No colours are baked in. Written with numpy + ``struct`` only.

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
_LINES, _TRIANGLES = 1, 4

GLYPHS = ("quad", "marker")
MARKER_RADIUS = 0.08  # marker ellipsoid semi-axis, as a fraction of the patch side
CROSS_ARM = 0.35  # marker crosshair arm, as a fraction of the centre-corner distance
_RINGS, _SEGMENTS = 8, 12  # ellipsoid tessellation (latitude, longitude)


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


def ellipsoid(
    rx: float, ry: float, rz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices (V,3) float32, normals (V,3) float32, faces (F,3) uint16)``: a
    closed ellipsoid centred on the origin with semi-axes ``rx, ry, rz``, faces wound
    counter-clockwise seen from outside."""
    theta = np.linspace(0.0, np.pi, _RINGS + 1)[1:-1, None]  # inner rings, from +z
    phi = np.linspace(0.0, 2.0 * np.pi, _SEGMENTS, endpoint=False)[None, :]
    rings = np.stack(
        np.broadcast_arrays(
            np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
        ),
        -1,
    ).reshape(-1, 3)
    unit = np.concatenate([[(0.0, 0.0, 1.0)], rings, [(0.0, 0.0, -1.0)]])
    radii = np.asarray([rx, ry, rz])
    normals = unit / radii  # gradient of the implicit surface
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    top, bottom = 0, len(unit) - 1

    def ring(k: int, j: int) -> int:
        return 1 + k * _SEGMENTS + j % _SEGMENTS

    faces = []
    for j in range(_SEGMENTS):
        faces.append((top, ring(0, j), ring(0, j + 1)))
        for k in range(_RINGS - 2):
            a, b = ring(k, j), ring(k, j + 1)
            c, d = ring(k + 1, j), ring(k + 1, j + 1)
            faces += [(a, c, d), (a, d, b)]
        faces.append((ring(_RINGS - 2, j), bottom, ring(_RINGS - 2, j + 1)))
    return (
        (unit * radii).astype(np.float32),
        normals.astype(np.float32),
        np.asarray(faces, dtype=np.uint16),
    )


def crosshair(
    height: float, width: float, arm: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (4,3) float32, segments (2,2) uint16)``: the diagonals of a
    ``width x height`` footprint centred on the origin in the z = 0 plane, each arm
    ``arm`` of the way from the centre to its corner."""
    vertices, _faces = quad(arm * height, arm * width)
    return vertices, np.asarray([(0, 2), (1, 3)], dtype=np.uint16)


def check_glyph(glyph: str, cross_arm: float) -> None:
    """Refuse an unknown glyph or a crosshair arm outside ``(0, 1]``."""
    if glyph not in GLYPHS:
        raise ValueError(f"unknown glTF glyph {glyph!r}; choose from {GLYPHS}")
    if not 0.0 < cross_arm <= 1.0:
        raise ValueError(f"cross arm must be in (0, 1], got {cross_arm}")


def _parts(glyph: str, height: float, width: float, cross_arm: float) -> list[tuple]:
    """``(vertices, normals or None, indices, mode)`` per primitive of one glyph."""
    if glyph == "quad":
        vertices, faces = quad(height, width)
        normals = np.tile(np.float32([0, 0, 1]), (len(vertices), 1))
        return [(vertices, normals, faces, _TRIANGLES)]
    r = MARKER_RADIUS * min(height, width)
    vertices, normals, faces = ellipsoid(
        MARKER_RADIUS * width, MARKER_RADIUS * height, r
    )
    lines, segments = crosshair(height, width, cross_arm)
    return [
        (vertices, normals, faces, _TRIANGLES),
        (lines, None, segments, _LINES),
    ]


def _geometry(glyph: str, cross_arm: float) -> str:
    if glyph == "quad":
        return "patch footprint quad, axis-aligned, z = plane index"
    return (
        f"ellipsoid at the footprint centre (semi-axes {MARKER_RADIUS} x footprint "
        f"side) and the footprint diagonals as lines, {cross_arm} of the way from "
        "the centre to each corner, z = plane index"
    )


_MATERIAL = {
    "quad": {"name": "footprint", "doubleSided": True},
    "marker": {"name": "marker"},
}


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(
    path: str,
    feature_ids: list[str],
    footprints: Footprints,
    *,
    glyph: str = "quad",
    cross_arm: float = CROSS_ARM,
    extras: dict | None = None,
) -> str:
    """Write the scene to ``path`` and return it; ``glyph`` is one of ``GLYPHS``,
    ``cross_arm`` the ``marker`` crosshair arm as a fraction of centre to corner."""
    check_glyph(glyph, cross_arm)
    sizes = footprints.size_zyx[:, 1:]  # (height, width) per location
    shapes, shape_of = np.unique(sizes, axis=0, return_inverse=True)
    blobs: list[tuple[bytes, int]] = []
    accessors: list[dict] = []
    primitives: list[list[dict]] = []  # per shape
    for height, width in shapes.tolist():
        shape_primitives = []
        for vertices, normals, indices, mode in _parts(glyph, height, width, cross_arm):
            n = len(vertices)
            attributes = {"POSITION": len(accessors)}
            blobs.append((vertices.tobytes(), _ARRAY_BUFFER))
            accessors.append(
                {"bufferView": len(accessors), "componentType": _FLOAT, "count": n,
                 "type": "VEC3", "min": vertices.min(0).tolist(),
                 "max": vertices.max(0).tolist()}
            )  # fmt: skip
            if normals is not None:
                attributes["NORMAL"] = len(accessors)
                blobs.append((normals.tobytes(), _ARRAY_BUFFER))
                accessors.append(
                    {"bufferView": len(accessors), "componentType": _FLOAT,
                     "count": n, "type": "VEC3"}
                )  # fmt: skip
            index = len(accessors)
            blobs.append((indices.reshape(-1).tobytes(), _ELEMENT_ARRAY_BUFFER))
            accessors.append(
                {"bufferView": index, "componentType": _USHORT,
                 "count": int(indices.size), "type": "SCALAR"}
            )  # fmt: skip
            shape_primitives.append(
                {"attributes": attributes, "indices": index, "material": 0,
                 "mode": mode}
            )  # fmt: skip
        primitives.append(shape_primitives)
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
                "primitives": primitives[int(shape_of[i])],
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
                "geometry": _geometry(glyph, cross_arm),
                **(extras or {}),
            },
        },
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [_MATERIAL[glyph]],
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
