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
  two footprint diagonals, each a thin closed prism with a triangular cross-section
  (circumradius ``CROSS_RADIUS`` of the shorter side) - faces, not glTF lines, which
  many viewers do not display. Each arm reaches ``cross_arm`` of the way from the
  centre to its corner (default ``CROSS_ARM``): full-length diagonals of neighbouring
  footprints are collinear and join up into a lattice that hides which patch is
  which.

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
_TRIANGLES = 4

GLYPHS = ("quad", "marker")
MARKER_RADIUS = 0.04  # marker ellipsoid semi-axis, as a fraction of the patch side
CROSS_RADIUS = 0.01  # crosshair prism circumradius, as a fraction of the patch side
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


def prism(
    p0: np.ndarray, p1: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices (24,3) float32, normals (24,3) float32, faces (8,3) uint16)``: a
    closed, flat-shaded prism from ``p0`` to ``p1`` whose cross-section is an
    equilateral triangle of circumradius ``radius`` (one corner towards +z for an
    axis in the xy plane), faces wound counter-clockwise seen from outside."""
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    axis = (p1 - p0) / np.linalg.norm(p1 - p0)
    up = np.float64([0, 0, 1]) if abs(axis[2]) < 0.9 else np.float64([1, 0, 0])
    side = np.cross(axis, up)
    side /= np.linalg.norm(side)
    up = np.cross(side, axis)
    angles = np.pi / 2 + 2 * np.pi / 3 * np.arange(3)
    ring = radius * (np.cos(angles)[:, None] * side + np.sin(angles)[:, None] * up)
    corners = np.concatenate([p0 + ring, p1 + ring])  # 0-2 at p0, 3-5 at p1
    triangles = [(0, 1, 2), (3, 4, 5)]
    for a in range(3):
        b = (a + 1) % 3
        triangles += [(a, b, b + 3), (a, b + 3, a + 3)]
    centre = (p0 + p1) / 2
    vertices, normals = [], []
    for triangle in triangles:  # one vertex per face corner, for flat shading
        pts = corners[list(triangle)]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if normal @ (pts.mean(0) - centre) < 0:  # convex: outward is away from centre
            pts, normal = pts[::-1], -normal
        vertices.append(pts)
        normals.append(np.tile(normal / np.linalg.norm(normal), (3, 1)))
    return (
        np.concatenate(vertices).astype(np.float32),
        np.concatenate(normals).astype(np.float32),
        np.arange(3 * len(triangles), dtype=np.uint16).reshape(-1, 3),
    )


def crosshair(
    height: float, width: float, arm: float = 1.0, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vertices, normals, faces)`` as for ``prism``: two prisms of circumradius
    ``radius`` along the diagonals of a ``width x height`` footprint centred on the
    origin in the z = 0 plane - first from ``(-x, -y)`` to ``(x, y)``, then from
    ``(x, -y)`` to ``(-x, y)`` - each arm ``arm`` of the way to its corner."""
    x, y = arm * width / 2.0, arm * height / 2.0
    parts = [
        prism((-x, -y, 0.0), (x, y, 0.0), radius),
        prism((x, -y, 0.0), (-x, y, 0.0), radius),
    ]
    n = len(parts[0][0])
    return (
        np.concatenate([v for v, _n, _f in parts]),
        np.concatenate([nrm for _v, nrm, _f in parts]),
        np.concatenate([f + k * n for k, (_v, _n, f) in enumerate(parts)]),
    )


def check_glyph(glyph: str, cross_arm: float) -> None:
    """Refuse an unknown glyph or a crosshair arm outside ``(0, 1]``."""
    if glyph not in GLYPHS:
        raise ValueError(f"unknown glTF glyph {glyph!r}; choose from {GLYPHS}")
    if not 0.0 < cross_arm <= 1.0:
        raise ValueError(f"cross arm must be in (0, 1], got {cross_arm}")


def _parts(glyph: str, height: float, width: float, cross_arm: float) -> list[tuple]:
    """``(vertices, normals, faces)`` per triangle primitive of one glyph."""
    if glyph == "quad":
        vertices, faces = quad(height, width)
        normals = np.tile(np.float32([0, 0, 1]), (len(vertices), 1))
        return [(vertices, normals, faces)]
    short = min(height, width)
    sphere = ellipsoid(
        MARKER_RADIUS * width, MARKER_RADIUS * height, MARKER_RADIUS * short
    )
    return [sphere, crosshair(height, width, cross_arm, CROSS_RADIUS * short)]


def _geometry(glyph: str, cross_arm: float) -> str:
    if glyph == "quad":
        return "patch footprint quad, axis-aligned, z = plane index"
    return (
        f"ellipsoid at the footprint centre (semi-axes {MARKER_RADIUS} x footprint "
        f"side) and the footprint diagonals as triangular prisms (circumradius "
        f"{CROSS_RADIUS} x footprint side), {cross_arm} of the way from the centre "
        "to each corner, z = plane index"
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
        for vertices, normals, indices in _parts(glyph, height, width, cross_arm):
            base, n = len(accessors), len(vertices)
            blobs += [
                (vertices.tobytes(), _ARRAY_BUFFER),
                (normals.tobytes(), _ARRAY_BUFFER),
                (indices.reshape(-1).tobytes(), _ELEMENT_ARRAY_BUFFER),
            ]
            accessors += [
                {"bufferView": base, "componentType": _FLOAT, "count": n,
                 "type": "VEC3", "min": vertices.min(0).tolist(),
                 "max": vertices.max(0).tolist()},
                {"bufferView": base + 1, "componentType": _FLOAT, "count": n,
                 "type": "VEC3"},
                {"bufferView": base + 2, "componentType": _USHORT,
                 "count": int(indices.size), "type": "SCALAR"},
            ]  # fmt: skip
            shape_primitives.append(
                {"attributes": {"POSITION": base, "NORMAL": base + 1},
                 "indices": base + 2, "material": 0, "mode": _TRIANGLES}
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
