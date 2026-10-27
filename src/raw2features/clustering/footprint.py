"""The area each embedding was computed from: an axis-aligned quad per location.

The grid patcher tiles axis-aligned squares: patch ``i`` of a grid covers the level-0
voxels ``x0 .. x0 + level0_patch - 1`` by ``y0 .. y0 + level0_patch - 1`` (with
``(x0, y0) = coords[i]``, the top-left) on the grid's z plane (``patching.z_index``,
0 for 2D). It is read at another pyramid level and resampled to ``patch_px``, but
the tissue it shows is exactly that box. Quads are therefore never rotated: their
edges are parallel to the image x/y axes, and their normal is the z axis.

Coordinates follow the OME-NGFF convention: the coordinate of a voxel is its centre,
so voxel ``i`` spans ``[i - 0.5, i + 0.5)``. A quad's continuous extent is
``[x0 - 0.5, x0 + level0_patch - 0.5)`` along x (likewise y) at ``z = z_index``.

NOTE - pixel units, to be replaced by physical units (µm)
-------------------------------------------------------------
For now the footprints, the glTF scene and the label rasterisation are all in
**level-0 voxel units** of the source image; coarser label levels are derived by
the per-axis downsample. To move to physical units consistently:

1. **Source of truth for the pixel->µm transform**: the source OME-Zarr multiscales
   metadata (``multiscales[0].axes[*].unit`` and each dataset's
   ``coordinateTransformations`` ``scale`` / ``translation``; NGFF 0.4 attrs or the
   0.5 ``ome`` key), as parsed by ``readers.omezarr.OmeZarrReader`` and
   ``clustering.labels._levels``. The embeddings store header snapshots the level-0
   part per grid: ``source.scale_um`` (x/y µm per voxel), ``source.z_scale_um``,
   ``source.level0_translation_um``, ``source.axis_units`` and ``patching.z_um``.
   Read those in ``collect.collect_locations`` and check them against the source
   when it is available (``--write-labels`` already opens it); refuse when units
   are missing/arbitrary (``source.axis_units`` not µm), as the patcher does.
2. **Footprints**: ``origin_um = index * scale_um + level0_translation_um`` per axis
   (x, y and z with ``z_scale_um``), applied to the voxel-centre extents above, i.e.
   ``x_um in [(x0 - 0.5) * sx + tx, (x0 + level0_patch - 0.5) * sx + tx)``. Use
   ``level0_patch * scale_um`` as the side (``level0_patch`` is the rounded level-0
   extent actually read; ``patching.achieved_patch_um`` is the nominal side and can
   differ by < 1 voxel). Keep x and y separate - pixels may be anisotropic.
3. **glTF**: glTF's unit is the metre, so write ``1e-6 * µm`` in node translations
   and quad vertices (or keep µm under a root node scaled by ``1e-6``), and record
   the units in ``asset.extras``. Image y points down while glTF is y-up; keep the
   handedness note (or add a root flip) when changing the frame.
4. **Label maps**: the arrays stay voxel grids with the source's
   ``coordinateTransformations`` (already copied). Rasterise each level by mapping
   the µm footprint into that level's index space with *its own* transform,
   ``index = (µm - translation_k) / scale_k``, instead of level-0 index /
   downsample, so labels and glTF agree in physical space for anisotropic or
   translated pyramids.
5. **JSON**: write the µm footprints next to (or instead of) the voxel ones and
   state the units in the record; bump ``CLUSTERS_FORMAT``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Footprints:
    """Level-0 voxel boxes: ``origin_zyx[i]`` (first voxel) and ``size_zyx[i]``."""

    origin_zyx: np.ndarray  # (N, 3) int64
    size_zyx: np.ndarray  # (N, 3) int64; z is 1 (one plane), y/x = level0_patch

    def centres_zyx(self) -> np.ndarray:
        """Continuous centres (voxel-centre convention), float64 ``(N, 3)``."""
        return self.origin_zyx + (self.size_zyx - 1) / 2.0


def grid_footprints(coords_xy: np.ndarray, level0_patch: int, z: int) -> Footprints:
    """Footprints of one grid's patches (``coords`` are level-0 top-left ``(x, y)``)."""
    coords = np.asarray(coords_xy, dtype=np.int64).reshape(-1, 2)
    n = coords.shape[0]
    origin = np.empty((n, 3), dtype=np.int64)
    origin[:, 0] = z
    origin[:, 1] = coords[:, 1]
    origin[:, 2] = coords[:, 0]
    size = np.empty((n, 3), dtype=np.int64)
    size[:, 0] = 1
    size[:, 1:] = int(level0_patch)
    return Footprints(origin, size)


def concat(parts: list[Footprints]) -> Footprints:
    if not parts:
        empty = np.zeros((0, 3), dtype=np.int64)
        return Footprints(empty, empty.copy())
    return Footprints(
        np.concatenate([p.origin_zyx for p in parts]),
        np.concatenate([p.size_zyx for p in parts]),
    )
