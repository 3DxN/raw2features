"""Point-cloud sampler - the 3D patch-around-point selection seam.

Where :class:`~raw2features.patcher.grid.GridPatcher` tiles a slide on a regular
grid and keeps tissue cells, the point sampler reads one 2D patch at each 3D point
of an external point cloud (physical µm). It shares the exact-MPP read plan with the
grid patcher (``reader.level_for_mpp`` + ``level0_patch`` maths) so a point-sampled
patch is read and resampled identically to a grid patch of the same geometry - only
the *selection* (and the per-patch z-plane) differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.mpp import level0_step_px
from raw2features.core.points import PointCloud, load_point_cloud, map_points_to_grid

from .base import PatchGrid

if TYPE_CHECKING:
    import numpy as np

    from raw2features.readers.base import WSISource


class PointSampler:
    """Sample one 2D patch per 3D point of a point cloud, at an exact target MPP."""

    name = "points"

    def __init__(
        self,
        points_path: str,
        target_mpp: float = 1.0,
        patch_px: int = 224,
        snap_to_level: bool = False,
        mpp_tolerance: float = 0.001,
        allow_upsample: bool = False,
    ) -> None:
        self.points_path = points_path
        self.target_mpp = target_mpp
        self.patch_px = patch_px
        self.snap_to_level = snap_to_level
        self.mpp_tolerance = mpp_tolerance
        self.allow_upsample = allow_upsample
        self._cloud: PointCloud | None = None

    def load(self) -> PointCloud:
        """Load (and cache) the point cloud; its ``sha256`` feeds the content hash."""
        if self._cloud is None:
            self._cloud = load_point_cloud(self.points_path)
        return self._cloud

    def build_grid(self, reader: WSISource) -> PatchGrid:
        """Resolve the shared read plan (identical to ``GridPatcher.build_grid``).

        The ``n_rows``/``n_cols``/``xs``/``ys`` grid fields do not apply to a point
        cloud and are left empty; the read geometry (level, read_px, level0_patch,
        resample) is what the decode path consumes and is resolved exactly as for a
        regular grid so the two paths read a patch the same way.
        """
        mpp0 = reader.mpp
        if mpp0 is None:
            raise ValueError(
                "source physical pixel size is unknown; pass --source-mpp so the µm "
                "point coordinates can be placed on the slide."
            )
        plan = reader.level_for_mpp(
            self.target_mpp,
            self.patch_px,
            snap_to_level=self.snap_to_level,
            tolerance=self.mpp_tolerance,
            allow_upsample=self.allow_upsample,
        )
        level0_patch = round(self.patch_px * plan.achieved_mpp / mpp0)
        level0_step = level0_step_px(self.patch_px, plan.achieved_mpp, mpp0)
        return PatchGrid(
            target_mpp=self.target_mpp,
            achieved_mpp=plan.achieved_mpp,
            patch_px=self.patch_px,
            step_out_px=self.patch_px,
            level0_patch=level0_patch,
            level0_step=level0_step,
            read_level=plan.level,
            read_px=plan.read_px,
            resample=plan.resample,
            needs_resample=plan.needs_resample,
            n_rows=0,
            n_cols=0,
            xs=(),
            ys=(),
        )

    def sample(
        self, reader: WSISource, grid: PatchGrid
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Place the cloud on the slide.

        Returns ``(coords, z_index, points_um, ids)``:

        * ``coords``: ``(M, 2)`` int32 level-0 top-left of each kept patch.
        * ``z_index``: ``(M,)`` int32 level-0 z-plane index per patch.
        * ``points_um``: ``(M, 3)`` float64 source µm center per patch.
        * ``ids``: ``(M,)`` object array of point ids (empty strings if none).

        Points outside the slide extent are dropped (warned), so ``M <= N``.
        """
        cloud = self.load()
        return map_points_to_grid(cloud, reader, grid.level0_patch)
