# Changelog

Notable changes to raw2features, newest first. This project follows
[Semantic Versioning](https://semver.org).

## Unreleased

### Added

- **3D point-cloud sampling** for z-stack OME-Zarr sources. Pass `raw2features embed
  --points FILE` (a CSV/Parquet of `x,y,z` in µm, plus an optional `id`) to sample one 2D
  patch per point on its nearest z-plane and embed it with any existing 2D model, instead of
  tissue-grid tiling. The reader now serves any z-plane (`Region.z`), and each grid gains 1:1
  `z_index` / `source_points` (and optional `point_ids`) sibling arrays plus a `sampling`
  header block (`coords_convention: "level0_xy_z_points"`); `coords` stay level-0 `(x, y)`, so
  2D consumers are unaffected. Add `--source-z-spacing` when the z axis is uncalibrated;
  `raw2features sample --z N` writes a synthetic z-stack to try it offline. See
  [`docs/SPEC.md`](docs/SPEC.md#3d-point-sampled-grids) and `docs/usage.md`.

## [0.1.0] - 2026-06-30

First public release.

raw2features reads whole-slide images in OME-Zarr / OME-NGFF - local or remote - and emits
patch- and slide-level foundation-model embeddings as a self-describing `.embeddings.zarr`
store. The reader, segmenter, patcher, embedder and sink are pluggable seams, so a new model
or storage backend ships as a small package with no fork.

What 0.1.0 provides:

- **Cloud-direct reads** of NGFF v0.4 and v0.5 (`http(s)`, `s3`, `gs`): the whole
  segment -> tile -> embed pipeline runs against a remote store with nothing downloaded.
- **Exact-MPP patch extraction** by downsampling from the nearest finer pyramid level, so
  embeddings are comparable across slides and scanners.
- **30+ patch encoders** (ImageNet baselines plus pathology foundation models) and several
  **slide encoders**, each with preprocessing transcribed from its model card and weights
  pinned to a sha256 + Hugging Face revision for full provenance.
- **Decode-once multi-model fan-out**: many encoders in one pass, one store, 1:1 aligned to
  the patch coordinates; resumable and additive across re-runs.
- **Optional quality control** (GrandQC artifact scoring) and **stain normalization**
  (Macenko, Reinhard, Vahadane).
- **Multiplex** (spatial proteomics) alongside H&E on the same seams: channel-aware reads,
  nuclear segmentation, and the marker-aware KRONOS encoder.
- A **specified, self-describing store** with a JSON Schema and a `validate-store` command,
  plus one-way exports to scverse SpatialData, pathology-MIL HDF5, and QuPath GeoJSON.
- **Cohort tooling**: `embed-many` (sharded, resumable, with SLURM templates), a benchmark
  harness, and six runnable tutorial notebooks.

[0.1.0]: https://github.com/CraigMyles/raw2features/releases/tag/v0.1.0
