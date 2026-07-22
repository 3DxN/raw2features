"""Tiny synthetic sample data, so the quickstart runs with no download / token / GPU.

``write_sample_slide`` writes a small multiscale OME-NGFF v0.4 RGB store with a
tissue-like blob on a bright background - enough for the tissue segmenter to find
patches and for the whole pipeline (read → segment → tile → embed) to run end to end
on a laptop CPU. The blob has two H&E-ish compartments (a haematoxylin-rich and an
eosin-rich side) and a few darker gland-like rings, so extracted features carry some
structure rather than uniform noise. It is synthetic, not real histology; use it to
learn the tool.
"""

from __future__ import annotations


def write_sample_slide(
    out_path: str,
    *,
    mpp0: float = 0.5,
    size: int = 1024,
    levels: int = 3,
    seed: int = 0,
    z: int = 1,
    z_spacing: float = 1.0,
) -> str:
    """Write a synthetic OME-NGFF v0.4 slide and return its path.

    ``size`` is the level-0 side in pixels (square); ``levels`` halved pyramid levels;
    ``mpp0`` the level-0 microns/pixel. ``z`` is the number of z-planes (a z-stack when
    ``> 1``; the default ``1`` is byte-identical to the flat 2D slide) and ``z_spacing``
    their µm spacing - enough to exercise 3D point sampling. Needs the ``[zarr]`` extra.
    """
    import numpy as np
    import zarr

    rng = np.random.default_rng(seed)
    h = w = int(size)
    # Level-0 RGB: bright background with an elliptical H&E-ish "tissue" blob, so the
    # default otsu (saturation) segmenter keeps a sensible patch set.
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ry, rx = (yy - h * 0.5) / (h * 0.40), (xx - w * 0.5) / (w * 0.32)
    inside = ry**2 + rx**2 <= 1.0
    # Two tissue compartments blended left -> right (haematoxylin-rich nuclei vs
    # eosin-rich stroma) so features separate into regions, not uniform noise.
    haem = np.array([120, 90, 165], dtype=np.float32)  # purple-ish, nuclei-dense
    eos = np.array([205, 120, 165], dtype=np.float32)  # pink-ish, stroma
    frac = np.clip(xx / max(w - 1, 1), 0.0, 1.0)[..., None]
    base = haem * (1.0 - frac) + eos * frac
    # A few darker gland-like rings for local structure.
    glands = np.zeros((h, w), dtype=np.float32)
    for _ in range(max(6, size // 256)):
        gy, gx = rng.uniform(0.2, 0.8, size=2) * np.array([h, w], dtype=np.float32)
        gr = rng.uniform(0.03, 0.07) * size
        dist = np.hypot(yy - gy, xx - gx)
        glands += np.exp(-((dist - gr) ** 2) / (2.0 * (gr * 0.3) ** 2))
    base -= 70.0 * np.clip(glands, 0.0, 1.0)[..., None]
    noise = rng.integers(-18, 18, size=(h, w, 3))
    blob = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[inside] = blob[inside]

    # Build the z-stack: plane 0 is ``img``; deeper planes shift brightness by a
    # per-plane constant so the planes are distinguishable (a test can assert the right
    # plane was read). z=1 leaves a single plane identical to the flat 2D slide.
    depth = max(1, int(z))
    vol = np.stack(
        [np.clip(img.astype(np.int16) + p * 8, 0, 255).astype(np.uint8)
         for p in range(depth)],
        axis=0,
    )  # (z, H, W, 3)

    g = zarr.open_group(str(out_path), mode="w", zarr_format=2)
    sizes: list[tuple[int, int]] = []
    cur = vol
    for i in range(int(levels)):
        lh, lw = cur.shape[1], cur.shape[2]
        sizes.append((lh, lw))
        a = g.create_array(
            str(i),
            shape=(1, 3, depth, lh, lw),
            chunks=(1, 1, 1, min(256, lh), min(256, lw)),
            dtype="uint8",
        )
        a[0] = np.transpose(cur, (3, 0, 1, 2))  # (z,H,W,3) -> (c=3, z, H, W)
        cur = cur[:, ::2, ::2]  # halve y/x for the next pyramid level (z unchanged)

    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1.0, 1.0, z_spacing, mpp0 * (2**i), mpp0 * (2**i)],
                }
            ],
        }
        for i in range(len(sizes))
    ]
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "sample", "axes": axes, "datasets": datasets}
    ]
    return str(out_path)


def write_sample_points(
    out_path: str,
    *,
    mpp0: float = 0.5,
    size: int = 1024,
    z: int = 4,
    z_spacing: float = 1.0,
    n_side: int = 4,
) -> str:
    """Write a tiny CSV point cloud (x,y,z in µm, + id) inside a sample slide's tissue.

    Places an ``n_side`` x ``n_side`` grid of points across the central tissue ellipse,
    cycling the z-plane per point, so the whole 3D point-sampling path runs offline. The
    coordinates are physical µm to match :func:`write_sample_slide` (``mpp0``, level-0
    ``size`` px, ``z`` planes at ``z_spacing`` µm).
    """
    import csv

    span = size * mpp0  # slide extent in µm
    lo, hi = 0.30 * span, 0.70 * span  # stay inside the tissue ellipse
    step = (hi - lo) / max(1, n_side - 1)
    rows = []
    k = 0
    for r in range(n_side):
        for c in range(n_side):
            x_um = lo + c * step
            y_um = lo + r * step
            z_um = (k % max(1, z)) * z_spacing
            rows.append((f"{x_um:.4f}", f"{y_um:.4f}", f"{z_um:.4f}", f"pt_{k:03d}"))
            k += 1
    with open(str(out_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "id"])
        writer.writerows(rows)
    return str(out_path)
