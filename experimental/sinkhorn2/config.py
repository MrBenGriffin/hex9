# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Shared configuration for the authalic-warp training pipeline.

The pipeline trains one ellipsoid at a time. The *index* ellipsoid is WGS84;
other reference ellipsoids are kept distinct so several can be trained
concurrently as separate processes — each stage script accepts ``--ellipsoid``
and tags every checkpoint / output filename with the ellipsoid name, so
concurrent runs never collide.

Lateral-edge handling (the round-trip fix)
------------------------------------------
The two pole-ward octant edges (the "lateral" edges, meeting at the apex) must
map to themselves for the warp to be invertible there. Each octant triangle has
its lateral edges on the line  y - √3·|x| + Ẇ = 0.

* Training  — lateral-edge vertices are pinned to zero displacement.
* Inference — the warp multiplies its displacement by ``edge_blend(d)``, a C1
  function that is exactly 0 on the lateral edges, so edge preservation holds
  regardless of any Clough–Tocher overshoot.
"""
from pathlib import Path
import numpy as np
from hhg9.h9 import H9K

SINKHORN_DIR = Path(__file__).resolve().parent
DATA_DIR     = SINKHORN_DIR.parent.parent / 'hhg9' / 'data'

OCTANT_ID = 0          # canonical octant; mode-1 octants reuse the warp by y-flip

# ── ellipsoids ────────────────────────────────────────────────────────────────
INDEX_ELLIPSOID = 'WGS84'

ELLIPSOIDS = {
    'WGS84B': dict(a=6378137.0, f=1.0 / 298.257223563),
    'WGS84':  dict(a=6378137.0, f=1.0 / 298.257223563),
    'GRS80':  dict(a=6378137.0, f=1.0 / 298.257222101),
    'Sphere': dict(a=6371008.8, f=0.0),
}


def apply_ellipsoid(registrar, name: str):
    """Set the registrar ellipsoid by name; returns the resolved name."""
    if name not in ELLIPSOIDS:
        raise ValueError(f'unknown ellipsoid {name!r}; known: {list(ELLIPSOIDS)}')
    registrar.set_ellipsoid(name=name, **ELLIPSOIDS[name])
    return name


# ── lateral-edge geometry ─────────────────────────────────────────────────────
R3 = H9K.radical.R3                 # √3
W_ = H9K.Ẇ                          # lateral-edge offset:  y - √3|x| + Ẇ = 0

EDGE_TOL  = 1e-5     # |y - √3|x| + Ẇ| below this  →  vertex is on a lateral edge
EDGE_BAND = 0.03     # equilateral units; width of the inference edge-blend ramp


def lateral_edge_mask(xy: np.ndarray) -> np.ndarray:
    """Boolean mask of points lying on either lateral octant edge."""
    return np.abs(xy[:, 1] - R3 * np.abs(xy[:, 0]) + W_) < EDGE_TOL


def lateral_distance(xy: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point to the nearest lateral edge.

    Right edge:  √3·x - y - Ẇ = 0    Left edge: -√3·x - y - Ẇ = 0
    Both lines have norm √(3+1) = 2.
    """
    dr = np.abs(R3 * xy[:, 0] - xy[:, 1] - W_) * 0.5
    dl = np.abs(-R3 * xy[:, 0] - xy[:, 1] - W_) * 0.5
    return np.minimum(dr, dl)


def edge_blend(xy: np.ndarray, band: float = EDGE_BAND) -> np.ndarray:
    """C1 smoothstep in [0,1]: 0 on the lateral edges, 1 at distance ≥ band.

    Multiplying the warp displacement by this guarantees the lateral edges (and
    the apex, where both edges meet) are mapped by the identity.
    """
    t = np.clip(lateral_distance(xy) / band, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
