#!/usr/bin/env python3
"""
export_warp_header.py — Phase 4 authalic warp grid export.

Samples the H9 authalic warp CloughTocher interpolators (with ghost-row
equator padding) onto a 256×256 regular grid and writes:

    proj-9.8.0/src/projections/h9_warp_data.h

Run from the project root:
    python export_warp_header.py

Requirements: numpy, scipy
"""

import os
import math
import datetime
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH   = os.path.join('/Users/ben/Documents/Projects/PyCharm/hex9/hhg9/data/WGS84_l5_warp_data.npz')
OUT_PATH   = os.path.join('h9_warp_data.h')
# ---------------------------------------------------------------------------
# Load NPZ
# ---------------------------------------------------------------------------
print(f"Loading {os.path.normpath(NPZ_PATH)} ...")
repo = np.load(NPZ_PATH, allow_pickle=True)
src  = repo['source_pts']    # (N_pts, 2)  mode-0 b_raw coordinates
dst  = repo['target_pts']    # (N_pts, 2)  warped coordinates
# mae  = float(repo['mae'])
# layer = int(repo['layer'])
print(f"  {len(src)} source points, layer=5")
diff = dst - src

# ---------------------------------------------------------------------------
# Ghost-row padding (equator stabilisation) — mirrors AuthalicWarp.__init__
# ---------------------------------------------------------------------------
Y_EQ = math.sqrt(6.0) / 6.0           # equator y in mode-0 b_raw space
eq_band_mask = (Y_EQ - src[:, 1]) < 0.05

if np.any(eq_band_mask):
    ghost_src  = src[eq_band_mask].copy()
    ghost_src[:, 1] = 2.0 * Y_EQ - ghost_src[:, 1]   # mirror across y=Y_EQ

    ghost_diff       = diff[eq_band_mask].copy()
    ghost_diff[:, 1] *= -1.0                            # negate y-displacement

    padded_src  = np.vstack([src,  ghost_src])
    padded_diff = np.vstack([diff, ghost_diff])
    print(f"  Ghost padding: {np.sum(eq_band_mask)} equator points → "
          f"{len(padded_src)} total")
else:
    padded_src  = src
    padded_diff = diff
    print("  (no ghost-row padding applied)")

# ---------------------------------------------------------------------------
# Build CT interpolators with nearest-neighbour fallback
# ---------------------------------------------------------------------------
print("Building CloughTocher2D interpolators ...")
ct_dx = CloughTocher2DInterpolator(padded_src, padded_diff[:, 0])
ct_dy = CloughTocher2DInterpolator(padded_src, padded_diff[:, 1])
nn_dx = NearestNDInterpolator(src, diff[:, 0])
nn_dy = NearestNDInterpolator(src, diff[:, 1])

def fwd_dx(xy):
    d = ct_dx(xy)
    m = np.isnan(d) & np.isfinite(xy).all(axis=1)
    if np.any(m):
        d[m] = nn_dx(xy[m])
    return d

def fwd_dy(xy):
    d = ct_dy(xy)
    m = np.isnan(d) & np.isfinite(xy).all(axis=1)
    if np.any(m):
        d[m] = nn_dy(xy[m])
    return d

# ---------------------------------------------------------------------------
# Grid bounds — mode-0 b_raw triangle bounding box
#
#   Vertices (mode-0, down triangle):
#     pole   (u=1): (  0,  -2H/3) = (0,       -sqrt(6)/3)  ← Y_MIN
#     v=1  vertex : (+W/2, +H/3)  = (+sqrt(2)/2, +sqrt(6)/6) ← X_MAX, Y_MAX
#     w=1  vertex : (-W/2, +H/3)  = (-sqrt(2)/2, +sqrt(6)/6) ← X_MIN, Y_MAX
#
#   where W = sqrt(2),  H = sqrt(6)/2  (from h9_boct beam-search space)
#
# Layout: row-major, iy=0 → y=Y_MIN (pole), iy=N-1 → y=Y_MAX (equator).
# ---------------------------------------------------------------------------
N     = 256
X_MIN = -math.sqrt(2.0) / 2.0    # = -W/2
X_MAX =  math.sqrt(2.0) / 2.0    # = +W/2
Y_MIN = -math.sqrt(6.0) / 3.0    # pole (bottom of mode-0 triangle)
Y_MAX =  math.sqrt(6.0) / 6.0    # equator (top of mode-0 triangle)

print(f"\nSampling {N}×{N} grid:")
print(f"  x ∈ [{X_MIN:.10f}, {X_MAX:.10f}]")
print(f"  y ∈ [{Y_MIN:.10f}, {Y_MAX:.10f}]")

xs = np.linspace(X_MIN, X_MAX, N)
ys = np.linspace(Y_MIN, Y_MAX, N)
gx, gy = np.meshgrid(xs, ys)          # shape (N, N); row 0 = y=Y_MIN
pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

print(f"  Evaluating dx ...")
grid_dx = fwd_dx(pts).reshape(N, N).astype(np.float32)
print(f"  Evaluating dy ...")
grid_dy = fwd_dy(pts).reshape(N, N).astype(np.float32)

print(f"\n  dx range: [{grid_dx.min():.7f}, {grid_dx.max():.7f}]")
print(f"  dy range: [{grid_dy.min():.7f}, {grid_dy.max():.7f}]")
print(f"  NaN count in dx: {np.isnan(grid_dx).sum()}")
print(f"  NaN count in dy: {np.isnan(grid_dy).sum()}")

# ---------------------------------------------------------------------------
# Spot-check: compare grid-lookup result to direct CT evaluation
# at a handful of source points.
# ---------------------------------------------------------------------------
def grid_lookup(grid, x, y):
    """Bilinear interpolation on the 256×256 grid (clamp at boundaries)."""
    xi = (x - X_MIN) / (X_MAX - X_MIN) * (N - 1)
    yi = (y - Y_MIN) / (Y_MAX - Y_MIN) * (N - 1)
    xi = float(np.clip(xi, 0, N - 1))
    yi = float(np.clip(yi, 0, N - 1))
    ix0, iy0 = int(xi), int(yi)
    ix1 = min(ix0 + 1, N - 1)
    iy1 = min(iy0 + 1, N - 1)
    tx = xi - ix0
    ty = yi - iy0
    v00 = grid[iy0, ix0]
    v10 = grid[iy0, ix1]
    v01 = grid[iy1, ix0]
    v11 = grid[iy1, ix1]
    return (v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) +
            v01 * (1 - tx) * ty       + v11 * tx * ty)

print("\nSpot-checks (CT direct vs grid bilinear):")
rng = np.random.default_rng(42)
test_idx = rng.choice(len(src), size=10, replace=False)
max_err = 0.0
for i in test_idx:
    px, py = src[i]
    ct_vx = float(ct_dx(np.array([[px, py]]))[0])
    ct_vy = float(ct_dy(np.array([[px, py]]))[0])
    g_vx  = grid_lookup(grid_dx, px, py)
    g_vy  = grid_lookup(grid_dy, px, py)
    ex, ey = abs(ct_vx - g_vx), abs(ct_vy - g_vy)
    max_err = max(max_err, ex, ey)
    print(f"  ({px:+.4f}, {py:+.4f}): CT=({ct_vx:+.6f},{ct_vy:+.6f})  "
          f"grid=({g_vx:+.6f},{g_vy:+.6f})  err=({ex:.2e},{ey:.2e})")
print(f"  Max error: {max_err:.3e}  (b_raw units; 1 unit ≈ 2.5e7 m)")

# ---------------------------------------------------------------------------
# Write C header
# ---------------------------------------------------------------------------
def format_float_array(arr, name, n):
    """Return a C float[N*N] static const array definition."""
    flat = arr.flatten()
    rows = []
    def fmt(v):
        s = f"{v:.9g}"
        # Ensure the literal has a decimal point or exponent so 'f' suffix is valid C
        if '.' not in s and 'e' not in s and 'n' not in s:
            s += '.0'
        return s + 'f'
    for i in range(0, len(flat), 8):
        chunk = flat[i:i + 8]
        rows.append("    " + ", ".join(fmt(v) for v in chunk))
    body = ",\n".join(rows)
    return f"static const float {name}[{n * n}] = {{\n{body}\n}};"

print(f"\nWriting {os.path.normpath(OUT_PATH)} ...")

header_text = f"""\
/* h9_warp_data.h — AUTO-GENERATED by export_warp_header.py — DO NOT EDIT
 *
 * Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
 * Source:    hhg9/data/WGS84_l5_warp_data.npz.npz  (layer {5})
 * Points:    {len(src)} source/target pairs (Sinkhorn optimal transport)
 *
 * Authalic warp displacement grid for the H9 octahedral projection (h9_boct).
 * Built from CloughTocher2DInterpolator with ghost-row equator padding.
 *
 * ── Coordinate space ────────────────────────────────────────────────────────
 *   Mode-0 b_raw triangle bounding box  (all lookups are done in mode-0 space;
 *   mode-1 octants use a y-flip trick: ry0 = -ry before lookup, ry = -ry0 after).
 *
 *   x ∈ [H9_WARP_XMIN, H9_WARP_XMAX]  =  [-√2/2, +√2/2]   (≈ ±0.7071)
 *   y ∈ [H9_WARP_YMIN, H9_WARP_YMAX]  =  [-√6/3, +√6/6]   (≈ -0.8165 … +0.4082)
 *   Pole (Z-vertex, mode-0) at (0, Y_MIN);  equator at y = Y_MAX.
 *
 * ── Grid layout ─────────────────────────────────────────────────────────────
 *   H9_WARP_N × H9_WARP_N  row-major float32 arrays.
 *   Row iy=0 → y = Y_MIN (pole);  row iy=N-1 → y = Y_MAX (equator).
 *   Column ix=0 → x = X_MIN;      column ix=N-1 → x = X_MAX.
 *   Index: i = iy * H9_WARP_N + ix
 *
 * ── Forward warp (b_raw oriented → b_raw warped) ───────────────────────────
 *   mode-0:  wx = rx + bilinear(dx, rx,  ry)   wy = ry + bilinear(dy, rx,  ry)
 *   mode-1:  ry0 = -ry
 *            wx = rx + bilinear(dx, rx, ry0)   wy = -(ry0 + bilinear(dy, rx, ry0))
 *
 * ── Inverse warp (Newton-Raphson, 25 iterations, |err|² < 1e-28) ────────────
 *   Unified: mode_sign = (oct_mode==0) ? 1.0 : -1.0
 *            wy0 = wy * mode_sign     (flip target to mode-0 space)
 *            Solve (cx + dx(cx,cy), cy + dy(cx,cy)) = (wx, wy0) for (cx,cy)
 *            rx = cx;   ry = cy * mode_sign
 *
 * ── Displacement statistics ─────────────────────────────────────────────────
 *   dx ∈ [{grid_dx.min():.7f}, {grid_dx.max():.7f}]  (b_raw units)
 *   dy ∈ [{grid_dy.min():.7f}, {grid_dy.max():.7f}]  (b_raw units)
 *   1 b_raw unit ≈ 2.5 × 10⁷ m  →  max displacement ≈ {grid_dx.max()*2.5e7/1000:.0f} km
 *****************************************************************************/
#ifndef H9_WARP_DATA_H
#define H9_WARP_DATA_H

#define H9_WARP_N     {N}
#define H9_WARP_XMIN  ({X_MIN:.17g})
#define H9_WARP_XMAX  ({X_MAX:.17g})
#define H9_WARP_YMIN  ({Y_MIN:.17g})
#define H9_WARP_YMAX  ({Y_MAX:.17g})

/* h9_warp_dx[iy*N+ix] = x-displacement at grid point (ix, iy) */
{format_float_array(grid_dx, 'h9_warp_dx', N)}

/* h9_warp_dy[iy*N+ix] = y-displacement at grid point (ix, iy) */
{format_float_array(grid_dy, 'h9_warp_dy', N)}

#endif /* H9_WARP_DATA_H */
"""

with open(OUT_PATH, 'w') as f:
    f.write(header_text)

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Done.  File size: {size_kb:.1f} KB")
print(f"\nNext step: rebuild PROJ and run test_boct_roundtrip.py --with-warp")
