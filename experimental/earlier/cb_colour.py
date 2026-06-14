#!/usr/bin/env python3
"""
cb_colour.py
Self-contained demo for a colorblind-friendly colormap (plasmagma_cb)
and shared robust normalization across RAW vs FITTED log-density plots.

Usage:
  python cb_colour.py                 # demo with synthetic data
  python cb_colour.py --npz data.npz  # expects uv, r_raw, r_fit_samples
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm
from pathlib import Path

import matplotlib as mpl


# ---------- colormap ----------
def plasmagma_cb():
    """Lower quartile = inverted top half of magma, upper = plasma."""
    plasma = mpl.colormaps.get_cmap("plasma")
    magma  = mpl.colormaps.get_cmap("magma")
    t_low  = np.linspace(1.0, 0.5, 64)     # invert top half of magma
    seg_low = magma(t_low)
    t_hi   = np.linspace(0.0, 1.0, 192)    # full plasma
    seg_hi = plasma(t_hi)
    seg = np.vstack([seg_low, seg_hi])
    return LinearSegmentedColormap.from_list("plasmagma_cb", seg)


# ---------- helpers ----------
def in_simplex_mask(u, v):
    return (u >= 0) & (v >= 0) & (u + v <= 1)

def random_uv(n):
    # sample uniformly in simplex via sorted exponentials
    a = np.random.rand(n)
    b = np.random.rand(n)
    s = a + b
    u = a / (s + 1e-12)
    v = b / (s + 1e-12)
    # fold down if u+v>1
    m = (u + v) > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return u, v

def make_synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    u, v = random_uv(n)
    # smooth signed field with a mild “bump” and edge taper
    r_raw = (0.22*np.cos(4*np.pi*u) * np.sin(3*np.pi*v)
             + 0.08*np.exp(-((u-0.33)**2 + (v-0.33)**2)/0.01)
             - 0.06*(u+v-0.66))
    # a fitted version that misses some low-end structure
    r_fit = r_raw - 0.03*np.exp(-((u-0.10)**2 + (v-0.70)**2)/0.006) + 0.01*rng.normal(size=n)
    uv = np.column_stack([u, v])
    return uv, r_raw, r_fit

def robust_shared_norm(a, b, pct=(1, 99)):
    lo = np.percentile(np.concatenate([a, b]), pct[0])
    hi = np.percentile(np.concatenate([a, b]), pct[1])
    v = max(abs(lo), abs(hi))
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=+v)

def robust_centered_norm(x, pct=99):
    q = np.percentile(np.abs(x), pct)
    return TwoSlopeNorm(vmin=-q, vcenter=0.0, vmax=+q)

def triangle_outline():
    return np.array([[0,0],[1,0],[0,1],[0,0]], dtype=float)


# ---------- helpers ----------
def safe_triangulation(uv):
    """
    Build a robust triangulation.
    1) Try Matplotlib's default (Qhull).
    2) Fallback to SciPy's Delaunay with 'QJ' joggle (if SciPy available).
    3) Final fallback: tiny jitter before calling Matplotlib again.
    """
    x = uv[:, 0]
    y = uv[:, 1]
    if uv.shape[0] < 3:
        raise ValueError(f"Need at least 3 points to triangulate, got {uv.shape[0]}")
    # 1) Try default
    try:
        return mtri.Triangulation(x, y)
    except RuntimeError:
        pass

    # 2) SciPy fallback with QJ (if available)
    try:
        from scipy.spatial import Delaunay  # local import; SciPy may be optional
        de = Delaunay(np.column_stack([x, y]), qhull_options="QJ")
        return mtri.Triangulation(x, y, triangles=de.simplices)
    except Exception:
        pass

    # 3) Tiny jitter fallback
    rng = np.random.default_rng(0)
    eps = np.finfo(float).eps * 1e3
    jitter = rng.standard_normal(uv.shape) * eps
    xj = x + jitter[:, 0]
    yj = y + jitter[:, 1]
    return mtri.Triangulation(xj, yj)


def interior_mask(uv, eps=5e-6):
    u = uv[:, 0]
    v = uv[:, 1]
    return (u > eps) & (v > eps) & (u + v < 1.0 - eps)


# ---------- plotting ----------
def plot_all(uv, r_raw, r_fit_samples, out_prefix="demo", cmap=None):
    if cmap is None:
        cmap = plasmagma_cb()

    # keep points strictly inside the simplex to avoid collinearity on edges
    kept = None
    for eps_tri in (5e-6, 5e-8, 0.0):  # progressively relax if we lose too many points
        m_int = interior_mask(uv, eps=eps_tri)
        if np.count_nonzero(m_int) >= 3:
            kept = m_int
            break
    if kept is None:
        # last resort: don't drop anything; if still <3, raise a clear error
        kept = np.ones(len(uv), dtype=bool)
    uv = uv[kept]
    r_raw = r_raw[kept]
    r_fit_samples = r_fit_samples[kept]
    if uv.shape[0] < 3:
        raise ValueError(
            f"Not enough points to triangulate after masking (got {uv.shape[0]}). "
            "Try providing more samples or relaxing the interior mask."
        )

    tri_samp = safe_triangulation(uv)
    resid = r_raw - r_fit_samples

    norm_shared = robust_shared_norm(r_raw, r_fit_samples, pct=(1, 99))
    norm_resid  = robust_centered_norm(resid, pct=99)
    tri_uv = triangle_outline()

    # raw
    fig1 = plt.figure(figsize=(9,9))
    tcf1 = plt.tricontourf(tri_samp, r_raw, levels=30, cmap=cmap, norm=norm_shared)
    plt.plot(tri_uv[:,0], tri_uv[:,1], color="k", lw=0.8)
    plt.gca().set_aspect('equal','box')
    plt.title("Raw log-density ℓ")
    cb1 = plt.colorbar(tcf1); cb1.set_label("ℓ")
    fig1.savefig(f"{out_prefix}_raw.png", dpi=160)

    # fitted (at sample locations to compare apples-to-apples)
    fig2 = plt.figure(figsize=(9,9))
    tcf2 = plt.tricontourf(tri_samp, r_fit_samples, levels=30, cmap=cmap, norm=norm_shared)
    plt.plot(tri_uv[:,0], tri_uv[:,1], color="k", lw=0.8)
    plt.gca().set_aspect('equal','box')
    plt.title("Fitted log-density ℓ̂ (at centroids)")
    cb2 = plt.colorbar(tcf2); cb2.set_label("ℓ̂")
    fig2.savefig(f"{out_prefix}_fit_samples.png", dpi=160)

    # residual
    fig3 = plt.figure(figsize=(9,9))
    tcf3 = plt.tricontourf(tri_samp, resid, levels=30, cmap=cmap, norm=norm_resid)
    plt.plot(tri_uv[:,0], tri_uv[:,1], color="k", lw=0.8)
    plt.gca().set_aspect('equal','box')
    plt.title("Residual ℓ − ℓ̂")
    cb3 = plt.colorbar(tcf3); cb3.set_label("Residual")
    fig3.savefig(f"{out_prefix}_resid.png", dpi=160)

    # print quick ranges (same units/scale)
    def rng(x):
        return float(np.min(x)), float(np.median(x)), float(np.max(x))
    lo1, med1, hi1 = rng(r_raw)
    lo2, med2, hi2 = rng(r_fit_samples)
    lo3, med3, hi3 = rng(resid)
    print(f"raw range:     [{lo1:.3f}, {med1:.3f}, {hi1:.3f}]")
    print(f"fitted range:  [{lo2:.3f}, {med2:.3f}, {hi2:.3f}] (same norm as raw)")
    print(f"resid range:   [{lo3:.3f}, {med3:.3f}, {hi3:.3f}]")

    plt.show()


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default=None,
                    help="optional npz with arrays: uv (hex_layer,2), r_raw (hex_layer,), r_fit_samples (hex_layer,)")
    ap.add_argument("--out", type=str, default=None, help="prefix for saved figs")
    args = ap.parse_args()

    if args.npz:
        path = Path(args.npz)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        uv = np.asarray(data["uv"], float)
        r_raw = np.asarray(data["r_raw"], float)
        r_fit_samples = np.asarray(data["r_fit_samples"], float)
        if uv.shape[1] != 2 or r_raw.shape != r_fit_samples.shape or r_raw.ndim != 1:
            raise ValueError("npz must contain uv (hex_layer,2), r_raw (hex_layer,), r_fit_samples (hex_layer,)")
        m = in_simplex_mask(uv[:,0], uv[:,1])
        uv, r_raw, r_fit_samples = uv[m], r_raw[m], r_fit_samples[m]
        out_prefix = args.out or path.stem
    else:
        uv, r_raw, r_fit_samples = make_synthetic(n=25000, seed=42)
        out_prefix = args.out or "demo"

    plot_all(uv, r_raw, r_fit_samples, out_prefix=out_prefix, cmap=plasmagma_cb())


if __name__ == "__main__":
    main()
    plt.show()
