# ex0063A_prep_ma.py
"""
Prep stage between Bernstein (φ-fit) and MA solver.
Generates a clean MA input bundle (uv_cent, ell) for a given hex_layer and fit degree,
without the iterative α-stepping loop from ex0063A_grid.

Usage: run directly; edit the config block below as needed.
Last tested: 02 Nov 2025
"""
from __future__ import annotations
from pathlib import Path
import time
import csv
from datetime import datetime
import numpy as np

from hhg9 import Registrar, Points
from hhg9.h9.polygon import tri_grid
from hhg9.algorithms.distance import wgs84_area

# --- Optional: reuse bernstein helpers if available ---
try:
    from examples.experiments.bernstein import bernstein_vals_uv_batch
except Exception:
    # Minimal local fallback
    from math import comb
    def bernstein_vals_uv(u, v, n, terms):
        w = 1.0 - u - v
        out = np.empty(len(terms), dtype=float)
        for t, (i, j, k) in enumerate(terms):
            i = int(i); j = int(j); k = int(k)
            out[t] = comb(n, i) * comb(n - i, j) * (u ** i) * (v ** j) * (w ** k)
        return out
    def bernstein_vals_uv_batch(uv, n, terms):
        m = len(uv); k = len(terms)
        b = np.empty((m, k), dtype=float)
        for idx, (u, v) in enumerate(uv):
            b[idx] = bernstein_vals_uv(u, v, n, terms)
        return b

def load_phi(path: Path):
    """Load Bernstein φ fit (degree, terms, coefficients). Accepts keys {n_fit|degree}, {terms}, {c|coeff}."""
    z = np.load(path, allow_pickle=True)
    # degree
    n = None
    if 'n_fit' in z:
        n = int(z['n_fit'])
    elif 'degree' in z:
        n = int(z['degree'])
    # terms
    terms_arr = np.asarray(z['terms'])
    terms_list = [tuple(int(x) for x in row.tolist()) for row in terms_arr]
    if len(terms_list) and len(terms_list[0]) == 2:
        if n is None:
            raise ValueError("Cannot expand (i,j) terms without a known degree.")
        terms_list = [(i, j, int(n - i - j)) for (i, j) in terms_list]
    # coefficients
    if 'c' in z:
        c = np.asarray(z['c'], dtype=float)
    elif 'coeff' in z:
        c = np.asarray(z['coeff'], dtype=float)
    else:
        raise KeyError("NPZ missing coefficients under 'c' or 'coeff'.")
    if n is None:
        n = max(int(i)+int(j)+int(k) for (i,j,k) in terms_list)
    assert all((int(i)+int(j)+int(k)) == n for (i,j,k) in terms_list), "Inconsistent term degrees."
    j = z['J'] if 'J' in z else None
    return terms_list, c, j, n

def make_triangles(reg: Registrar, layer: int, octant_id: int | None):
    """Return (Points for all triangles in b_oct) and flat XY, UV arrays."""
    b_oct = reg.domain('b_oct')
    tg0 = tri_grid(layer, 0).reshape([-1, 2])
    tg1 = tri_grid(layer, 1).reshape([-1, 2])
    tgx = [tg0, tg1]
    if octant_id is None:
        parts = []
        for oct_name in b_oct.signs.keys():
            oid = b_oct.sign_to_id[oct_name]
            mode = b_oct.oid_mo[oid]
            parts.append(Points(tgx[mode].copy(), b_oct, components=oct_name))
        pts = Points.concat(parts)
    else:
        mode = b_oct.oid_mo[octant_id]
        cmp = b_oct.signs_by_id[octant_id]
        pts = Points(tgx[mode], b_oct, components=cmp)
    xy = pts.coords.reshape(-1, 2)
    uv_from_xy = b_oct.x2b[0, 0]
    uv = uv_from_xy(xy).reshape(-1, 2)
    return pts, xy, uv

def eval_authalic(reg: Registrar, pts_xy: np.ndarray, pts_template: Points):
    """Compute areas on WGS84 and return baseline-mean-centred fractional error ℓ, plus absolute areas."""
    b_oct = reg.domain('b_oct')
    pts = Points(pts_xy, b_oct, components=pts_template.components)
    g = reg.project(pts, ['b_oct', 'g_gcd'])
    a = wgs84_area(reg, g, 3).astype(float)
    t_avg = float(np.mean(a))
    ell = (a / t_avg) - 1.0
    return ell, a, t_avg

def export_ma_bundle(out_npz: Path, terms, degree, uv_cent, ell, j, c_init, meta: dict):
    np.savez(
        out_npz,
        terms=np.array(terms, dtype=object),
        degree=int(degree),
        uv_cent=np.asarray(uv_cent, dtype=float),
        ell=np.asarray(ell, dtype=float),
        J=(j if j is not None else np.array([])),
        c_init=np.asarray(c_init, dtype=float),
        meta=meta
    )


if __name__ == '__main__':
    # --- Config ---
    depth = 5           # triangle hex_layer
    fit_deg = 16        # Bernstein degree (must match φ file)
    octant = 0          # set None for all octants, or 0..7
    do_refit = False    # optional small re-centering LS at centroids (no α stepping)
    edge_pin_w = 1e-4   # boundary penalty if refitting
    ridge = 3e-6        # ridge if refitting

    rg = Registrar()
    b_oct = rg.domain('b_oct')

    # Build grid and base geometry
    pts_all, xy0, uv0 = make_triangles(rg, depth, octant)

    # Round-trip sanity
    xy_from_uv = b_oct.x2b[0, 1]
    rt = xy_from_uv(uv0).reshape(-1, 2) - xy0
    rt_rms = float(np.sqrt(np.mean(rt**2)))
    rt_max = float(np.max(np.abs(rt)))
    print(f"[roundtrip] xy→uv→xy  RMS={rt_rms:.3e} max|Δ|={rt_max:.3e}")

    # Baseline authalic (per triangle XY, mean-centred)
    ell_raw, areas, t_avg = eval_authalic(rg, xy0, pts_all)
    rmse_raw = float(np.sqrt(np.mean(ell_raw**2)))
    print(f"Authalic baseline: RMSE={rmse_raw:.6f}  range=[{ell_raw.min():.3f},{ell_raw.max():.3f}]")

    # Load φ-fit (Bernstein)
    phi_npz = Path(f"phi_fit_L{depth}_n{fit_deg}.npz")
    if not phi_npz.exists():
        raise FileNotFoundError(f"Missing φ fit file: {phi_npz}")
    terms, c_phi, j, n_fit = load_phi(phi_npz)
    print(f"[φ-fit] loaded {phi_npz.name}: K={len(terms)} degree={n_fit}")

    # UV triangle centroids → MA collocation points
    uv_tri = uv0.reshape(-1, 3, 2)
    uv_cent = uv_tri.mean(axis=1)

    # Optional: light re-centering LS at centroids only (no quadrature, no α loop)
    c_init = c_phi
    if do_refit:
        b_cent = bernstein_vals_uv_batch(uv_cent, n_fit, terms)   # (ntri, K)
        ell_cyc = ell_raw - float(np.mean(ell_raw))
        jtj = b_cent.T @ b_cent
        k = jtj.shape[0]
        jtj.flat[::k+1] += ridge
        rhs = b_cent.T @ ell_cyc
        # boundary pin ψ≈0
        t = np.linspace(1e-8, 1.0 - 1e-8, 128)
        e = np.vstack([np.stack([t, 0*t], 1), np.stack([0*t, t], 1), np.stack([t, 1.0 - t], 1)])
        be = bernstein_vals_uv_batch(e, n_fit, terms)
        jtj += edge_pin_w * (be.T @ be)
        try:
            c_init = np.linalg.solve(jtj, rhs)
        except np.linalg.LinAlgError:
            c_init = np.linalg.lstsq(jtj, rhs, rcond=None)[0]
        print(f"[refit] ||c_init||={np.linalg.norm(c_init):.3e} (ridge={ridge:g}, edge_pin={edge_pin_w:g})")

    # Export MA bundle
    out_npz = Path(f"ma_alt_input_L{depth}_n{n_fit}.npz")
    meta = dict(depth=depth, degree=n_fit, fit_file=phi_npz.name, timestamp=float(time.time()))
    export_ma_bundle(out_npz, terms, n_fit, uv_cent, ell_raw, j, c_init, meta)
    print(f"[ma-export] wrote {out_npz.name}: uv_cent={uv_cent.shape}, ell std={np.std(ell_raw):.3g}")

    # Tiny CSV crumb for provenance
    out_csv = Path(f"ma_alt_input_L{depth}_n{n_fit}.csv")
    hdr = ["timestamp","depth","degree","rmse_baseline","ell_min","ell_max","uv_centroid_count"]
    row = [datetime.now().isoformat(timespec="seconds")+"Z", depth, n_fit,
           f"{rmse_raw:.6f}", f"{ell_raw.min():.6f}", f"{ell_raw.max():.6f}", int(len(uv_cent))]
    if not out_csv.exists():
        with out_csv.open('w', newline='') as f:
            csv.writer(f).writerow(hdr)
    with out_csv.open('a', newline='') as f:
        csv.writer(f).writerow(row)
    print(f"[ma-export] csv → {out_csv}")