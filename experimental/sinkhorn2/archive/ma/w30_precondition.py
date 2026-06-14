"""
MA preconditioning and file preparation.
Generates a clean MA input bundle (uv_cent, ell / ell_resid, initial coeffs, metric, etc.)  for a given hex_layer and fit degree,
Usage: run directly; edit the config block below as needed.
Last tested: 06 Nov 2025
"""
from __future__ import annotations
from pathlib import Path
import time
import csv
from datetime import datetime
import numpy as np
from hhg9 import Registrar
from w20_bernstein_fit_phi import bernstein_eval_vec


def load_phi(layer: int, mode: int, degree: int = 16, tweak: str = 'base'):
    """
    Load Bernstein φ fit (degree, terms, coefficients) for a given (hex_layer, net_mode).
    Expects an NPZ of the form:
      phi_fit_l{hex_layer}_m{net_mode}_n{n_fit}.npz
    with keys:
      - 'n_fit': Bernstein degree (int)
      - 'terms': array of (i,j,k) or (i,j)
      - 'c': coefficient vector
      - 'metric': (g11, g12, g22) metric tensor
      - 'tweak': tweak string
    """
    phi_npz = Path(f"phi_fit_l{layer}_m{mode}_{tweak}_n{degree}.npz")
    if not phi_npz.exists():
        raise FileNotFoundError(f"Missing φ fit file: {phi_npz}")
    repo = np.load(phi_npz, allow_pickle=True)

    # Bernstein degree 'n_fit'
    try:
        n = int(repo['n_fit'])
    except KeyError:
        raise KeyError(f"{phi_npz} missing Bernstein degree 'n_fit'")
    if n != degree:
        raise ValueError(f"Inconsistent Bernstein degree in {phi_npz}: {n} vs {degree}")

    # Bernstein terms
    try:
        terms_arr = np.asarray(repo['terms'], dtype=object)
    except KeyError:
        raise KeyError(f"{phi_npz} missing Bernstein terms 'terms'")

    terms_list = []
    for row in terms_arr:
        t = tuple(int(x) for x in (row.tolist() if hasattr(row, "tolist") else row))
        if len(t) == 3:
            i, j, k = t
        elif len(t) == 2:
            i, j = t
            k = n - i - j
        else:
            raise ValueError(f"Unexpected term length {len(t)} in {phi_npz}: {t}")
        if i + j + k != n:
            raise ValueError(f"Inconsistent term degree in {phi_npz}: {(i,j,k)} vs n={n}")
        terms_list.append((i, j, k))

    # Coefficients
    try:
        c = np.asarray(repo['c'], dtype=float)
    except KeyError:
        raise KeyError(f"{phi_npz} missing Bernstein coefficients 'c'")

    # Metric
    try:
        metric = repo['metric']
        if len(metric) == 3:
            g11, g12, g22 = metric
            _g_inv = np.array([[g11, g12], [g12, g22]])
        else:
            _g_inv = np.array(metric)
    except KeyError:
        raise KeyError(f"{phi_npz} missing metric tensor 3-tuple 'metric'")

    try:
        laplacian_h = repo['laplacian_h']
    except KeyError:
        raise KeyError(f"{phi_npz} missing laplacian 'laplacian_h'")

    print(f"[φ-fit] loaded {phi_npz.name}: K={len(terms_list)} degree={n}")
    return terms_list, c, n, _g_inv, laplacian_h, phi_npz.name


def load_grid(layer: int, mode: int):
    """
    Load precomputed simplex grid for (hex_layer, net_mode) from w10.

    Returns:
      tri_uv     : (N_tri,3,2)
      tri_xy     : (N_tri,3,2)
      uv_cent    : (N_tri,2)
      ell        : (N_tri,)
      components : (3,)
      area_true  : (N_tri,)
      area_mean  : float
    """
    grid_npz = Path(f"grid_l{layer}_m{mode}_simplex.npz")
    if not grid_npz.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_npz}")

    z = np.load(grid_npz, allow_pickle=True)

    depth = int(z['depth'])
    if depth != layer:
        raise ValueError(f"{grid_npz} has depth={depth}, expected hex_layer={layer}")

    # tri_uv = np.asarray(z['tri_uv'], dtype=float)
    # tri_xy = np.asarray(z['tri_xy'], dtype=float)
    # uv_cent = np.asarray(z['uv_cent'], dtype=float)
    # p_cent = np.asarray(z['p_centroids'], dtype=float)  # projected centroids in c_ell domain
    # j_cent = np.asarray(z['j_centroids'], dtype=float)  # AK Jacobian of those centroids in c_ell domain
    # ell = np.asarray(z['ell'], dtype=float)
    # components = np.asarray(z['components'])
    # area_true = np.asarray(z['area_true'], dtype=float)
    # area_mean = float(z['area_mean'])
    v_ell = np.asarray(z['v_ell'], dtype=float)
    uv_vert = np.asarray(z['uv_vert'], dtype=float)

    print(f"[grid] loaded {grid_npz.name}: Points={uv_vert.shape[0]} ell_std={v_ell.std():.3g}")
    return uv_vert, v_ell
    # return tri_uv, tri_xy, uv_cent, p_cent, j_cent, ell, components, area_true, area_mean


def export_ma_bundle(out_npz: Path, terms, degree, uv_cent, ell, ell_resid, c_init, g_inv, laplacian_h, meta: dict):
    np.savez(
        out_npz,
        terms=np.array(terms, dtype=object),
        degree=int(degree),
        uv_cent=np.asarray(uv_cent, dtype=float),
        ell=np.asarray(ell, dtype=float),
        ell_resid=np.asarray(ell_resid, dtype=float),
        c_init=np.asarray(c_init, dtype=float),
        g_inv=g_inv,
        laplacian_h=laplacian_h,
        meta=meta
    )


if __name__ == '__main__':
    # --- Config ---
    fit_deg = 16        # Bernstein degree (must match φ file)
    octant = 0          # set 0...7

    rg = Registrar()
    b_oct = rg.domain('b_oct')
    mode = b_oct.oid_mo[octant]

    for layer in [5]:  # range(7):  # range(5):
        # --- Load precomputed grid from w10 ---
        # tri_uv, tri_xy, uv_cent, p_cent, j_cent, ell, components, area_true, area_mean = load_grid(hex_layer, net_mode)
        vert, ell = load_grid(layer, mode)

        # Baseline authalic from stored ℓ
        rmse_raw = float(np.sqrt(np.mean(ell**2)))
        print(f"Authalic baseline: RMSE={rmse_raw:.6f}  range=[{ell.min():.3f},{ell.max():.3f}]")
        # phi_fit_l5_m0_v0408_n16.npz
        for tweak in ['v0305']:
            terms, c_phi, n_fit, g_inv, laplacian_h, phi_filename = load_phi(layer, mode, fit_deg, tweak)

            b = vert[:, 0]
            c = vert[:, 1]
            a = 1.0 - b - c
            ell_fit = bernstein_eval_vec(a, b, c, n_fit, terms, c_phi)  # <- pass coeffs
            ell_resid = ell - ell_fit
            c_init = c_phi

            # Export MA bundle
            out_npz = Path(f"ma_input_l{layer}_m{mode}_n{n_fit}_{tweak}.npz")
            meta = dict(depth=layer, degree=n_fit, octant=octant, fit_file=phi_filename, timestamp=float(time.time()))
            export_ma_bundle(out_npz, terms, n_fit, vert, ell, ell_resid, c_init, g_inv, laplacian_h, meta)
            print(f"[ma-export] wrote {out_npz.name}: pts={vert.shape}, ell std={np.std(ell):.3g}")

            # Tiny CSV crumb for provenance
            out_csv = Path(f"ma_input_l{layer}_m{mode}_n{n_fit}.csv")
            hdr = ["timestamp","depth","degree","rmse_baseline","ell_min","ell_max","uv_points_count"]
            row = [datetime.now().isoformat(timespec="seconds")+"Z", layer, n_fit,
                   f"{rmse_raw:.6f}", f"{ell.min():.6f}", f"{ell.max():.6f}", int(len(vert))]
            if not out_csv.exists():
                with out_csv.open('w', newline='') as f:
                    csv.writer(f).writerow(hdr)
            with out_csv.open('a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"[ma-export] csv → {out_csv}")
