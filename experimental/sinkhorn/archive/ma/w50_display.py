"""
Part of the H9 project
For a given hex_layer, generate the canonical triangle grid and display on the globe.
Last Tested 6 November 2025 √
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from experimental.algorithms.warp import Warper
from hhg9.h9.polygon import tri_grid
from w99_plot import plot_bwr


def tri_area(tri_uv):
    """
    Compute signed areas of triangles in 2D using the cross product formula.
    tri_uv: (n_tri, 3, 2)
    Returns: (n_tri,) signed areas
    """
    vec0 = tri_uv[:, 1, :] - tri_uv[:, 0, :]
    vec1 = tri_uv[:, 2, :] - tri_uv[:, 0, :]
    return vec0[:, 0] * vec1[:, 1] - vec0[:, 1] * vec1[:, 0]


def count_flips(reg, warper, tri_uv_orig, area_orig, scale, verbosity=0):
    """
    Count how many triangles flip orientation after applying warp at given scale.
    """
    s_oct = reg.domain('s_oct')
    n_tri = tri_uv_orig.shape[0]
    pts = Points(tri_uv_orig.reshape(-1, 2), s_oct)
    warped = warper.warp(pts, scale=scale)
    warped_tri = warped.coords.reshape(n_tri, 3, 2)
    area_warp = tri_area(warped_tri)
    flipped = (area_orig * area_warp) < 0
    n_flipped = int(flipped.sum())
    if verbosity > 0:
        frac = n_flipped / n_tri
        print(f"[count_flips] scale={scale:.3e} flipped={n_flipped}/{n_tri} ({frac:.3%})")
    return n_flipped, flipped


def find_max_safe_scale(reg, ma_file, layer=3, octant_id=0, degree=None,
                        scale_init=1e-4, factor=2.0,
                        tol=1e-6, max_scale=1.0, max_bisect=40, verbosity=1):
    """
    Find maximum warp scale that does not cause any triangle flips using binary search.

    tol is interpreted relative to current hi (so for tiny scales we keep searching).
    """
    # Build grid and areas once
    b_grid = get_grid(reg, layer, octant_id)
    s_grid = reg.project(b_grid, ['b_oct', 's_oct'])
    tri_uv_orig = s_grid.coords.reshape(-1, 3, 2)
    area_orig = tri_area(tri_uv_orig)

    # Load MA coeffs
    ma_in = np.load(ma_file, allow_pickle=True)
    terms = ma_in["terms"]
    c = ma_in["c"]
    deg = degree
    if 'degree' in ma_in:
        deg = int(ma_in['degree'])
    if deg is None:
        deg = 16

    warper = Warper()
    warper.set_values(terms, c, deg)

    # 1) Probe initial scale
    flips_init, _ = count_flips(reg, warper, tri_uv_orig, area_orig,
                                scale_init, verbosity)
    if flips_init > 0:
        # Safe region is somewhere between 0 and scale_init
        lo = 0.0
        hi = scale_init
    else:
        # 2) Grow upward until we hit flips or max_scale
        lo = scale_init
        hi = scale_init
        while True:
            hi *= factor
            if hi > max_scale:
                hi = max_scale
            flips_hi, _ = count_flips(reg, warper, tri_uv_orig, area_orig,
                                      hi, verbosity)
            if flips_hi > 0 or hi >= max_scale:
                break
        if flips_hi == 0:
            if verbosity:
                print(f"[find_max_safe_scale] No flips detected up to max_scale={max_scale:.3e}")
            return max_scale

    if verbosity:
        print(f"[find_max_safe_scale] Bracket found lo={lo:.3e} hi={hi:.3e}")

    # 3) Bisection in [lo, hi] where flips(lo)=0, flips(hi)>0
    for i in range(max_bisect):
        mid = 0.5 * (lo + hi)
        flips_mid, _ = count_flips(reg, warper, tri_uv_orig, area_orig,
                                   mid, verbosity)
        if flips_mid == 0:
            lo = mid
        else:
            hi = mid

        if verbosity:
            print(
                f"[find_max_safe_scale] bisect {i+1}: "
                f"lo={lo:.6e} hi={hi:.6e} mid={mid:.6e} flips_mid={flips_mid}"
            )

        # relative tolerance: stop when bracket is tight *relative* to scale
        if hi > 0 and (hi - lo) <= tol * hi and i >= 3:
            break

    if verbosity:
        print(f"[find_max_safe_scale] max safe scale ≈ {lo:.6e}")
    return lo


def get_grid(reg: Registrar, layer: int = 3, octant_id: int = 0):
    """
    :param reg: h9 Registrar
    :param layer: hex_layer index (0 is coarsest: 72 global triangles)
    :param octant_id: which octant to extract (0–7); default is 1
    :return: b_oct - Points of triangle grid, in clockwise order.
    Because this only needs a single octant - we can choose any single octant.
    Simplex values fit 'net_mode' during xy projection.
    """
    b_oct = reg.domain('b_oct')
    mode = b_oct.oid_mo[octant_id]
    cmp = b_oct.signs_by_id[octant_id]
    t_grid = tri_grid(layer, mode).reshape([-1, 2])  # triangle CW
    return Points(t_grid, b_oct, components=cmp)


_grid_cache = {}


def get_cache(rg: Registrar, layer: int = 3, octant_id: int = 0):
    """Compute and cache geometric data (grid, centers, areas, authalic ell) for a given hex_layer/octant.

    The result is cached in-memory keyed by (hex_layer, octant_id) so repeated calls with different
    MA coefficient files or warp scales don't redo the heavy geometry work.
    """
    key = (layer, octant_id)
    if key in _grid_cache:
        return _grid_cache[key]

    b_oct = rg.domain('b_oct')
    mode = b_oct.oid_mo[octant_id]
    b_grid = get_grid(rg, layer=layer-1, octant_id=octant_id)
    s_grid = rg.project(b_grid, ['b_oct', 's_oct'])
    tri_uv = s_grid.coords.reshape([-1, 3, 2])
    uv_cent = tri_uv.mean(axis=1)

    g_grid = rg.project(b_grid, ['b_oct', 'g_gcd'])
    area_m2 = wgs84_area(rg, g_grid, 3)
    area_m2_mean = area_m2.mean()
    ell = np.log(area_m2 / area_m2_mean)  # authalic log-density ℓ

    data = {
        'net_mode': mode,
        'b_grid': b_grid,
        's_grid': s_grid,
        'uv_cent': uv_cent,
        'ell': ell,
    }
    _grid_cache[key] = data
    return data


def run(rg: Registrar, ma_file, layer: int = 3, octant_id: int = 0, warp_scale: float = 1.0, idx=0):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(hex_layer+1).
    """
    cache = get_cache(rg, layer=layer, octant_id=octant_id)
    mode = cache['net_mode']
    b_grid = cache['b_grid']
    s_grid = cache['s_grid']
    uv_cent = cache['uv_cent']
    ell = cache['ell']

    ma_in = np.load(ma_file, allow_pickle=True)

    # ma_in = np.load(f'ma_psi_l6_m0_n16.npz', allow_pickle=True)
    terms = ma_in["terms"]
    c_orig = ma_in["c"]
    c = c_orig.copy()
    # c = symmetrise_d3(terms, c_orig)
    warper = Warper()
    warper.set_values(terms, c, 16)
    _, grad = warper.eval_and_grad_uv(uv_cent)
    grad_norm = np.linalg.norm(grad, axis=1)
    print(f"[w50] grad_norm: min={grad_norm.min():.3e} med={np.median(grad_norm):.3e} max={grad_norm.max():.3e}")

    # L2 warp_scale = 0.000072
    # warp_scale = 0.00042
    wgd = warper.warp(s_grid, scale=warp_scale)
    wbc = rg.project(wgd, ['s_oct', 'b_oct'])
    wdg = rg.project(wbc, ['b_oct', 'g_gcd'])
    w_area_m2 = wgs84_area(rg, wdg, 3)
    w_area_m2_mean = w_area_m2.mean()
    w_adj = np.abs(w_area_m2 / w_area_m2_mean) + 1e-12
    w_ell = np.log(w_adj)  # authalic log-density ℓ
    delta = w_ell - ell
    corr = np.corrcoef(ell, delta)[0, 1]
    rmse_ell = np.sqrt(np.mean(ell ** 2))
    rmse_w = np.sqrt(np.mean(w_ell ** 2))
    rmse_delta = rmse_w - rmse_ell
    print(f"[w50] hex_layer={layer} net_mode={mode} warp_scale={warp_scale}")
    print(f"  ell:   min={ell.min():+.4f} max={ell.max():+.4f} std={ell.std():.4f}")
    print(f"  w_ell: min={w_ell.min():+.4f} max={w_ell.max():+.4f} std={w_ell.std():.4f}")
    print(f"  RMSE(ell vs 0)   = {rmse_ell:.6f}")
    print(f"  RMSE(w_ell vs 0) = {rmse_w:.6f}")
    print(f"  ΔRMSE            = {rmse_delta:+.6e}")
    print(f"  Corr(ell, Δℓ)    = {corr}")
    if rmse_delta > 0:
        print(f"[w50] WARNING: +ve ΔRMSE. Not plotting")
    else:
        plot_bwr(idx, b_grid, ell, w_ell, delta)
    return grad_norm


if __name__ == '__main__':
    reg = Registrar()
    # ma_file = "ma_psi_l4_sft_lam_300_mu_1e6_l4_m0_n16.npz"
    # ma L4 warp_scale 5.949617767734e-04
    layer = 5
    octant_id = 0

    for i, file in enumerate([
        'ma_psi_xl5_v0305_l5_m0_n16_v0305.npz',
        # 'ma_psi_l5_sft_lambda_95e2_l5_m0_n16_base.npz'  # 0.00123725662231445
        # 'ma_psi_l5_sft_lambda_925e2_l5_m0_n16_vtw010820.npz',   # 0.00118653140068054
        # 'ma_psi_l5_sft_lambda_625e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_650e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_675e2_l5_m0_n16_base.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_800e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_700e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_750e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_875e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_900e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
        # 'ma_psi_l5_sft_lambda_925e2_l5_m0_n16_vtw010820.npz',  # 0.000921085691452027
    ]):
        ma_file = file
        # run(reg, ma_file, hex_layer, octant_id)

        # First, run with zero warp to inspect gradient norms and derive a geometric scale.
        # grad_norm = run(reg, ma_file, hex_layer, octant_id, warp_scale=0.0)
        # gmin, gmed, gmax = grad_norm.min(), np.median(grad_norm), grad_norm.max()
        # print(f"[w50_main] grad_norm stats: min={gmin:.3e} med={gmed:.3e} max={gmax:.3e}")

        # Diagnostic for L5: skip max_safe and probe absolute warp scales
        # using the best L4 warp_scale (~5.9496e-04) as a geometric baseline.
        # NOTE: this intentionally ignores triangle flip safety and should
        # only be used for analysis, not production.
        # warp_scale_geom = 5.949617767734e-04  # L4 sweet-spot warp_scale
        # max_safe_L4 = 5.949617767734e-04
        max_safe_L5 = 0.0025
        # for j, factor in enumerate(np.linspace(0.30, 0.60, 20)):
        for factor in [0.44210]:
            idx = int(layer * 1000000 + i * 100000 + factor * 1000)
            scale = factor * max_safe_L5
            print(f"\n[w50_main] === {file}; factor={factor}; scale={scale:.3e}  ===")
            run(reg, ma_file, layer, octant_id, warp_scale=scale, idx=idx)
