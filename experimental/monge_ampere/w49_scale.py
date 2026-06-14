"""
Part of the H9 project
For a given hex_layer, generate the canonical triangle grid and display on the globe.
Last Tested 6 November 2025 √
"""
import numpy as np
from hhg9 import Registrar, Points
from experimental.algorithms.warp import Warper
from hhg9.h9.polygon import tri_grid
from pathlib import Path



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


def tri_area(tri_uv):
    """
    Compute signed areas of triangles in 2D using the cross product formula.
    tri_uv: (n_tri, 3, 2)
    Returns: (n_tri,) signed areas
    """
    vec0 = tri_uv[:, 1, :] - tri_uv[:, 0, :]
    vec1 = tri_uv[:, 2, :] - tri_uv[:, 0, :]
    return vec0[:, 0] * vec1[:, 1] - vec0[:, 1] * vec1[:, 0]


def count_flips(reg, warper, tri_uv_orig, area_orig, scale, verbosity=0, batch_size=200_000):
    """
    Count how many triangles flip orientation after applying warp at given scale.
    Processes triangles in batches to avoid huge peak memory usage.

    tri_uv_orig: (n_tri, 3, 2)
    area_orig:   (n_tri,)
    """
    s_oct = reg.domain('s_oct')
    n_tri = tri_uv_orig.shape[0]

    n_flipped = 0
    flipped = np.zeros(n_tri, dtype=bool)

    for start in range(0, n_tri, batch_size):
        end = min(start + batch_size, n_tri)

        # (batch_size, 3, 2) → (batch_size*3, 2)
        pts = Points(tri_uv_orig[start:end].reshape(-1, 2), s_oct)
        warper.scale = scale
        warped = warper.warp(pts)
        warped_tri = warped.coords.reshape(-1, 3, 2)

        area_warp = tri_area(warped_tri)
        flipped_batch = (area_orig[start:end] * area_warp) < 0

        flipped[start:end] = flipped_batch
        n_flipped += int(flipped_batch.sum())

    if verbosity > 0:
        frac = n_flipped / n_tri
        print(f"[count_flips] scale={scale:.3e} flipped={n_flipped}/{n_tri} ({frac:.3%})")

    return n_flipped, flipped


def find_max_safe_scale(reg, ma_file, degree=None,
                        tri_uv_orig=None, area_orig=None,
                        scale_init=1e-4, factor=2.0,
                        tol=1e-6, max_scale=1.0, max_bisect=40, verbosity=1):
    """
    Find maximum warp scale that does not cause any triangle flips using binary search.
    tol is interpreted relative to current hi (so for tiny scales we keep searching).
    """

    # Load MA coeffs
    ma_in = np.load(ma_file, allow_pickle=True)
    terms = ma_in["terms"]
    c = ma_in["c"]
    deg = degree
    if 'degree' in ma_in:
        deg = int(ma_in['degree'])
    if deg is None:
        deg = 16

    if verbosity:
        print(f"[find_max_safe_scale] {ma_file}; deg:{deg}; max_scale={max_scale:.3e}")

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


if __name__ == '__main__':
    reg = Registrar()
    # hex_layer at which the MA coefficients are meant to be evaluated
    layer = 5
    octant_id = 0

    # Use a (possibly deeper) safety hex_layer to detect flips more conservatively.
    # You can set safety_layer = hex_layer for equal resolution, or hex_layer+1 for extra safety.
    safety_layer = layer

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"safetygrid_O{octant_id}_S{safety_layer}.npz"

    if cache_file.exists():
        data = np.load(cache_file)
        tri_uv_orig = data["tri_uv_orig"]
        area_orig = data["area_orig"]
        print(f"[cache] Loaded safety grid from {cache_file}")
    else:
        print(f"[cache] Computing safety grid")
        b_grid = get_grid(reg, layer=safety_layer, octant_id=octant_id)
        print(f"[cache] Projecting safety grid")
        s_grid = reg.project(b_grid, ['b_oct', 's_oct'])
        tri_uv_orig = s_grid.coords.reshape(-1, 3, 2)
        print(f"[cache] Calculating safety grid Areas")
        area_orig = tri_area(tri_uv_orig)
        print(f"[cache] Storing safety grid Areas")
        np.savez(cache_file, tri_uv_orig=tri_uv_orig, area_orig=area_orig)
        print(f"[cache] Saved safety grid to {cache_file}")

    for file in [
        'ma_psi_xl5_v0408_l5_m0_n16_v0408.npz'
    ]:
        max_safe = find_max_safe_scale(
            reg,
            file,
            tri_uv_orig=tri_uv_orig,
            area_orig=area_orig,
            max_scale=1.0,
            verbosity=0,
        )
        print(f"[w50_scale] file={file} max_safe_scale≈{max_safe:.15g}")
