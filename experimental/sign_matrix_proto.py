"""
Prototype: derive b_oct<->c_oct projection matrices analytically from octant sign 3-tuples.

Hypothesis:
  Given (sx, sy, sz) ∈ {±1}³, the OctantBary matrix and orient are fully determined:
    - (sx, sy) quadrant → rotation count r ∈ {0,1,2,3}
    - sz              → hemisphere (north base matrix vs south; z-column sign)
    - sx·sy·sz sign   → orient angle (θ=2π/3 or θ=5π/3)

  This yields a precomputed (8, 3, 3) combined-matrix tensor C[oid] = (M.T @ orient).T,
  which vectorises the per-octant dispatch in _project_composites to a single einsum.

  The warp (AuthalicWarp) is already octant-agnostic (one CT interpolator).  The only
  per-octant difference is the net_mode-y-flip, which vectorises trivially via np.where.

Run standalone to verify and benchmark:
    python experimental/sign_matrix_proto.py
"""
import time
import numpy as np

# --- The two primitives (identical to OctahedralBarycentric.__init__) ---
_TRANS    = np.array([[-1., 0.], [1., -2.], [1., 1.]])
_R90      = np.array([[0., 1.], [-1., 0.]])
_MIRROR   = np.array([[0., 1.], [1., 0.]])   # mirror_y_neg_x
_SCALE    = np.sqrt([2., 6., 3.])[:, None]
_QUAD_ROT = {(1, 1): 0, (-1, 1): 1, (-1, -1): 2, (1, -1): 3}
_Z_OFF    = 1.0 / np.sqrt(3)


def matrix_from_signs(sx: int, sy: int, sz: int) -> np.ndarray:
    """Return the (3,3) OctantBary projection matrix for octant (sx, sy, sz)."""
    r    = _QUAD_ROT[(sx, sy)]
    base = _TRANS if sz > 0 else _TRANS @ _MIRROR
    rot  = np.linalg.matrix_power(_R90, r)   # (2,2)
    xy   = base @ rot                         # (3,2)
    z    = np.full((3, 1), float(sz))
    return np.column_stack([xy, z]) / _SCALE


def orient_from_signs(sx: int, sy: int, sz: int) -> np.ndarray:
    """Return the (3,3) orient rotation matrix for octant (sx, sy, sz)."""
    theta = (2 if sx * sy * sz > 0 else 5) * np.pi / 3.
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.], [st, ct, 0.], [0., 0., 1.]])


def build_combined_tensor() -> np.ndarray:
    """
    Precompute the (8, 3, 3) tensor C where C[oid] = (M.T @ orient).T
    i.e. the matrix such that:  xyz_3d = xy_with_z @ C[oid]   (b_oct -> c_oct)
    """
    from hhg9.h9.constants import H9O
    C = np.empty((8, 3, 3), dtype=np.float64)
    for oid in range(8):
        sx, sy, sz = H9O.oid_cmp[oid]
        M = matrix_from_signs(sx, sy, sz)
        O = orient_from_signs(sx, sy, sz)
        C[oid] = (M.T @ O).T
    return C


# --- Vectorised warp (octant-agnostic CT, only net_mode-y-flip varies per octant) ---

def vec_warp_do(xy: np.ndarray, oids: np.ndarray, warp, oid_mo: np.ndarray) -> np.ndarray:
    """Vectorised AuthalicWarp.do across mixed octants."""
    mode_factors = np.where(oid_mo[oids] == 0, 1.0, -1.0)   # (N,)
    xy_v = xy.copy()
    xy_v[:, 1] *= mode_factors
    dx = warp.fwd_dx(xy_v)
    dy = warp.fwd_dy(xy_v)
    res = xy_v + np.stack([dx, dy], axis=1)
    res[:, 1] *= mode_factors
    return res


def vec_warp_undo(xy: np.ndarray, oids: np.ndarray, warp, oid_mo: np.ndarray,
                  iterations: int = 25) -> np.ndarray:
    """Vectorised AuthalicWarp.undo across mixed octants (Newton-Raphson)."""
    mode_factors = np.where(oid_mo[oids] == 0, 1.0, -1.0)
    target = xy.copy()
    target[:, 1] *= mode_factors

    guess = warp.inv_linear(target)
    nan_mask = np.isnan(guess[:, 0])
    if np.any(nan_mask):
        guess[nan_mask] = warp.inv_nearest(target[nan_mask])

    curr = guess.copy()
    for _ in range(iterations):
        dx = warp.fwd_dx(curr)
        dy = warp.fwd_dy(curr)
        bad = np.isnan(dx)
        if np.any(bad):
            curr[bad] = warp.inv_nearest(target[bad])
            dx = warp.fwd_dx(curr)
            dy = warp.fwd_dy(curr)
        curr -= curr + np.stack([dx, dy], axis=1) - target
        nf = ~np.isfinite(curr).all(axis=1)
        if np.any(nf):
            curr[nf] = warp.inv_nearest(target[nf])

    curr[:, 1] *= mode_factors
    return curr


# --- Vectorised b_oct ↔ c_oct ---

def vec_boct_to_coct(xy: np.ndarray, oids: np.ndarray, C: np.ndarray,
                     warp=None, oid_mo: np.ndarray = None) -> np.ndarray:
    """
    Vectorised b_oct -> c_oct for a mixed-octant batch.
    Matches OctantBary.backward exactly (including warp if provided).
    """
    if warp is not None:
        xy = vec_warp_do(xy, oids, warp, oid_mo)
    xyz = np.empty((len(xy), 3), dtype=np.float64)
    xyz[:, :2] = xy
    xyz[:, 2]  = _Z_OFF
    return np.einsum('ni,nij->nj', xyz, C[oids])


def vec_coct_to_boct(xyz: np.ndarray, oids: np.ndarray, C: np.ndarray,
                     warp=None, oid_mo: np.ndarray = None) -> np.ndarray:
    """
    Vectorised c_oct -> b_oct for a mixed-octant batch.
    Matches OctantBary.forward exactly (including warp if provided).
    """
    # C[oid].T is the inverse (orthogonal matrix)
    xy3 = np.einsum('ni,nji->nj', xyz, C[oids])
    xy  = xy3[:, :2]
    if warp is not None:
        xy = vec_warp_undo(xy, oids, warp, oid_mo)
    return xy


# --- Verification ---

def verify(C: np.ndarray):
    from hhg9 import Registrar
    from hhg9.h9.constants import H9O
    reg   = Registrar()
    b_oct = reg.domain('b_oct')

    all_ok = True
    print(f"{'Octant':<6}  {'M ok':>6}  {'O ok':>6}  {'C ok':>6}")
    print("-" * 32)
    for oid in range(8):
        sx, sy, sz = H9O.oid_cmp[oid]
        face = H9O.oid_str[oid]
        proj = b_oct.projs[face]
        m_ok = np.allclose(proj.matrix,                      matrix_from_signs(sx, sy, sz), atol=1e-12)
        o_ok = np.allclose(proj.orient,                      orient_from_signs(sx, sy, sz), atol=1e-12)
        c_ok = np.allclose((proj.matrix.T @ proj.orient).T,  C[oid],                        atol=1e-12)
        all_ok &= m_ok and o_ok and c_ok
        print(f"{face:<6}  {str(m_ok):>6}  {str(o_ok):>6}  {str(c_ok):>6}")
    print()
    print("All match." if all_ok else "MISMATCH — check derivation.")
    return all_ok


def roundtrip_check(C: np.ndarray):
    from hhg9 import Registrar, Points
    from hhg9.h9.constants import H9O
    reg   = Registrar()
    b_oct = reg.domain('b_oct')
    c_oct = reg.domain('c_oct')
    warp  = b_oct.warp
    oid_mo = H9O.oid_mo

    rng  = np.random.default_rng(0)
    N    = 1_000
    oids = rng.integers(0, 8, N, dtype=np.uint8)
    xy   = rng.uniform(-0.15, 0.15, (N, 2))

    # Vectorised path (with warp)
    c3d_vec = vec_boct_to_coct(xy, oids, C, warp, oid_mo)
    xy_rt_v = vec_coct_to_boct(c3d_vec, oids, C, warp, oid_mo)

    # Reference: registered per-octant dispatch
    pts_b  = Points(xy, b_oct, oids)
    pts_c  = reg.project(pts_b, [b_oct, c_oct])
    pts_b2 = reg.project(pts_c, [c_oct, b_oct])

    err_c = np.abs(c3d_vec - pts_c.coords).max()
    err_b = np.abs(xy_rt_v - pts_b2.coords).max()
    print(f"\nRound-trip vs. registered projection (N={N}, mixed octants, with warp):")
    print(f"  c_oct error: {err_c:.2e}  ({'OK' if err_c < 1e-10 else 'FAIL'})")
    print(f"  b_oct error: {err_b:.2e}  ({'OK' if err_b < 1e-10 else 'FAIL'})")


def benchmark(C: np.ndarray, N: int = 100_000, reps: int = 10):
    from hhg9 import Registrar, Points
    from hhg9.h9.constants import H9O
    reg   = Registrar()
    b_oct = reg.domain('b_oct')
    c_oct = reg.domain('c_oct')
    warp  = b_oct.warp
    oid_mo = H9O.oid_mo

    rng  = np.random.default_rng(1)
    oids = rng.integers(0, 8, N, dtype=np.uint8)
    xy   = rng.uniform(-0.15, 0.15, (N, 2))
    pts_b = Points(xy, b_oct, oids)

    # Warm up
    _ = vec_boct_to_coct(xy, oids, C, warp, oid_mo)
    _ = reg.project(pts_b, [b_oct, c_oct])

    t0 = time.perf_counter()
    for _ in range(reps):
        vec_boct_to_coct(xy, oids, C, warp, oid_mo)
    t_vec = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        reg.project(pts_b, [b_oct, c_oct])
    t_ref = (time.perf_counter() - t0) / reps * 1000

    print(f"\nBenchmark  b_oct -> c_oct  (N={N:,}, {reps} reps, with warp):")
    print(f"  Vectorised (einsum + vec warp):  {t_vec:6.2f} ms")
    print(f"  Current dispatch:                {t_ref:6.2f} ms")
    print(f"  Speedup:                         {t_ref / t_vec:.1f}×")

    # Also benchmark matrix-only (warp dominates; see how much matrix costs)
    t0 = time.perf_counter()
    for _ in range(reps):
        vec_boct_to_coct(xy, oids, C, warp=None, oid_mo=None)
    t_no_warp = (time.perf_counter() - t0) / reps * 1000
    print(f"  Vectorised (matrix only, no warp): {t_no_warp:.2f} ms  (warp cost: {t_vec - t_no_warp:.2f} ms)")


if __name__ == '__main__':
    C = build_combined_tensor()
    print("=== Matrix derivation from signs ===")
    ok = verify(C)
    if ok:
        print("\n=== Correctness vs. registered projection ===")
        roundtrip_check(C)
        print("\n=== Benchmark ===")
        benchmark(C)
