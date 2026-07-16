# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Option-B fundamental-domain gradient refinement (true 1/6 solve).

Trains the spherical authalic warp on the fundamental Moebius triangle of the
D3 face group — corners C (pole face corner), M (midpoint of the adjacent
lateral edge), G (face centroid) — at any layer (designed for L6).

Key facts this design rests on (verified 2026-07-14, see scoping brief):
  * The unwarped AK area-distortion field under the Sphere registrar is
    D3-equivariant to ~1e-12 and the lattice maps onto itself exactly.
  * Face-median mirror lines never bisect lattice cells: every triangle is
    either generic (orbit size 6) or self-symmetric across one median
    (orbit size 3). The 1/6 loss is therefore an exactly orbit-weighted sum;
    no half-cells exist.

Pipeline (per Ben's addendum: warm-started gradient refinement, NO transport):
  1. Build full-face lattice (tri_mesh), select representative triangles by
     centroid membership of the closed wedge; orbit weights 6/3.
  2. Mean-slice warm start from the deployed Sphere L5 field: CT-evaluate the
     L5 displacement at the six group images of each vertex, fold back with
     inverse transforms, average; then enforce BCs exactly (tangential on the
     three wedge edges, zero at the three corners).
  3. Stage-3-style Adam loop: orbit-weighted Sum w.log^2(r) loss, gradient
     fold-back from ghosts (grad[parent] += T @ grad[ghost], T orthogonal),
     tangent-projected gradients for on-edge vertices (projected BEFORE moment
     accumulation so Adam moments live in the tangent space), exact snap after
     every step, ghost re-slaving each step, lr warmup, dual-patience cooling,
     strict-best saving. All conventions mirror stage3_gradient_l5.py.

Boundary semantics:
  wedge edge C-G  : on face median x=0            -> displacement dx=0 (slide)
  wedge edge G-M  : on face median through A and M -> displacement along u2
  wedge edge C-M  : half lateral (face) edge       -> displacement along u3
  corners C, G, M : pinned to zero displacement
Under the D3 unfold these give: equator dy=0, lateral edges slide (never pin),
face corners fixed — the F6 loader invariants, uniformly on all edges.

Usage:
  python stage3b_fund_l6.py --ellipsoid Sphere --layer 6 \
      --warm-start ../../hhg9/data/Sphere_l5_warp_data.npz \
      --output_dir output/stage3b
"""
from pathlib import Path
import numpy as np

from hhg9 import Registrar, Points
from hhg9.h9 import H9K
from experimental.sinkhorn import config
from experimental.sinkhorn.fast_area import fast_wgs84_area as wgs84_area_coarse

# ── CONFIGURATION (stage3 conventions) ───────────────────────────────────────
ITERATIONS    = 10000
STEP_SIZE     = 0.00001   # L6 triangles are 3x smaller than L5 -> ~1/2 of L5 lr
STEP_MIN      = 1e-10
STEP_COOL     = 0.95
STEP_PATIENCE_LOCAL  = 300
STEP_PATIENCE_GLOBAL = 600
WARMUP_ITERS  = 200
SHAPE_DAMPEN  = 0.8
MB_LAMBDA     = 0.5
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999
ADAM_EPS      = 1e-8
TOL           = 1e-9      # on-line tolerance (lattice pitch at L6 ~6.5e-4)
COT_IDEAL     = 1.0 / np.sqrt(3)

# ── wedge geometry (mode-0 face) ─────────────────────────────────────────────
TR, VF, VC = H9K.limits.TR, H9K.limits.VF, H9K.limits.VC
C_ = np.array([0.0, VF])                    # pole face corner
B_ = np.array([TR, VC])                     # right equatorial face corner
M_ = 0.5 * (C_ + B_)                        # midpoint of right lateral edge
G_ = np.array([0.0, 0.0])                   # face centroid
U2 = M_ / np.linalg.norm(M_)                # G–M median direction (= (√3/2,−1/2))
N2 = np.array([-U2[1], U2[0]])              # normal; sign fixed below
if N2 @ C_ < 0:
    N2 = -N2                                # wedge side: N2·p >= 0
U3 = (B_ - C_) / np.linalg.norm(B_ - C_)    # lateral-edge direction (=(1/2,√3/2))
N3 = np.array([-U3[1], U3[0]])
if N3 @ (G_ - C_) < 0:
    N3 = -N3                                # face-interior side: N3·(p−C) >= 0

R120 = np.array([[-0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, -0.5]])
S_P  = np.diag([-1.0, 1.0])
GROUP = [np.eye(2), R120, R120.T, S_P, S_P @ R120, S_P @ R120.T]


def in_wedge(pts, tol=TOL):
    """Closed-wedge membership (face membership assumed)."""
    return (pts[:, 0] >= -tol) & (pts @ N2 >= -tol)


def snap_and_slave(x, roles, parent_idx, parent_T):
    """Exact constraint projection + ghost slaving. roles: dict of masks."""
    x = x.copy()
    x[roles['cC']] = C_
    x[roles['cG']] = G_
    x[roles['cM']] = M_
    m = roles['e1']                                   # x=0 median: x -> 0
    x[m, 0] = 0.0
    m = roles['e2']                                   # G–M median line (origin)
    x[m] = np.outer(x[m] @ U2, U2)
    m = roles['e3']                                   # lateral edge through C
    x[m] = C_ + np.outer((x[m] - C_) @ U3, U3)
    g = roles['ghost']
    # x(ghost) = T^-1 x(parent);  parent_T maps ghost->parent so inverse = T^T
    x[g] = np.einsum('nij,nj->ni', parent_T[g].transpose(0, 2, 1), x[parent_idx[g]])
    return x


def project_tangent(vec, roles):
    """Project per-vertex vectors onto each constraint tangent; zero pinned/ghost."""
    v = vec.copy()
    v[roles['pinned']] = 0.0
    m = roles['e1']
    v[m, 0] = 0.0
    m = roles['e2']
    v[m] = np.outer(v[m] @ U2, U2)
    m = roles['e3']
    v[m] = np.outer(v[m] @ U3, U3)
    v[roles['ghost']] = 0.0
    return v


def measure(x, tris, w, rg, b_raw, g_gcd):
    gpts = rg.project(Points(x, b_raw, oid=0), [b_raw, g_gcd])
    ll = gpts.coords[tris.ravel()]
    areas = wgs84_area_coarse(rg, Points(ll, g_gcd), 3)
    c_ideal = float((w * areas).sum() / w.sum())
    ratios = areas / c_ideal
    return areas, ratios, c_ideal


def compute_gradient(x, tris, ratios, w, move_ok, is_polar, tri_modes):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    p0, p1, p2 = x[v0], x[v1], x[v2]
    e1, e2 = p1 - p0, p2 - p0
    flat = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    sign_f = np.where(flat >= 0.0, 1.0, -1.0)
    absf = np.maximum(np.abs(flat), 1e-30)

    cot0 = np.sum((p1 - p0) * (p2 - p0), axis=1) / (2 * absf)
    cot1 = np.sum((p0 - p1) * (p2 - p1), axis=1) / (2 * absf)
    cot2 = np.sum((p0 - p2) * (p1 - p2), axis=1) / (2 * absf)
    min_cot = np.minimum(np.minimum(cot0, cot1), cot2)
    shape_dev = (COT_IDEAL - min_cot) / COT_IDEAL
    shape_w = np.clip(1.0 - SHAPE_DAMPEN * shape_dev, 0.0, 1.0)
    angle_w = np.where(is_polar, 1.0, shape_w)

    log_r = np.log(ratios)
    if MB_LAMBDA > 0.0 and tri_modes is not None:
        w0, w1 = w[tri_modes == 0], w[tri_modes == 1]
        m0 = (w0 * log_r[tri_modes == 0]).sum() / w0.sum()
        m1 = (w1 * log_r[tri_modes == 1]).sum() / w1.sum()
        shift = (m0 - m1) * MB_LAMBDA * 0.5
        log_r = log_r - np.where(tri_modes == 0, -shift, shift)

    wt = 2.0 * log_r * sign_f / absf * angle_w * w   # orbit weight folded in
    grad = np.zeros_like(x)
    for vi, pa, pb in ((v0, p1, p2), (v1, p2, p0), (v2, p0, p1)):
        opp = pb - pa
        dg = 0.5 * wt[:, None] * np.stack([-opp[:, 1], opp[:, 0]], axis=1)
        mk = move_ok[vi]
        np.add.at(grad, vi[mk], dg[mk])
    return grad


if __name__ == '__main__':
    import argparse
    from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
    from scipy.spatial import cKDTree
    from hhg9.h9.polygon import tri_mesh

    ap = argparse.ArgumentParser(description='Option-B fundamental-domain L6 refinement.')
    ap.add_argument('--ellipsoid', default='Sphere', choices=list(config.ELLIPSOIDS))
    ap.add_argument('--layer', type=int, default=6)
    ap.add_argument('--warm-start', required=False,
                    default=str(config.DATA_DIR / 'Sphere_l5_warp_data.npz'))
    ap.add_argument('--resume', default=None)
    ap.add_argument('--iterations', type=int, default=ITERATIONS)
    ap.add_argument('--lr', type=float, default=STEP_SIZE)
    ap.add_argument('--cool', type=float, default=STEP_COOL,
                    help='lr multiplier on patience trip (default %(default)s)')
    ap.add_argument('--patience-local', type=int, default=STEP_PATIENCE_LOCAL)
    ap.add_argument('--patience-global', type=int, default=STEP_PATIENCE_GLOBAL)
    ap.add_argument('--save-every', type=int, default=100,
                    help='periodic "_last" state save interval (0 = off)')
    ap.add_argument('--output_dir', default='output/stage3b')
    args = ap.parse_args()
    ITERATIONS, STEP_SIZE = args.iterations, args.lr
    STEP_COOL = args.cool
    STEP_PATIENCE_LOCAL = args.patience_local
    STEP_PATIENCE_GLOBAL = args.patience_global
    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    L = args.layer

    rg = Registrar()
    ellipsoid = config.apply_ellipsoid(rg, args.ellipsoid)
    b_raw, g_gcd = rg.domain('b_raw'), rg.domain('g_gcd')
    print(f'Ellipsoid: {ellipsoid}   layer: {L}')

    # ── 1. lattice + representatives ────────────────────────────────────────
    print(f'Building L{L} face lattice (tri_mesh)...')
    verts, _edges, tris_all = tri_mesh(L, 0)
    print(f'  face: {len(verts)} vertices, {len(tris_all)} triangles')

    cen = verts[tris_all].mean(axis=1)
    rep = in_wedge(cen)
    tris_rep = tris_all[rep]
    cen_rep = cen[rep]
    on_m1 = np.abs(cen_rep[:, 0]) <= TOL
    on_m2 = np.abs(cen_rep @ N2) <= TOL
    w_rep = np.where(on_m1 | on_m2, 3.0, 6.0)
    total_w = w_rep.sum()
    print(f'  representatives: {len(tris_rep)} triangles '
          f'({int((w_rep == 3).sum())} on-median), sum(w) = {total_w:.0f} '
          f'(expect {9 ** (L + 1)})')
    assert total_w == 9 ** (L + 1), 'orbit weights do not tile the face'

    # training vertex set = all vertices used by representative triangles
    used = np.unique(tris_rep)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    tris = remap[tris_rep]
    src = verts[used].copy()
    n = len(src)
    print(f'  training vertices: {n}')

    # roles
    inside = in_wedge(src)
    d_cC = np.linalg.norm(src - C_, axis=1)
    d_cG = np.linalg.norm(src - G_, axis=1)
    d_cM = np.linalg.norm(src - M_, axis=1)
    roles = {}
    roles['cC'] = d_cC <= TOL
    roles['cG'] = d_cG <= TOL
    roles['cM'] = d_cM <= TOL
    corner = roles['cC'] | roles['cG'] | roles['cM']
    roles['ghost'] = ~inside
    roles['e1'] = inside & ~corner & (np.abs(src[:, 0]) <= TOL)
    roles['e2'] = inside & ~corner & (np.abs(src @ N2) <= TOL) & ~roles['e1']
    roles['e3'] = inside & ~corner & (np.abs((src - C_) @ N3) <= TOL) \
                  & ~roles['e1'] & ~roles['e2']
    roles['pinned'] = corner
    free = inside & ~corner
    move_ok = free.copy()          # ghosts excluded from direct scatter... see below
    # ghosts DO contribute gradient (fold-back), so scatter must include them:
    move_ok = ~corner              # everything except pinned corners accumulates
    print(f"  roles: free-interior {int((free & ~(roles['e1']|roles['e2']|roles['e3'])).sum())}, "
          f"e1(x=0) {int(roles['e1'].sum())}, e2(G-M) {int(roles['e2'].sum())}, "
          f"e3(C-M) {int(roles['e3'].sum())}, corners {int(corner.sum())}, "
          f"ghosts {int(roles['ghost'].sum())}")

    # ghost -> parent mapping via group image landing in the wedge
    gidx = np.flatnonzero(roles['ghost'])
    parent_idx = np.arange(n)
    parent_T = np.tile(np.eye(2), (n, 1, 1))
    tree = cKDTree(src[inside])
    inside_ids = np.flatnonzero(inside)
    unresolved = gidx.copy()
    for T in GROUP[1:]:
        if len(unresolved) == 0:
            break
        img = src[unresolved] @ T.T
        ok = in_wedge(img)
        if not ok.any():
            continue
        d, j = tree.query(img[ok], k=1)
        good = d < TOL * 10
        ids = unresolved[ok][good]
        parent_idx[ids] = inside_ids[j[good]]
        parent_T[ids] = T
        unresolved = np.setdiff1d(unresolved, ids)
    assert len(unresolved) == 0, f'{len(unresolved)} ghosts unresolved'
    pd = np.linalg.norm(np.einsum('nij,nj->ni', parent_T[gidx], src[gidx])
                        - src[parent_idx[gidx]], axis=1).max() if len(gidx) else 0.0
    print(f'  ghost slaving: {len(gidx)} ghosts, max lattice mismatch {pd:.2e}')

    # per-triangle mode from orientation (verified against region_grid at L4/L5):
    # mode-0 (∇): two vertices above the centroid; mode-1 (△): one above.
    above = (src[tris][:, :, 1] > cen_rep[:, None, 1]).sum(axis=1)
    tri_modes = np.where(above == 2, 0, 1).astype(np.int8)
    print(f'  modes: {(tri_modes == 0).sum()} mode-0, {(tri_modes == 1).sum()} mode-1')

    is_polar = roles['cC'][tris].any(axis=1)

    # ── 2. warm start ────────────────────────────────────────────────────────
    if args.resume:
        ck = np.load(args.resume, allow_pickle=True)
        x_prime = ck['target_pts'].copy()
        assert len(x_prime) == n
        print(f"Resumed {args.resume}: MAE {float(ck['mae']):.8f}")
    else:
        print(f'Mean-slice warm start from {args.warm_start}')
        ws = np.load(args.warm_start, allow_pickle=True)
        ws_src, ws_tgt = ws['source_pts'], ws['target_pts']
        ct = CloughTocher2DInterpolator(ws_src, ws_tgt - ws_src)
        nn = NearestNDInterpolator(ws_src, ws_tgt - ws_src)
        D = np.zeros_like(src)
        for T in GROUP:
            q = src @ T.T
            d = ct(q)
            bad = np.any(np.isnan(d), axis=1)
            if bad.any():
                d[bad] = nn(q[bad])
            D += d @ T          # T^-1 = T^T; row-vector form: d @ (T^T)^T = d @ T
        D /= len(GROUP)
        x_prime = snap_and_slave(src + D, roles, parent_idx, parent_T)

    # ── 3. Adam loop (stage3 conventions) ────────────────────────────────────
    areas, ratios, c_ideal = measure(x_prime, tris, w_rep, rg, b_raw, g_gcd)
    wmae = float((w_rep * np.abs(ratios - 1.0)).sum() / total_w)
    print(f'\n[Baseline]  wMAE: {wmae:.8f} | Min/Max: {ratios.min():.6f}/{ratios.max():.6f}')

    best_mae, best_min, best_max = wmae, ratios.min(), ratios.max()
    best_x = x_prime.copy()
    step = STEP_SIZE
    adam_m = np.zeros_like(x_prime)
    adam_v = np.zeros_like(x_prime)
    adam_t = 0
    no_loc = no_glob = 0
    prev_mae = wmae
    window = [wmae] * 6
    out_path = OUT / f'l0{L}_{ellipsoid}_fund_best.npz'

    print(f'\n--- FUNDAMENTAL-DOMAIN GRADIENT DESCENT ({ITERATIONS} iters, lr={STEP_SIZE}) ---')
    for k in range(ITERATIONS):
        if step < STEP_MIN:
            print(f'Step below minimum, stopping.')
            break
        grad = compute_gradient(x_prime, tris, ratios, w_rep, move_ok, is_polar, tri_modes)
        # fold ghost gradients back onto parents with the orthogonal transform
        if len(gidx):
            gg = np.einsum('nij,nj->ni', parent_T[gidx], grad[gidx])
            np.add.at(grad, parent_idx[gidx], gg)
        grad = project_tangent(grad, roles)

        adam_t += 1
        adam_m = ADAM_BETA1 * adam_m + (1 - ADAM_BETA1) * grad
        adam_v = ADAM_BETA2 * adam_v + (1 - ADAM_BETA2) * grad ** 2
        m_hat = adam_m / (1 - ADAM_BETA1 ** adam_t)
        v_hat = adam_v / (1 - ADAM_BETA2 ** adam_t)
        delta = (step * min(1.0, adam_t / WARMUP_ITERS)) * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        delta = project_tangent(delta, roles)
        x_prime = snap_and_slave(x_prime - delta, roles, parent_idx, parent_T)

        areas, ratios, c_ideal = measure(x_prime, tris, w_rep, rg, b_raw, g_gcd)
        wmae = float((w_rep * np.abs(ratios - 1.0)).sum() / total_w)

        window.append(wmae); window.pop(0)
        no_loc = no_loc + 1 if prev_mae - min(window) < 1e-6 else 0
        prev_mae = wmae
        cmin, cmax = ratios.min(), ratios.max()
        no_regress = (cmin >= best_min) and (cmax <= best_max)
        strict_one = (cmin > best_min) or (cmax < best_max)
        best = (wmae < best_mae and no_regress) or (wmae == best_mae and no_regress and strict_one)
        if best:
            no_glob = 0
        else:
            no_glob += 1
        if k >= WARMUP_ITERS and (no_loc >= STEP_PATIENCE_LOCAL or no_glob >= STEP_PATIENCE_GLOBAL):
            step *= STEP_COOL
            why = 'local' if no_loc >= STEP_PATIENCE_LOCAL else 'global'
            no_loc = no_glob = 0
            print(f'  -> cooling step to {step:.2e} ({why})')
        if best:
            best_mae, best_min, best_max = wmae, cmin, cmax
            best_x = x_prime.copy()
            np.savez(out_path, layer=L, source_pts=src, target_pts=best_x,
                     mae=best_mae, step_size=step, weights=w_rep,
                     ghost_mask=roles['ghost'], adam_t=np.array(adam_t))
            print(f'[Iter {k:5d}]  wMAE: {wmae:.8f} | lr={step:.1e} | '
                  f'Min/Max: {cmin:.8f}/{cmax:.8f} | saved best')
        elif k % 25 == 0:
            print(f'[Iter {k:5d}]  wMAE: {wmae:.8f} | lr={step:.1e} | '
                  f'Min/Max: {cmin:.6f}/{cmax:.6f} | '
                  f'patience L:{STEP_PATIENCE_LOCAL - no_loc} G:{STEP_PATIENCE_GLOBAL - no_glob}')
        if args.save_every and k % args.save_every == 0:
            np.savez(OUT / f'l0{L}_{ellipsoid}_fund_last.npz', layer=L,
                     source_pts=src, target_pts=x_prime, mae=wmae,
                     step_size=step, weights=w_rep, ghost_mask=roles['ghost'],
                     adam_t=np.array(adam_t), iteration=np.array(k))

    print(f'\nBest wMAE: {best_mae:.8f}   checkpoint: {out_path}')
