# Part of the Hex9 (H9) Project
# Copyright ©2026, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Pack the Option-B fundamental-domain checkpoint into the two deliverables:

1. <ELL>_l6_fund_warp_data.npz  — canonical fundamental-domain artifact.
   Keys:
     source_pts (N,2) float64 — unwarped fundamental-wedge lattice, mode-0
                                b_raw coords (ghost rows EXCLUDED)
     target_pts (N,2) float64 — warped positions (same inverse src->dst
                                semantics as the L5 files)
     layer      ()    int     — lattice layer (6)
     corners    (3,2) float64 — wedge corners C (pole), M (lateral-edge
                                midpoint), G (face centroid)
     edge_class (N,)  uint8   — 0 interior, 1 on C-G median (x=0),
                                2 on G-M median, 3 on C-M half face edge,
                                4 corner (pinned)

2. <ELL>_l6_warp_data.npz — FULL-FACE unfolded field, exactly the existing
   convention (source_pts, target_pts, layer), consumable by the PROJ
   exporter unchanged. Unfolded via the six D3 group ops; lossless because
   the trained field is D3-exact by construction: x'(T v) = T x'(v).

Usage:
  python pack_unfold_l6.py output/stage3b/l06_Sphere_fund_best.npz \
      --ellipsoid Sphere --outdir ../../hhg9/data
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from hhg9.h9.polygon import tri_mesh
from experimental.sinkhorn.stage3b_fund_l6 import (
    GROUP, C_, G_, M_, N2, N3, U2, U3, TOL, in_wedge)


def classify(src):
    d = lambda p: np.linalg.norm(src - p, axis=1)
    cls = np.zeros(len(src), dtype=np.uint8)
    on1 = np.abs(src[:, 0]) <= TOL
    on2 = np.abs(src @ N2) <= TOL
    on3 = np.abs((src - C_) @ N3) <= TOL
    cls[on1] = 1
    cls[on2 & (cls == 0)] = 2
    cls[on3 & (cls == 0)] = 3
    cls[(d(C_) <= TOL) | (d(G_) <= TOL) | (d(M_) <= TOL)] = 4
    return cls


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--ellipsoid', default='Sphere')
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()

    ck = np.load(args.checkpoint, allow_pickle=True)
    src_all = np.asarray(ck['source_pts'], dtype=np.float64)
    tgt_all = np.asarray(ck['target_pts'], dtype=np.float64)
    layer = int(ck['layer'])
    ghost = np.asarray(ck['ghost_mask'], dtype=bool)
    mae = float(ck['mae']) if 'mae' in ck.files else float('nan')
    print(f'checkpoint: {args.checkpoint}  layer={layer}  wMAE={mae:.8f}')
    print(f'  {len(src_all)} training vertices ({int(ghost.sum())} ghosts stripped)')

    # ── deliverable 1: fundamental artifact (no ghosts) ──────────────────────
    keep = ~ghost
    f_src, f_tgt = src_all[keep], tgt_all[keep]
    cls = classify(f_src)
    out1 = args.outdir / f'{args.ellipsoid}_l{layer}_fund_warp_data.npz'
    np.savez(out1, source_pts=f_src, target_pts=f_tgt, layer=np.array(layer),
             corners=np.array([C_, M_, G_]), edge_class=cls)
    disp = np.linalg.norm(f_tgt - f_src, axis=1)
    print(f'wrote {out1}: {len(f_src)} pts, max|D|={disp.max():.4e}, '
          f'classes {np.bincount(cls)}')

    # ── deliverable 2: full-face unfold ──────────────────────────────────────
    print(f'Unfolding to full face (tri_mesh({layer}))...')
    verts, _e, _t = tri_mesh(layer, 0)
    n = len(verts)
    tree = cKDTree(f_src)
    face_tgt = np.full((n, 2), np.nan)
    todo = np.ones(n, dtype=bool)
    for T in GROUP:
        if not todo.any():
            break
        img = verts[todo] @ T.T                     # row form of T @ v
        ok = in_wedge(img)
        if not ok.any():
            continue
        dist, j = tree.query(img[ok], k=1)
        good = dist < 1e-8
        ids = np.flatnonzero(todo)[ok][good]
        # x'(v) = T^-1 x'(T v) ; row form: x'(Tv) @ T (since T^-T = T)
        face_tgt[ids] = f_tgt[j[good]] @ T
        todo[ids] = False
    assert not todo.any(), f'{int(todo.sum())} face vertices failed to fold'
    assert not np.isnan(face_tgt).any()
    out2 = args.outdir / f'{args.ellipsoid}_l{layer}_warp_data.npz'
    np.savez(out2, source_pts=verts, target_pts=face_tgt, layer=np.array(layer))
    disp = np.linalg.norm(face_tgt - verts, axis=1)
    print(f'wrote {out2}: {n} pts, max|D|={disp.max():.4e}')
