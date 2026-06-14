# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Stage 4 — pack a stage-3 (L5) checkpoint into a production warp file.

Picks the chosen L5 checkpoint, strips the training-only arrays (Adam moments,
step size, iteration counters) and writes a lean float64 ``source_pts`` /
``target_pts`` pair as ``<ellipsoid>_l5_warp_data.npz``.

float64 is mandatory: the domain is ±√2 in scale, and float32 (abs precision
~1e-7) introduces ~µm-level noise in the seam displacement derivative.

By default the packed file is written next to this script; copy it into
``hhg9/data/`` to deploy. Pass ``--deploy`` to write straight into ``hhg9/data``.
"""
import argparse
from pathlib import Path
import numpy as np

from experimental.sinkhorn import config

KEEP = ('source_pts', 'target_pts')


def pack(checkpoint: Path, out: Path):
    repo = np.load(checkpoint, allow_pickle=True)
    missing = [k for k in KEEP if k not in repo.files]
    if missing:
        raise KeyError(f'{checkpoint} is missing required arrays: {missing}')

    src = np.asarray(repo['source_pts'], dtype=np.float64)
    dst = np.asarray(repo['target_pts'], dtype=np.float64)
    layer = int(repo['layer']) if 'layer' in repo.files else 5
    mae = float(repo['mae']) if 'mae' in repo.files else float('nan')
    repo.close()

    disp = np.linalg.norm(dst - src, axis=1)
    np.savez(out, source_pts=src, target_pts=dst, layer=np.array(layer))
    print(f'Packed {checkpoint.name}  →  {out}')
    print(f'  layer={layer}  vertices={len(src)}  training MAE={mae:.8f}')
    print(f'  max displacement={disp.max():.3e}  (b_oct units)')
    print(f'  size={out.stat().st_size / 1e6:.2f} MB  dtype=float64')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 4 — pack production warp.')
    ap.add_argument('checkpoint', type=Path, help='stage-3 L5 checkpoint .npz')
    ap.add_argument('--ellipsoid', default=config.INDEX_ELLIPSOID,
                    choices=list(config.ELLIPSOIDS))
    ap.add_argument('--deploy', action='store_true',
                    help='write into hhg9/data/ instead of the local directory')
    args = ap.parse_args()

    name = f'{args.ellipsoid}_l5_warp_data.npz'
    out_dir = config.DATA_DIR if args.deploy else config.SINKHORN_DIR
    out = out_dir / name
    pack(args.checkpoint, out)
    if not args.deploy:
        print(f'\nTo deploy:  cp {out}  {config.DATA_DIR / name}')
