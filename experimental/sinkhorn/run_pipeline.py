# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Orchestrate the authalic-warp training pipeline for one ellipsoid.

    stage1_sinkhorn_l4   Sinkhorn OT on the L4 triangle grid
    stage2_gradient_l4   Adam gradient polish, still L4
    stage3_gradient_l5   Adam gradient refinement, L5 (warm-started from L4)
    stage4_pack          strip + float64 → <ellipsoid>_l5_warp_data.npz

Each stage is run as its own process, so several ellipsoids can train
concurrently — just launch one ``run_pipeline.py --ellipsoid X`` per ellipsoid.
Checkpoints and outputs are ellipsoid-tagged, so concurrent runs never collide.

Examples
--------
    python run_pipeline.py                         # full WGS84 run
    python run_pipeline.py --ellipsoid GRS80       # full GRS80 run
    python run_pipeline.py --stages 3,4            # resume from L5 onward
    python run_pipeline.py --stages 2,3,4          # skip the Sinkhorn solve

Resumption: each stage script accepts --resume; if a matching checkpoint for
this ellipsoid already exists, this orchestrator passes it automatically.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from experimental.sinkhorn import config

HERE = config.SINKHORN_DIR


def _newest(pattern: str) -> Path | None:
    """Most recently modified file matching a glob, or None."""
    hits = sorted(HERE.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _run(script: str, *cli_args: str):
    cmd = [sys.executable, str(HERE / script), *cli_args]
    print(f'\n>>> {" ".join(cmd)}\n', flush=True)
    res = subprocess.run(cmd, cwd=HERE)
    if res.returncode != 0:
        raise SystemExit(f'{script} exited with code {res.returncode}')


def stage1(ell: str):
    resume = _newest(f'output/q_{ell}_l4_iter*.npz')
    args = ['--ellipsoid', ell]
    if resume:
        print(f'stage1: resuming from {resume.name}')
        args += ['--resume', str(resume)]
    _run('stage1_sinkhorn_l4.py', *args)


def stage2(ell: str):
    resume = _newest(f'l4_tgrid_{ell}_*.npz')
    warm   = _newest(f'output/q_{ell}_l4_iter*.npz')
    args = ['--ellipsoid', ell]
    if resume:
        print(f'stage2: resuming from {resume.name}')
        args += ['--resume', str(resume)]
    elif warm:
        print(f'stage2: warm-starting from {warm.name}')
        args += ['--warm-start', str(warm)]
    else:
        raise SystemExit('stage2: no stage-1 output found — run stage 1 first.')
    _run('stage2_gradient_l4.py', *args)


def stage3(ell: str):
    resume = _newest(f'l05_{ell}_k*.npz')
    warm   = _newest(f'l4_tgrid_{ell}_best.npz') or _newest(f'l4_tgrid_{ell}_*.npz')
    args = ['--ellipsoid', ell]
    if resume:
        print(f'stage3: resuming from {resume.name}')
        args += ['--resume', str(resume)]
    elif warm:
        print(f'stage3: warm-starting from {warm.name}')
        args += ['--warm-start', str(warm)]
    else:
        raise SystemExit('stage3: no stage-2 output found — run stage 2 first.')
    _run('stage3_gradient_l5.py', *args)


def stage4(ell: str):
    ckpt = _newest(f'l05_{ell}_k*.npz')
    if not ckpt:
        raise SystemExit('stage4: no stage-3 L5 checkpoint found.')
    print(f'stage4: packing {ckpt.name}')
    _run('stage4_pack.py', str(ckpt), '--ellipsoid', ell)


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run the warp training pipeline.')
    ap.add_argument('--ellipsoid', default=config.INDEX_ELLIPSOID,
                    choices=list(config.ELLIPSOIDS))
    ap.add_argument('--stages', default='1,2,3,4',
                    help='comma-separated subset of stages to run')
    args = ap.parse_args()

    wanted = [int(s) for s in args.stages.split(',') if s.strip()]
    print(f'Pipeline: ellipsoid={args.ellipsoid}  stages={wanted}')
    for s in wanted:
        STAGES[s](args.ellipsoid)
    print('\nPipeline complete.')
