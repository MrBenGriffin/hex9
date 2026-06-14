#!/bin/zsh
# F6 lateral-edge-tangency retrain — full WGS84 pipeline, stages driven
# explicitly (run_pipeline.py globs predate the output/stage2|3 layout).
set -e
cd "$(dirname "$0")"

echo "=== F6 retrain start: $(date) ==="

echo "--- stage 1: Sinkhorn L4 (fresh) ---"
python stage1_sinkhorn_l4.py --ellipsoid WGS84

S1=$(ls -t output/q_WGS84_l4_iter*.npz | head -1)
echo "--- stage 2: Adam polish L4 (warm-start: $S1) ---"
python stage2_gradient_l4.py --ellipsoid WGS84 --warm-start "$S1"

echo "--- stage 3: Adam refine L5 ---"
python stage3_gradient_l5.py --ellipsoid WGS84 \
    --warm-start output/stage2/l4_tgrid_WGS84_best.npz

echo "--- stage 4: pack ---"
python stage4_pack.py output/stage3/l05_WGS84_best.npz --ellipsoid WGS84

echo "=== F6 retrain complete: $(date) ==="
