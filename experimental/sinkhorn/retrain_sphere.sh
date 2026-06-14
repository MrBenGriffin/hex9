#!/bin/zsh
# Sphere (astronomical) warp retrain — fast path per sphere-retrain-brief.md:
# skip stage 1, warm-start stage 2 from the deployed WGS84 F6 field
# (spherical optimum is ~0.3% away; flattening is the only difference).
# --lr 5e-5 per stage2's own guidance for near-converged warm-starts.
set -e
cd "$(dirname "$0")"

echo "=== Sphere retrain start: $(date) ==="

echo "--- stage 2: Adam polish L4 (warm-start: WGS84 F6 deployed field) ---"
python stage2_gradient_l4.py --ellipsoid Sphere \
    --warm-start ../../hhg9/data/WGS84_l5_warp_data.npz \
    --lr 5e-5

echo "--- stage 3: Adam refine L5 ---"
python stage3_gradient_l5.py --ellipsoid Sphere \
    --warm-start output/stage2/l4_tgrid_Sphere_best.npz

echo "--- stage 4: pack ---"
python stage4_pack.py output/stage3/l05_Sphere_best.npz --ellipsoid Sphere

echo "=== Sphere retrain complete: $(date) ==="
