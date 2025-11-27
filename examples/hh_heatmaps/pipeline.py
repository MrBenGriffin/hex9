from __future__ import annotations
from pathlib import Path
import importlib
import time

# Ordered list of stage modules. Add/remove as you refactor stages.
STAGES = [
    "pr0001_csv",      # CSV -> NPY (lon,lat,pop)
    "pr0002_prj",      # NPY -> barycentric coords/components (memmap)
    "pr0003_prj",     # Population extract + boundary projection (signatured)
    "pr0004_gcd_img",  # OSM background compositor
    "pr0005_theta",    # θ/centroid from GCD bounds
    "pr0006_grid",
    "pr0007_hh_heatmap",
]


def run_stage(modname: str, file_id: str, **opts):
    """Import module and call its stage(file_id, **opts)."""
    t0 = time.time()
    mod = importlib.import_module(modname)
    if not hasattr(mod, "stage"):
        print(f"[SKIP] {modname}.stage() not found.")
        return None
    out = mod.stage(file_id, **opts)
    dt = time.time() - t0
    print(f"[DONE] {modname}.stage('{file_id}') in {dt:.2f}s → {out}")
    return out


def run(file_id: str, from_stage: int = 1, to_stage: int | None = None, **opts):
    """Run STAGES[from_stage..to_stage] inclusive (1-based)."""
    i0 = max(1, from_stage) - 1
    i1 = len(STAGES) if to_stage is None else min(to_stage, len(STAGES))
    for i, modname in enumerate(STAGES[i0:i1], start=i0 + 1):
        print(f"\n=== Stage {i}: {modname} ===")
        run_stage(modname, file_id, **opts)


if __name__ == "__main__":
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description="KISS pipeline runner")
    # FIXME --file shouldn't really be a switch.
    p.add_argument('--file', type=str, default='ggy', help="dataset id, e.g. 'btn', 'jan'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')

    p.add_argument("--from", dest="from_stage", type=int, default=1, help="start stage index (1-based)")
    p.add_argument("--to", dest="to_stage", type=int, default=None, help="end stage index (1-based, inclusive)")

    # Common options passed through to stages (only those that stages understand will be used)
    p.add_argument("--bounds", type=float, nargs=4, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    p.add_argument("--preset", type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument("--tag", type=str, help="signature tag to embed in output filenames")
    p.add_argument("--batch", type=int, help="batch size (stage-specific)")
    p.add_argument("--workers", type=int, help="parallel workers (stage-specific)")
    p.add_argument("--accuracy", type=float, help="projection accuracy in cm (stage-specific)")
    p.add_argument("--force", action="store_true", help="force recompute where supported")

    p.add_argument('--layers', type=int, nargs=2, default=None,
                   help='[start end) layers to render in stage 7; e.g. 12 13 renders only layer 12')
    p.add_argument('--hex-layers', type=int, default=None,
                   help='classification depth for hex addressing (max depth for ugc_regions)')

    args = p.parse_args()

    # Build kwargs dict only with provided (non-None) options
    opts = {
        "src_dir": args.src,
        "dst_dir": args.dst,
        "bounds": tuple(args.bounds) if args.bounds else None,
        "preset": args.preset,
        "tag": args.tag,
        "force": args.force,
    }
    if args.batch is not None:
        opts["batch_size"] = args.batch
    if args.workers is not None:
        opts["n_workers"] = args.workers
    if args.accuracy is not None:
        opts["accuracy_cm"] = args.accuracy
    if args.layers is not None:
        opts["layer_range"] = (int(args.layers[0]), int(args.layers[1]))
    if args.hex_layers is not None:
        opts["hex_layers"] = int(args.hex_layers)

    run(args.file, from_stage=args.from_stage, to_stage=args.to_stage, **opts)
