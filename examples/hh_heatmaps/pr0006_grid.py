"""
Part of the H9 project - Preparation 0006
Project the plate-carree imagemap onto barycentric via sampling.
Last Tested 16 August 2025 √
"""

import numpy as np
from matplotlib import image
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.h9.grid import qa_grid
from hhg9.projections import PlatePixelGCD
from experimental.support import Util
from PIL import Image  # Pillow for clean image saving

from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          width_px: int = 6000,
          force: bool = False,
          # theta_bias = None,
          **_):
    """
    Stage 5: Sample a plate-carree background into a barycentric grid using θ and centroid.
    Reads signatured inputs from previous stages and writes a signatured grid image.

    Inputs (preferred):
      - <src>/<file>__<sig>_lat_lon_bounds.npy        (from stage 2b)
      - <src>/<file>__<sig>_gcd.png                   (from stage 3)
      - <src>/<file>__<sig>_bry_border.npy            (from stage 4)
      - <src>/<file>__<sig>_bounds_bry_cmp.npy        (from stage 2b)
      - <src>/<file>__<sig>_theta.npy                 (from stage 4)
      - <src>/<file>__<sig>_centroid.npy              (from stage 4)

    Output:
      - <dst>/<file>__<sig>_grid.png
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Resolve signature + bounds (use explicit/preset → fallback to latest signatured bounds)
    presets = load_presets()
    if preset:
        extent, sig = resolve_bounds(file, presets, preset=preset)
        base = f"{file}__{tag or sig}"
        bounds_file = src_dir / f"{base}_lat_lon_bounds.npy"
    elif bounds is not None:
        extent, sig = resolve_bounds(file, presets, explicit=bounds)
        base = f"{file}__{tag or sig}"
        bounds_file = src_dir / f"{base}_lat_lon_bounds.npy"
    else:
        cand = list(src_dir.glob(f"{file}__*_lat_lon_bounds.npy"))
        if cand:
            cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            bounds_file = cand[0]
            base = bounds_file.name.replace("_lat_lon_bounds.npy", "")
            extent = np.load(bounds_file).astype(float)
        else:
            bounds_file = src_dir / f"{file}_lat_lon_bounds.npy"
            base = file
            if not bounds_file.exists():
                raise FileNotFoundError(
                    f"Bounds file not found: {bounds_file}. Provide --preset/--bounds or run stage 2b.")
            extent = np.load(bounds_file).astype(float)

    # Ensure extent ordering [min_lat, min_lon, max_lat, max_lon]
    extent = np.asarray(extent, dtype=float).reshape(4,)
    min_lat, min_lon, max_lat, max_lon = extent
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon

    # Signatured input/output paths
    gcd_img   = src_dir / f"{base}_gcd.png"
    b_border  = src_dir / f"{base}_bry_border.npy"

    # Prefer per-corner components saved by stage 4; fallback to 2b if missing
    bryc_path = src_dir / f"{base}_bry_border_cmp.npy"
    if bryc_path.exists():
        cmp_arr = np.load(bryc_path)  # shape (4, 3)
    else:
        # Fallback: recompute components by projecting the 4 GCD corners again
        try:
            gcd_r = np.array([
                [min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lon]
            ], dtype=float)
            reg = Registrar()
            g_gcd = reg.domain('g_gcd')
            c_ell = reg.domain('c_ell')
            c_oct = reg.domain('c_oct')
            b_oct = reg.domain('b_oct')
            ak = reg.projection('oct_ell')
            ak.set_accuracy(0.0000000001)
            b_gcd = Points(gcd_r, g_gcd)
            b_data = reg.project(b_gcd, [g_gcd, c_ell, c_oct, b_oct])
            cmp_arr = b_data.components.copy()
        except Exception:
            cmp_arr = np.load(src_dir / f"{base}_bounds_bry_cmp.npy")  # last resort: (2,3) min/max only

    # Ensure we have 4 components; if only 2 were available, duplicate as best-effort
    if cmp_arr.shape[0] == 2:
        cmp_arr = np.array([cmp_arr[0], cmp_arr[0], cmp_arr[1], cmp_arr[1]], dtype=cmp_arr.dtype)

    b_cmp     = src_dir / f"{base}_bounds_bry_cmp.npy"
    theta_np  = src_dir / f"{base}_theta.npy"
    cent_np   = src_dir / f"{base}_centroid.npy"
    rot_bry   = src_dir / f"{base}_rot_bry_border.npy"
    out_grid  = dst_dir / f"{base}_grid.png"

    # Up-to-date check (skip unless force)
    inputs = [bounds_file, gcd_img, b_border, b_cmp, theta_np, cent_np, rot_bry]
    if not force and not needs_run(inputs, [out_grid]):
        print(f"[pr0006] up-to-date: {out_grid}")
        return out_grid

    # Registrar & domains
    util = Util()
    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    c_oct = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    ak = reg.projection('oct_ell')
    ak.set_accuracy(0.0000000001)

    # Load inputs: use the *rotated* border so the grid is tight in its own extent
    rot_border = np.load(rot_bry)  # polygon corners in rotated b_oct (shape (4,2))

    # Ensure the 4 corners are ordered consistently around their centroid (CCW)
    poly4 = np.asarray(rot_border, dtype=float)
    ctr4 = poly4.mean(axis=0)
    ang = np.arctan2(poly4[:, 1] - ctr4[1], poly4[:, 0] - ctr4[0])
    order = np.argsort(ang)  # CCW order
    rot_border = poly4[order]
    # Reorder per-corner components in the same way (cmp_arr already prepared above)
    if 'cmp_arr' in locals() and getattr(cmp_arr, 'shape', (0,))[0] == 4:
        cmp_arr = cmp_arr[order]

    # Triangulate the quad (0-1-2) and (0-2-3) in *rotated* space
    A, B, C, D = rot_border

    # Prepare grid and rotate back into b_oct space
    img_w = int(width_px)
    (ww, hh, pxl, msk, (grid_org, (sx, sy))) = qa_grid(rot_border, img_w)
    np.save(src_dir / f"{base}_bg_extent.npy", grid_org)
    # Fallback: if convex mask is empty (quad may be bow‑tie across a seam),
    # recompute mask as union of two triangles (0-1-2) ∪ (0-2-3)
    def _inside_tri(P, T0, T1, T2, eps=1e-12):
        # Robust barycentric test, orientation agnostic
        v0 = T1 - T0
        v1 = T2 - T0
        a = P - T0  # (hex_layer,2)
        den = v0[0]*v1[1] - v0[1]*v1[0]
        # signed sub-areas (scaled by 2*area)
        s = a[:, 0]*v1[1] - a[:, 1]*v1[0]
        t = v0[0]*a[:, 1] - v0[1]*a[:, 0]
        if den < 0:
            den = -den; s = -s; t = -t
        return (s >= -eps) & (t >= -eps) & (s + t <= den + eps)
    if not np.any(msk):
        msk_tri = _inside_tri(pxl, A, B, C) | _inside_tri(pxl, A, C, D)
        if np.any(msk_tri):
            print("[pr0006] convex mask empty; using triangle-union fallback")
            msk = msk_tri
    data = pxl[msk]

    # Rotate sampled points back into the original b_oct space (inverse of stage-4 rotation)
    theta = float(np.load(theta_np))

    centroid = np.load(cent_np)
    c_th, s_th = np.cos(theta), np.sin(theta)
    rot = np.array([[c_th, -s_th], [s_th, c_th]])
    bary = (data - centroid) @ rot.T + centroid  # inverse rotation

    # If the mask produced no interior points, write an empty (transparent) image and return
    if bary.size == 0:
        out = np.zeros((hh + 1, ww + 1, 4), dtype=np.uint8)
        Image.fromarray(out).save(out_grid)
        print(f"[pr0005] Empty interior after qa_grid; saved blank {out_grid}  size={ww+1}x{hh+1}")
        return out_grid

    def inside_tri(P, T0, T1, T2, eps=1e-12):
        v0 = T1 - T0
        v1 = T2 - T0
        a = P - T0
        den = v0[0]*v1[1] - v0[1]*v1[0]
        s = a[:, 0]*v1[1] - a[:, 1]*v1[0]
        t = v0[0]*a[:, 1] - v0[1]*a[:, 0]
        if den < 0:
            den = -den; s = -s; t = -t
        return (s >= -eps) & (t >= -eps) & (s + t <= den + eps)

    t1_mask = inside_tri(bary, A, B, C)
    # t1_mask is defined over the *subset* `bary` (pxl[msk]), so its complement is also subset-sized
    t2_mask = ~t1_mask

    # Choose a component for each triangle (mode of its corners; if tie, pick first)
    def dominant_cmp(idxs):
        tri_cmps = [tuple(cmp_arr[i]) for i in idxs]
        # Count occurrences
        uniq, counts = np.unique(np.array(tri_cmps), axis=0, return_counts=True)
        return tuple(uniq[np.argmax(counts)])

    cmp_t1 = dominant_cmp([0, 1, 2])
    cmp_t2 = dominant_cmp([0, 2, 3])

    if cmp_t1 != cmp_t2:
        print(f"[pr0005] seam split: tri1 cmp={cmp_t1}, tri2 cmp={cmp_t2}")

    # Helper to create Points with replicated components
    def _pts_with_cmp(coords_subset, cmp_tuple, subdomain):
        if coords_subset.size == 0:
            return None
        comp_row = np.array(cmp_tuple, dtype=np.int8)
        comps = np.repeat(comp_row[None, :], coords_subset.shape[0], axis=0)
        return Points(coords_subset, subdomain, components=comps)

    # Project triangle subsets back to GCD via b_oct with their component (octant) assigned
    out_coords = np.empty_like(bary)
    if np.any(t1_mask):
        refs1 = _pts_with_cmp(bary[t1_mask], cmp_t1, b_oct)
        if refs1 is not None:
            sp1a = reg.project(refs1, [b_oct, c_oct, c_ell, g_gcd])
            out_coords[t1_mask] = sp1a.coords
    if np.any(t2_mask):
        refs2 = _pts_with_cmp(bary[t2_mask], cmp_t2, b_oct)
        if refs2 is not None:
            sp1b = reg.project(refs2, [b_oct, c_oct, c_ell, g_gcd])
            out_coords[t2_mask] = sp1b.coords

    sp1_coords = out_coords

    # Load plate-carree image and build a pixel domain instance bound to the extent
    if not gcd_img.exists():
        raise FileNotFoundError(f"{gcd_img} not found. Run stage 3.")
    img = image.imread(str(gcd_img), 'png')
    p_pix = reg.domain('p_pix')
    pc_px = p_pix.adopt(img)  # create pixel cloud from image
    # Order for set_dim is (lon_min, lon_max, lat_min, lat_max)
    PlatePixelGCD(reg).set_dim(pc_px, (min_lon, max_lon, min_lat, max_lat))
    pc_sp = reg.project(pc_px, [p_pix, g_gcd])

    # KDTree sample
    src_tree = KDTree(pc_sp.coords)
    _, idx = src_tree.query(sp1_coords, workers=-1)
    samples = pc_sp.samples[idx]
    if samples.shape[1] == 3:
        rgba = np.hstack((samples, np.ones((samples.shape[0], 1), dtype=samples.dtype)))
    else:
        rgba = samples

    # Paint into raster
    pvx = np.round((pxl[:, 0] - grid_org[0]) * sx).astype(int)
    pvy = np.round((pxl[:, 1] - grid_org[1]) * sy).astype(int)
    pvy = hh - pvy  # flip y so top row is y=0
    out = np.zeros((hh + 1, ww + 1, 4), dtype=float)
    # Sanity check: only draw samples that fall within the raster bounds
    vx = pvx[msk]
    vy = pvy[msk]
    valid = (vx >= 0) & (vx <= ww) & (vy >= 0) & (vy <= hh)
    if not np.all(valid):
        dropped = int((~valid).sum())
        if dropped:
            print(f"[pr0006] warning: {dropped} samples fell outside raster and were dropped")
    out[vy[valid], vx[valid]] = rgba[valid]

    image_uint8 = (out * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(out_grid)
    print(f"[pr0005] Saved {out_grid}  size={ww+1}x{hh+1}")
    return out_grid


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 5: sample plate-carree into barycentric grid')
    # FIXME --file shouldn't really be a switch.
    p.add_argument('--file', type=str, default='gbr', help="dataset id, e.g. 'ggy', 'btn', 'jpn'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')

    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--width', type=int, default=6000, help='grid width in pixels')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          width_px=args.width, force=args.force)
