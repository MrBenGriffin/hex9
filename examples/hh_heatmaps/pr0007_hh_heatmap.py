"""
Part of the H9 project - Preparation 0007
Hex-hex heatmap overlay in barycentric space
Last Tested 16 August 2025 √
"""
import numpy as np
from PIL import Image
from matplotlib.colors import Normalize, LogNorm
from hhg9 import Registrar, Points
from hhg9.h9 import H9P
from hhg9.h9.polygon import hex_reduce, hex_parents, ctr_from_pars
from hhg9.h9.addressing import tail_unpack_reversible, tail_key_from_reversible
from matplotlib import pyplot as plt, image
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run


def _hex_verts_b(hex_par, xpm, xc2, scale):
    """Full-hex vertices in b_oct barycentric space. Returns (N, 6, 2)."""
    return (hex_par[:, None, :] + H9P.hx[xpm, xc2] * scale).reshape(-1, 6, 2)


def stage(file: str,
          src_dir: str | Path = "src",
          dst_dir: str | Path = "src",
          bounds: tuple[float, float, float, float] | None = None,
          preset: str | None = None,
          tag: str | None = None,
          hex_layers: int = 15,
          layer_range: tuple[int, int] = (4, 10),
          cmap_name: str = 'plasma',
          face_alpha: float = 0.10,
          dpi: int = 100,
          img_w_pix: int = 6000,
          force: bool = False,
          show_legend: bool = False,
          save_mask: bool = False,
          scale: str = 'linear',
          log_eps: float = 1e-6,
          show_stats: bool = True,
          debug: bool = False,
          **_):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Resolve signature/base (prefer preset/bounds; else pick most recent signatured theta)
    presets = load_presets()
    if preset:
        _, sig = resolve_bounds(file, presets, preset=preset)
        base = f"{file}__{tag or sig}"
    elif bounds is not None:
        _, sig = resolve_bounds(file, presets, explicit=bounds)
        base = f"{file}__{tag or sig}"
    else:
        # fallback to most recent signature from *_theta.npy
        cand = sorted(src_dir.glob(f"{file}__*_theta.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
        base = cand[0].name.replace("_theta.npy","") if cand else file

    # Highest rendered hex_layer tag (e.g., L12) for distinct outputs per run
    hi_layer = layer_range[1] - 1
    layer_tag = f"L{hi_layer:02d}"

    # Inputs
    theta_f    = src_dir / f"{base}_theta.npy"
    centroid_f = src_dir / f"{base}_centroid.npy"
    grid_img_f = src_dir / f"{base}_grid.png"
    bg_ext_f   = src_dir / f"{base}_bg_extent.npy"
    bry_f      = src_dir / f"{base}_bry_border.npy"
    # Prefer per-corner components from stage 4; fallback to older 2-corner file
    bryc_f     = src_dir / f"{base}_bry_border_cmp.npy"
    bryc_old   = src_dir / f"{base}_bounds_bry_cmp.npy"
    pop_f      = src_dir / f"{base}_pop_data.npy"
    out_png    = dst_dir / f"{base}_{layer_tag}_heatmap.png"

    if debug:
        print(f"[pr0007] output: {out_png.name}")

    # Up-to-date check: include whichever cmp file exists
    cmp_for_inputs = bryc_f if bryc_f.exists() else bryc_old
    inputs = [theta_f, centroid_f, grid_img_f, bg_ext_f, bry_f, cmp_for_inputs, pop_f]
    if not force and not needs_run(inputs, [out_png]):
        print(f"[pr0007] up-to-date: {out_png}")
        return out_png

    # Load numerics
    theta = float(np.load(theta_f))
    centroid = np.load(centroid_f)
    cos_, sin_ = np.cos(theta), np.sin(theta)
    matrix = np.array([[cos_, -sin_], [sin_, cos_]])

    pop_data = np.load(pop_f)
    # Normalize population input: allow (hex_layer,) weights or Nx2/3 [lon,lat,(w)]
    pop_weights = None
    if pop_data.ndim == 1:
        pop_weights = pop_data.astype(float)
    elif pop_data.ndim == 2 and pop_data.shape[1] >= 3:
        pop_weights = pop_data[:, 2].astype(float)
    pos_b = np.load(bry_f)              # (4,2)
    if bryc_f.exists():
        pos_c = np.load(bryc_f)         # expected (4,3)
    else:
        pos_c = np.load(bryc_old)       # legacy (2,3)
        if pos_c.shape[0] == 2:
            # Expand 2-corner comps to 4 corners (min→A,B ; max→C,D) as a best-effort fallback
            pos_c = np.array([pos_c[0], pos_c[0], pos_c[1], pos_c[1]], dtype=pos_c.dtype)
    if pos_c.shape != (4, 3):
        raise ValueError(f"[pr0007] invalid cmp shape: expected (4,3), got {pos_c.shape}")

    reg = Registrar()
    g_gcd = reg.domain('g_gcd')
    c_ell = reg.domain('c_ell')
    c_oct = reg.domain('c_oct')
    b_oct = reg.domain('b_oct')
    ak = reg.projection('oct_ell')
    ak.set_accuracy(0.0000000001)

    # --- Population‑driven classification/enmesh ---
    # Use rotated border only for plotting extent
    rot_border = np.load(src_dir / f"{base}_rot_bry_border.npy")
    minx, miny = float(np.min(rot_border[:, 0])), float(np.min(rot_border[:, 1]))
    maxx, maxy = float(np.max(rot_border[:, 0])), float(np.max(rot_border[:, 1]))
    if debug:
        print(f"[pr0007] extent (rot): x=[{minx:.6f},{maxx:.6f}] y=[{miny:.6f},{maxy:.6f}], theta={np.degrees(theta):.6f}°")

    # Prefer precomputed barycentric population (stage 2); fallback to reprojection
    pop_bry_f = src_dir / f"{base}_bry.npy"
    pop_cmp_f  = src_dir / f"{base}_bry_cmp.npy"
    if pop_bry_f.exists() and pop_cmp_f.exists():
        pop_bary = np.load(pop_bry_f)              # (hex_layer,2)
        pop_cmp  = np.load(pop_cmp_f)               # (hex_layer,3)
        pop_pts_b = Points(pop_bary, b_oct, components=pop_cmp)
        if debug:
            print(f"[pr0007] loaded precomputed bary points: {pop_bary.shape}")
    else:
        # Fallback from lon/lat ONLY if provided in pop_data
        if not (pop_data.ndim == 2 and pop_data.shape[1] >= 2):
            raise ValueError(
                f"[pr0007] missing precomputed bary arrays ({pop_bry_f.name},{pop_cmp_f.name}) and no lon/lat in pop_data; "
                f"run earlier stages to produce *_pop_bary.npy / *_pop_cmp.npy or provide Nx2/3 lon/lat[(w)]."
            )
        lon = pop_data[:, 0].astype(float)
        lat = pop_data[:, 1].astype(float)
        ll  = np.column_stack([lat, lon])
        pop_pts_g = Points(ll, g_gcd)
        pop_pts_c = reg.project(pop_pts_g, [g_gcd, c_ell, c_oct])
        pop_cmp   = c_oct.binning(pop_pts_c.coords)
        pop_pts_b = reg.project(pop_pts_c, [c_oct, b_oct])
        pop_pts_b = Points(pop_pts_b.coords, b_oct, components=pop_cmp)
        if debug:
            print(f"[pr0007] projected bary points: {pop_pts_b.coords.shape}")

    # Bin population points to the final hex layer
    L_final = layer_range[1] - 1
    if L_final <= 0:
        raise ValueError(f"[pr0007] invalid layer_range {layer_range}")

    hex_num, hex_v, hex_inv, _ = hex_reduce(pop_pts_b, L_final)

    # Aggregate population per hex (weighted sum or point count)
    if pop_weights is not None and len(pop_weights) == len(pop_pts_b):
        hex_pop = np.zeros(hex_num, dtype=float)
        np.add.at(hex_pop, hex_inv, pop_weights)
    else:
        hex_pop = np.bincount(hex_inv, minlength=hex_num).astype(float)

    if show_stats:
        print(f"[pr0007] L{L_final}: {hex_num} hexes, "
              f"pop range [{hex_pop.min():.0f}, {hex_pop.max():.0f}], "
              f"total {hex_pop.sum():.0f}")

    # Hex parents + centroids
    tails = tail_key_from_reversible(hex_v[:, -1])
    hex_par, hex_oid, scale = hex_parents(b_oct, hex_v, hex_num)
    ctr_pts = ctr_from_pars(b_oct, hex_par, hex_oid, scale, tails)

    # Build polygon layers: ancestor outlines then final colored layer
    final_polys = []
    for anc_level in range(layer_range[0], L_final):
        anc_num, anc_v, _, _ = hex_reduce(ctr_pts, anc_level)
        anc_par, anc_oid, anc_scale = hex_parents(b_oct, anc_v, anc_num, anc_level)
        anc_xpm, anc_xc2, _, _ = tail_unpack_reversible(anc_v[:, -1])
        verts = _hex_verts_b(anc_par, anc_xpm, anc_xc2, anc_scale)
        # Rotate into aligned frame
        flat = (verts.reshape(-1, 2) - centroid) @ matrix + centroid
        final_polys.append(flat.reshape(verts.shape))
        if show_stats:
            print(f"[pr0007] L{anc_level}: {anc_num} ancestor hexes")

    # Final (population) layer
    xpm, xc2, _, _ = tail_unpack_reversible(hex_v[:, -1])
    verts = _hex_verts_b(hex_par, xpm, xc2, scale)
    flat = (verts.reshape(-1, 2) - centroid) @ matrix + centroid
    final_polys.append(flat.reshape(verts.shape))

    final_pops = hex_pop

    if not final_polys:
        print("[pr0007] nothing to draw (population)")
        return out_png

    # Figure size using rotated extent
    ratio = (maxy - miny) / max(1e-12, (maxx - minx))
    img_h_pix = int(img_w_pix * ratio)
    fig = plt.figure(figsize=(img_w_pix / dpi, img_h_pix / dpi), dpi=dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)

    if not grid_img_f.exists():
        raise FileNotFoundError(f"{grid_img_f} not found (run stage 5)")
    rgb = image.imread(grid_img_f.as_posix(), 'png')
    # force RGBA (handle grayscale or RGB)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb, np.ones_like(rgb)], axis=-1)
    elif rgb.shape[2] == 3:
        a = np.ones(rgb.shape[:2], dtype=rgb.dtype)
        rgb = np.dstack([rgb, a])

    # Draw the stage-5 grid image in the *same rotated barycentric extent*
    # Use a math-style axis (y increases upward) to match polygon coordinates
    # Draw with image origin at the top to match how the PNG was saved in stage 5
    bg_artist = ax.imshow(rgb, extent=(minx, maxx, miny, maxy), alpha=1.0, origin='upper')

    # --- Robust colormap handling ---
    cmap = plt.get_cmap(cmap_name)
    fp = np.asarray(final_pops, dtype=float)
    if fp.size == 0:
        print("[pr0007] no final_pops; skipping fill")
        return out_png

    # Basic stats
    if show_stats:
        finite = fp[np.isfinite(fp)]
        if finite.size:
            stats = dict(
                count=int(finite.size),
                min=float(finite.min()),
                max=float(finite.max()),
                mean=float(finite.mean()),
                median=float(np.median(finite)),
                std=float(finite.std(ddof=0)),
            )
            print(f"[pr0007] stats: {stats}")
        else:
            print("[pr0007] stats: no finite values")

    # Normalization
    if scale.lower() == 'log':
        pos = fp > 0
        if not np.any(pos):
            # fallback to linear with dummy range
            vmin, vmax = 0.0, 1.0
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            fp_plot = fp
            print("[pr0007] warning: no positive values for LogNorm; using linear pixels")
        else:
            vmin = max(float(fp[pos].min()), float(log_eps))
            vmax = float(fp.max())
            if np.isclose(vmin, vmax):
                vmax = vmin * 10.0
            # Replace non-positive values by vmin so LogNorm is defined
            fp_plot = fp.copy()
            fp_plot[~pos] = vmin
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        vmin = float(np.nanmin(fp))
        vmax = float(np.nanmax(fp))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        fp_plot = fp
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    colors = cmap(norm(fp_plot))
    colors = np.asarray(colors, dtype=float)
    if colors.ndim == 1:
        colors = colors[None, :]
    # Apply alpha to faces
    colors[:, 3] = float(face_alpha)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Debug overlay: draw rotated border outline and polygon edges for inspection
    if debug:
        # draw rotated border outline
        rb = np.vstack([rot_border, rot_border[0]])
        ax.plot(rb[:,0], rb[:,1], lw=2, color=(0,0,0,0.6))

    if final_polys is not None:
        for level, polys in enumerate(final_polys):
            is_final = (level == len(final_polys) - 1)
            if debug:
                collection = PolyCollection(
                    polys,
                    ec=(1, 0, 0, 0.8),
                    fc='none',
                    linewidth=0.8,
                    antialiaseds=True,
                )
            elif is_final:
                collection = PolyCollection(
                    polys,
                    facecolors=colors,
                    ec=(0, 0, 0, 0.2),
                    linewidth=0.25,
                    antialiaseds=True,
                )
            else:
                collection = PolyCollection(
                    polys,
                    ec='k',
                    fc='none',
                    linewidth=2,
                    alpha=0.01,
                    antialiaseds=True,
                )
            ax.add_collection(collection)

    if show_legend:
        # Scale legend typography by image height so it's visible at large resolutions
        # Roughly 0.005 * pixels gives a readable point size at DPI=100
        cb_font = max(12, int(0.005 * img_h_pix))
        cb_tick_len = max(6, int(0.004 * img_h_pix))
        cb_tick_w   = max(1.0, 0.0015 * img_h_pix)

        try:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1.5%", pad=0.03)
            cb = plt.colorbar(mappable, cax=cax)
        except Exception:
            cb = plt.colorbar(mappable)

        # Place ticks/label on the right, use high-contrast styling
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')
        cb.set_label(
            "Population (weighted)" if pop_weights is not None else "Population count",
            rotation=90,
            fontsize=cb_font,
            color='white',
        )
        cb.ax.tick_params(labelsize=cb_font, length=cb_tick_len, width=cb_tick_w, colors='white')
        try:
            cb.outline.set_edgecolor('white')
            cb.outline.set_linewidth(cb_tick_w)
        except Exception:
            pass

        # Generate ticks from the data range and format with compact thousands
        try:
            if scale.lower() == 'log':
                ticks = np.geomspace(vmin, vmax, num=5)
            else:
                ticks = np.linspace(vmin, vmax, num=5)
            cb.set_ticks(ticks)

            def _fmt(val, _pos=None):
                # Compact but precise: 1.2k, 3.4M, or 3.4
                if val == 0:
                    return '0'
                neg = val < 0
                v = abs(val)
                if v >= 1e9:
                    s = f"{v/1e9:.2g}G"
                elif v >= 1e6:
                    s = f"{v/1e6:.2g}M"
                elif v >= 1e3:
                    s = f"{v/1e3:.2g}k"
                else:
                    s = f"{v:.2g}"
                return f"-{s}" if neg else s

            cb.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        except Exception:
            pass

        # Nudge the main axes left so cbar labels stay in-frame at big fonts
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.982, box.height])

    ax.axis('off')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', 'box')

    # Render and capture from the canvas at its true pixel size
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    im_argb = buf.reshape(h, w, 4)
    # Convert ARGB -> RGBA for Pillow
    im_rgba = im_argb[:, :, [1, 2, 3, 0]]

    if save_mask:
        # Hide background and re-render to extract overlay alpha as a mask
        bg_artist.set_visible(False)
        fig.canvas.draw()
        buf2 = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        im2_argb = buf2.reshape(h, w, 4)
        alpha_mask = im2_argb[:, :, 0]  # A channel in ARGB
        mask_path = dst_dir / f"{base}_{layer_tag}_heatmask.png"
        Image.fromarray(alpha_mask).save(mask_path)
        print(f"[pr0007] Saved {mask_path}")

    plt.close(fig)
    Image.fromarray(im_rgba).save(out_png)
    print(f"[pr0007] Saved {out_png}")
    return out_png


if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).parent
    p = argparse.ArgumentParser(description='Stage 7: hex-hex heatmap overlay')
    # FIXME --file shouldn't really be a switch.
    p.add_argument('--file', type=str, default='ggy', help="dataset id, e.g. 'btn', 'jan'")
    p.add_argument('--src', type=Path, default=script_dir / 'src')
    p.add_argument('--dst', type=Path, default=script_dir / 'src')
    p.add_argument('--bounds', type=float, nargs=4, metavar=('MIN_LAT','MIN_LON','MAX_LAT','MAX_LON'))
    p.add_argument('--preset', type=str, help="named bounds preset (per-dataset, see bounds_presets.yaml)")
    p.add_argument('--tag', type=str, help='explicit signature tag to embed in output filenames')
    p.add_argument('--layers', type=int, nargs=2, default=(4, 12),
                   help='[start, end): render layers in this half-open range; e.g. 12 13 renders only hex_layer 12')
    p.add_argument('--hex-layers', type=int, default=15,
                   help='classification depth for hex addressing (max depth passed to ugc_regions)')
    p.add_argument('--cmap', type=str, default='plasma')
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--width', type=int, default=6000)
    p.add_argument('--force', action='store_true')
    p.add_argument('--legend', default=True, action='store_true', help='show colorbar legend')
    p.add_argument('--mask', action='store_true', help='also save a transparency-only mask of the overlay')
    p.add_argument('--pixels', choices=['linear','log'], default='linear', help='color pixels')
    p.add_argument('--log-eps', type=float, default=1e-6, help='epsilon for log pixels; replaces nonpositive with this threshold')
    p.add_argument('--no-stats', action='store_true', help='suppress stats printout')
    p.add_argument('--debug', action='store_true', help='draw border outline and polygon edges for inspection')
    args = p.parse_args()

    stage(args.file, src_dir=args.src, dst_dir=args.dst,
          bounds=tuple(args.bounds) if args.bounds else None,
          preset=args.preset, tag=args.tag,
          hex_layers=args.hex_layers, layer_range=tuple(args.layers),
          cmap_name=args.cmap, face_alpha=args.alpha,
          dpi=args.dpi, img_w_pix=args.width,
          force=args.force,
          show_legend=args.legend,
          save_mask=args.mask,
          scale=args.pixels,
          log_eps=args.log_eps,
          show_stats=not args.no_stats,
          debug=args.debug)
