"""
Part of the H9 project - Preparation 0007
Hex-hex heatmap overlay in barycentric space
Last Tested 16 August 2025 √
"""
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import Normalize, LogNorm
from hhg9 import Registrar, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.h9.polygon import enmesh
from hhg9.h9.region import xy_regions
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
from matplotlib import pyplot as plt, image, patches
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from bounds_utils import load_presets, resolve_bounds, needs_run


def heatmap(df, addresses, polys, layers, weights: dict | None = None):
    """
    Generates a hexbin heatmap from binned addresses data in barycentric space.
    If `weights` is provided (mapping address tuple -> value), those values are
    used for the final layer instead of counting rows.
    """
    if df.empty:
        print("Heatmap data is empty. Nothing to plot.")
        return [], []

    layers_to_plot = range(*layers)
    final_layer = layers[1] - 1
    polygons_to_plot: list[np.ndarray] = []
    final_pops: list[int] = []

    # Precompute row-wise address tuples once to avoid repeated pandas ops
    cols = [c for c in df.columns if c.startswith('L')]
    addr_tuples = [tuple(int(x) for x in row if pd.notna(x)) for row in df[cols].to_numpy()]

    for layer in layers_to_plot:
        last_col = f'L{layer-1}'
        next_col = f'L{layer}'
        if last_col not in df.columns:
            continue
        if next_col not in df.columns:  # deepest possible layer
            mask = df[last_col].notna().to_numpy()
        else:
            mask = (df[last_col].notna() & df[next_col].isna()).to_numpy()
        if not np.any(mask):
            continue

        # Collect polygons in this layer
        polys_for_layer = []
        # Iterate only over selected rows, but use precomputed tuples
        for i, ok in enumerate(mask):
            if not ok:
                continue
            address_tuple = addr_tuples[i][:layer]
            if layer == final_layer:
                if weights is not None:
                    final_pops.append(float(weights.get(address_tuple, 0.0)))
                else:
                    # fallback: count occurrences of this prefix in addr_tuples
                    cnt = sum(1 for t in addr_tuples if t[:layer] == address_tuple)
                    final_pops.append(float(cnt))
            poly_indices = addresses[address_tuple]
            polys_for_layer.append(polys[poly_indices])

        if polys_for_layer:
            polygons_to_plot.append(np.array(polys_for_layer))

    return polygons_to_plot, final_pops


class AddressCounter:
    """Count things"""
    def __init__(self, unique_polygons, shape_array):
        self.unique_polygons = unique_polygons
        max_depth = max(len(key) for key in unique_polygons)
        self.df = pd.DataFrame(unique_polygons.keys(),
                               columns=[f'L{i}' for i in range(max_depth)])

    def prefix_mask(self, prefix_key):
        """
        Performs a fast, vectorized match of addresses matching a prefix,
        Returning the mask of those which match the prefix.
        This returns a `pandas` mask, so use to_numpy() to convert for numpy array filtering.
        """
        mask = pd.Series(True, index=self.df.index)
        for k, prefix_val in enumerate(prefix_key):
            mask &= (self.df[f'L{k}'] == prefix_val)
        return mask

    def count_children(self, prefix_key):
        """
        Performs a fast, vectorized count of addresses matching a prefix.
        """
        mask = self.prefix_mask(prefix_key)
        return mask.sum()

    def count_by_length(self, n):
        """
        Counts the number of addresses that have a specific length n.
        """
        if n == 0:
            return 0

        if f'L{n}' not in self.df.columns:
            return 0  # Or handle as an error

        last_col = f'L{n - 1}'
        next_col = f'L{n}'

        # Create the boolean mask and return the sum
        mask = self.df[last_col].notna() & self.df[next_col].isna()
        return mask.sum()


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

    # Highest rendered layer tag (e.g., L12) for distinct outputs per run
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
    # Normalize population input: allow (layer,) weights or Nx2/3 [lon,lat,(w)]
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
    g_gcd = GeneralGCD(reg)
    c_ell = EllipsoidCartesian(reg)
    c_oct = OctahedralCartesian(reg)
    b_oct = OctahedralBarycentric(reg, c_oct)
    EllipsoidGCD(reg)
    AKOctahedralEllipsoid(reg).set_accuracy(0.01) # 1cm per-hex area.

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
        pop_bary = np.load(pop_bry_f)              # (layer,2)
        pop_cmp  = np.load(pop_cmp_f)               # (layer,3)
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

    # URIs at requested hex depth
    _, mo_pop = pop_pts_b.cm()
    uri_pop   = xy_regions(pop_pts_b.coords, mo_pop, hex_layers)

    # Build per‑cell weights for the final layer (count of children or sum of weights)
    L_final = layer_range[1] - 1
    if L_final <= 0:
        raise ValueError(f"[pr0007] invalid layer_range {layer_range}")
    weights_by_addr: dict[tuple, float] = {}
    if pop_weights is not None and len(pop_weights) == uri_pop.shape[0]:
        for i, row in enumerate(uri_pop):
            key = tuple(int(x) for x in row[:L_final])
            weights_by_addr[key] = weights_by_addr.get(key, 0.0) + float(pop_weights[i])
    else:
        for row in uri_pop:
            key = tuple(int(x) for x in row[:L_final])
            weights_by_addr[key] = weights_by_addr.get(key, 0.0) + 1.0

    # Enmesh population URIs
    shapes_mesh, ums_mesh = enmesh(uri_pop)

    # Build AddressCounter and produce polygons + final pops (weighted)
    counter = AddressCounter(ums_mesh, shapes_mesh)
    layers = layer_range
    if show_stats:
        try:
            max_depth = len(counter.df.columns)
            start, stop = layers
            start = max(1, start)
            stop = min(stop, max_depth + 1)
            layer_counts = []
            for n in range(start, stop):
                c = counter.count_by_length(n)
                layer_counts.append((n, int(c)))
            if layer_counts:
                msg = ", ".join([f"L{n}:{c}" for (n, c) in layer_counts])
                total = int(counter.df.shape[0])
                print(f"[pr0007] address counts by layer [{start},{stop}): {msg}  | total={total}")
        except Exception as e:
            print(f"[pr0007] layer-count report failed: {e}")

    layer_polys, final_pops = heatmap(counter.df, ums_mesh, shapes_mesh, layers, weights=weights_by_addr)
    if not layer_polys:
        print("[pr0007] nothing to draw (population)")
        return out_png

    lengths = [len(arr) for arr in layer_polys]
    split_indices = np.cumsum(lengths)[:-1]
    points = np.concatenate(layer_polys).reshape(-1, 2)

    # Rotate polygons into the rotated frame for overlay with the stage‑5 PNG
    squared = (points - centroid) @ matrix + centroid
    n_polys = sum(lengths)
    n_points = squared.shape[0]
    if n_polys == 0 or n_points % n_polys != 0:
        raise ValueError(f"[pr0007] cannot infer verts-per-poly: points={n_points}, polys={n_polys}")
    verts_per_poly = n_points // n_polys
    if debug:
        print(f"[pr0007] verts_per_poly={verts_per_poly} (points={n_points}, polys={n_polys})")
    final_polys = np.split(squared.reshape(-1, verts_per_poly, 2), split_indices)
    if verts_per_poly == 5:
        final_polys = [arr[:, :4, :] for arr in final_polys]

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
            layer = level + layers[0]
            if debug:
                collection = PolyCollection(
                    polys,
                    ec=(1,0,0,0.8),
                    fc='none',
                    linewidth=0.8,
                    antialiaseds=True,
                )
            elif layer == layers[1] - 1:
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
                   help='[start, end): render layers in this half-open range; e.g. 12 13 renders only layer 12')
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
          scale=args.scale,
          log_eps=args.log_eps,
          show_stats=not args.no_stats,
          debug=args.debug)
