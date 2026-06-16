# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Anatomy plotter for the §10.0 figure family (F3 / F4 / F5).

Draws the t_cell / d_cell / x_cell vocabulary directly from the H9P polygon
LUTs (metric, flat-orthographic — angles and congruence read true), so the
figures are generated, not hand-drawn.

Primitives used (all in metric xy, centred per cell, scalable):
  H9P.sv[mode]       -> 3 supercell (region/parent-triangle) vertices
  H9P.tx[mode, c2]   -> 3 t_cell triangles per c2  (9 t_cells total)
  H9P.hh[mode, c2]   -> half-hex (d_cell), 4 verts
  H9P.hx[mode, c2]   -> full hex (x_cell), 6 verts
  region_grid(L,m)   -> per-cell (path, mode, origin_xy, scale) at depth L

This first pass renders a single DIAGNOSTIC sheet of the four primitives so the
geometry can be eyeballed; the publication panels (F3 2x3, F4 children, F5 band
partition) are composed once the primitives are confirmed.

Run:
    python -m examples.ex0400_anatomy
Output: examples/output/ex0400_anatomy_diag.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon, Patch
from hhg9.h9 import H9C, H9R, H9P
from hhg9.h9.polygon import region_grid

_OUT = Path(__file__).resolve().parent / 'output'

# Canonical colour scheme (shared across F3/F4/F7 per paper_figures.md).
# Colour-blind-safe (deuteranopia/protanopia). The design keeps every encoding on
# axes that r/g vision retains (luminance + blue<->yellow), and NEVER relies on
# blue<->green (which r/g vision cannot resolve, especially at thin line widths):
#   - mode fills (F3) are NEUTRAL greys separated by LIGHTNESS, so no fill shares
#     a hue with any edge — killing both green->grey and warm-edge-on-warm-fill.
#   - t/d/x grid edges (F3) use Paul Tol's high-contrast trio (blue / gold / red),
#     engineered to stay mutually distinct for THREE categories under all CVD.
#   - F4 high-trit class fills reuse the SAME blue/gold/red hue families, darkened
#     so white digits read on them (and so F4 ties visually to F3's x-layer).
MODE0 = '#c2cbd2'   # Down / V   (net_mode 0)  mid cool grey
MODE1 = '#f1f4f6'   # Up   / Λ   (net_mode 1)  near-white grey
GCOL = {'t': '#004488', 'd': '#DDAA33', 'x': '#BB5566', }
# F4 high-trit class -> fill (x_dig // 3): 0-2, 3-5, 6-8(splits). Dark enough for
# white digits; blue/amber/red echo F3's t/d/x so the two figures read as a set.
F4_FILL = ['#1b4f7a', '#9c6a16', '#9c3550']
SPLIT = '#f6cda0'   # split / straddling cell highlight
C2COL = ['#e8a3a0', '#a0c4e8', '#b8e0a8']  # c2 = 0,1,2
EDGE = '#33444f'
MODECOL = [MODE0, MODE1]
DCOL = '#1f3a4d'    # d_cell (half-hex) boundary
XCOL = '#7a3b00'    # x_cell (hexagon) boundary
PMO = ['#15406b', '#2e6b2e']  # coarse (level i) stroke, hued by PARENT mode
JCOL = '#9aa0a6'    # fine (level j) stroke in the i+j overlay (neutral grey)


def _poly(ax, verts, fc, ec=EDGE, lw=0.8, alpha=1.0, z=2, ls='-'):
    ax.add_patch(MplPolygon(verts, closed=True, facecolor=fc, edgecolor=ec,
                            linewidth=lw, alpha=alpha, zorder=z, linestyle=ls))


def _ctr(verts):
    return np.asarray(verts).mean(axis=0)


def panel_tcells(ax, mode):
    """The 9 t_cells of a parent, coloured by each cell's own net_mode."""
    ax.set_title(f't_cells (parent mode {mode})')
    _poly(ax, H9P.sv[mode], 'none', ec='k', lw=1.6, z=5)
    for c2 in range(3):
        for ti in range(3):
            tri = H9P.tx[mode, c2, ti]
            # a t_cell's own orientation: up or down by its winding
            sub_mode = _tri_mode(tri)
            _poly(ax, tri, MODECOL[sub_mode], lw=0.7)
            cx, cy = _ctr(tri)
            ax.text(cx, cy, f'{c2}', ha='center', va='center', fontsize=6, color=EDGE)


def panel_dcells(ax, mode):
    """The 3 d_cells (half-hexes), coloured by c2."""
    ax.set_title(f'd_cells (parent mode {mode})')
    _poly(ax, H9P.sv[mode], 'none', ec='k', lw=1.6, z=5)
    for c2 in range(3):
        hh = H9P.hh[mode, c2]
        _poly(ax, hh, C2COL[c2], lw=0.9, alpha=0.85)
        cx, cy = _ctr(hh)
        ax.text(cx, cy, f'c2={c2}', ha='center', va='center', fontsize=7, color=EDGE)


def panel_xcells(ax, mode):
    """The x_cells (full hexes) emerging across the parent boundary."""
    ax.set_title(f'x_cells (parent mode {mode})')
    _poly(ax, H9P.sv[mode], 'none', ec='0.6', lw=1.2, z=1, ls='--')
    for c2 in range(3):
        hx = H9P.hx[mode, c2]
        _poly(ax, hx, C2COL[c2], lw=0.9, alpha=0.55)
        cx, cy = _ctr(hx)
        ax.text(cx, cy, f'{c2}', ha='center', va='center', fontsize=7, color=EDGE)


def panel_children(ax, mode):
    """One parent region subdivided ×9 (level i -> i+1), coloured by child mode."""
    ax.set_title(f'9 children (parent mode {mode})')
    _poly(ax, H9P.sv[mode], 'none', ec='k', lw=1.6, z=5)
    # region_grid(levels) subdivides `levels` times BEYOND the initial 9 children,
    # so the direct children are levels=0.
    for path, cmode, origin, scale in region_grid(0, mode):
        tri = origin + H9P.sv[cmode] * scale
        _poly(ax, tri, MODECOL[cmode], lw=0.7)
        cx, cy = _ctr(tri)
        cell_id = int(np.atleast_1d(path)[-1])
        ax.text(cx, cy, f'{cell_id:02x}', ha='center', va='center',
                fontsize=6, color=EDGE)


def _tri_mode(tri):
    """Up/Λ (1) if one apex sits above the base (2 verts low), else down/V (0)."""
    ys = np.asarray(tri)[:, 1]
    return 1 if (ys > ys.mean()).sum() == 1 else 0


def supercells(depth, mode):
    """List of (origin_xy, scale, mode) regions at a given refinement depth.

    depth 0 = the single root region; depth d = region_grid(d-1) reinterpreted
    as the regions that each carry 9 t / 3 d / 3 x at the next level down.
    """
    if depth == 0:
        return [(np.zeros(2), 1.0, mode)]
    out = []
    for path, cmode, origin, scale in region_grid(depth - 1, mode):
        out.append((np.asarray(origin, dtype=float), float(scale), int(cmode)))
    return out


def _d_mode(mode, c2):
    """A d_cell's mode is the 2-1 majority of its three t_cell modes."""
    s = sum(_tri_mode(H9P.tx[mode, c2, ti]) for ti in range(3))
    return 1 if s >= 2 else 0


def _far_half(origin, scale, mode, c2):
    """The opposite-mode half of a hexagon (the d_cell from the adjacent
    region). hx verts are CW; the long diagonal splits v0..v3 (= the near
    half-hex hh) from v3,v4,v5,v0 (the far half)."""
    h = origin + H9P.hx[mode, c2] * scale
    return h[[3, 4, 5, 0]]


def _draw_layer(ax, cells, show, lw=1.0, layer=0):
    """Draw one column's layer over a set of (origin, scale, mode) regions.

    Columns advance the assembly: t (triangles, mode two-tone) -> d (half-hexes
    filled by their 2-1 majority mode) -> x (hexagons filled solid as their two
    mode-halves: near = this region's d, far = the adjacent opposite-mode d, so
    the shifted-aperture-9 packing reads when fine hexes nest in a coarse one).
    `lw` scales all stroke widths; `stroke` (if set) overrides every boundary
    colour with a single hue (used to make the fine layer one neutral colour in
    the i+j overlay row).
    """
    alpha = 1.0 if layer == 0 else 0.5
    alpha = 1.0
    ec = GCOL[show]
    for origin, scale, mode in cells:
        if show == 't':
            for c2 in range(3):
                for ti in range(3):
                    tri = origin + H9P.tx[mode, c2, ti] * scale
                    fc = MODECOL[_tri_mode(tri)] if layer == 0 else 'none'
                    _poly(ax, tri, fc, ec=ec, lw=lw, z=layer+1, alpha=alpha)
        elif show == 'd':
            for c2 in range(3):
                dm = _d_mode(mode, c2)
                fc = MODECOL[dm] if layer == 0 else 'none'
                _poly(ax, origin + H9P.hh[mode, c2] * scale, fc, ec=ec, lw=lw, z=layer+1, alpha=alpha)
        elif show == 'x':
            for c2 in range(3):
                dm = _d_mode(mode, c2)
                fc = MODECOL[dm] if layer == 0 else 'none'
                oc = MODECOL[1-dm]
                _poly(ax, origin + H9P.hx[mode, c2] * scale, fc=oc, ec='none', lw=lw, alpha=alpha, z=layer-10)
                _poly(ax, origin + H9P.hh[mode, c2] * scale, fc=fc, ec='none', lw=0, z=layer+1, alpha=alpha)
                _poly(ax, origin + H9P.hx[mode, c2] * scale, fc='none', ec=ec, lw=lw, alpha=alpha, z=layer+2)


# common drawing extent (root mode 0, padded to include hexagon overhang)
def _limits(mode=0, pad=0.08):
    pts = [H9P.hx[mode, c2] for c2 in range(3)] + [H9P.sv[mode]]
    P = np.vstack(pts)
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)

# --- F7 adjacency -----------------------------------------------------------
# Ground-truth neighbour LUT (region.py): [region, super_mode, c2] -> [neighbour,
# neighbour_parent_mode]. A child's adjacency class is the number of its three
# edge-neighbours that belong to a DIFFERENT parent (parent_mode != super_mode):
# 0 = interior, 1 = mid-edge, 2 = vertex. Per mode this splits the 9 children 3-3-3.
from hhg9.h9.region import _neighbour_builder as _nbuild

_NLUT = _nbuild(0x60, 0x0F)
ADJ_NAMES = ['interior', 'mid-edge', 'vertex']
# green-free palette (mode uses blue/green in F3; high-trit uses green in F4)
ADJ_COL = ['#5b8fb0', '#e0a13c', '#c0504d']  # interior / mid-edge / vertex


def _adj_class(cell_id, mode):
    """Adjacency class from the ground-truth neighbour LUT: number of a child's
    three edge-neighbours that belong to a different parent (0/1/2)."""
    return int(sum(int(_NLUT[cell_id, mode, c2, 1]) != mode for c2 in range(3)))


INT_COL = '#2c6fbb'   # internal edge (shared with a sibling in the SAME parent)
EXT_COL = '#c0392b'   # external edge (cross-parent — lies on the parent boundary)


def _on_seg(p, a, b, eps=1e-7):
    """True if point p lies on segment ab."""
    ab = b - a
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    L2 = ab @ ab
    if cross * cross > (eps * eps) * L2:
        return False
    t = (ap @ ab) / L2
    return -eps <= t <= 1 + eps


def _edge_external(e0, e1, parent_verts):
    """A child edge is cross-parent iff it lies on a parent-triangle side."""
    for k in range(3):
        a, b = parent_verts[k], parent_verts[(k + 1) % 3]
        if _on_seg(e0, a, b) and _on_seg(e1, a, b):
            return True
    return False


def figure_f7(root_mode=0):
    """F7 — sibling adjacency in the ex0110 idiom. Each child t_cell's edges are
    coloured blue (internal: shared with a sibling in the SAME parent) or red
    (external: cross-parent — it lies on the parent boundary). A child's class
    (interior / mid-edge / vertex) is then just its red-edge count (0/1/2). The
    edge test is geometric (edge on parent side) and is asserted equal to the
    ground-truth `_neighbour_builder` class. The central hex cluster (x_dig 0 for
    mode 0, 1 for mode 1) is called out."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, mode in zip(axes, (0, 1)):
        xlim, ylim = _limits(mode)
        parent = H9P.sv[mode]
        for path, cmode, origin, scale in region_grid(0, mode):
            cid = int(np.atleast_1d(path)[-1])
            tri = origin + H9P.sv[cmode] * scale
            # faint class fill for quick reading
            cls = _adj_class(cid, mode)
            _poly(ax, tri, ADJ_COL[cls], ec='none', alpha=0.18, z=1)
            # colour each edge internal/external
            ext_count = 0
            for v in range(3):
                e0, e1 = tri[v], tri[(v + 1) % 3]
                ext = _edge_external(e0, e1, parent)
                ext_count += ext
                ax.plot([e0[0], e1[0]], [e0[1], e1[1]],
                        color=EXT_COL if ext else INT_COL,
                        lw=3.0 if ext else 1.3, zorder=5,
                        solid_capstyle='round')
            assert ext_count == cls, \
                f'edge/LUT mismatch cell {cid:02x} mode {mode}: {ext_count}!={cls}'
            cx, cy = _ctr(tri)
            ax.text(cx, cy, f'{cid:02x}', ha='center', va='center',
                    fontsize=9, color='#333', zorder=6)
        # central hex cluster: the x_dig 0 (mode 0) / 1 (mode 1) hexes
        for c2 in range(3):
            _poly(ax, H9P.hx[mode, c2], 'none', ec='#444', lw=1.2, z=4)
            cx, cy = _ctr(H9P.hx[mode, c2])
            ax.text(cx, cy, str(mode), ha='center', va='center', fontsize=15,
                    fontweight='bold', color='#111', zorder=9)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'parent mode {mode}  —  central cluster = x_dig {mode}',
                     fontsize=11)
    handles = [
        plt.Line2D([], [], color=INT_COL, lw=2, label='internal edge (same parent)'),
        plt.Line2D([], [], color=EXT_COL, lw=3, label='external edge (cross-parent)'),
    ] + [Patch(facecolor=ADJ_COL[i], alpha=0.18, label=ADJ_NAMES[i])
         for i in range(3)]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('F7 — sibling adjacency: internal vs cross-parent edges '
                 '(class = red-edge count)', fontsize=13)
    fig.tight_layout()
    out = str(_OUT / 'ex0400_anatomy_f7.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def _xdig_lut(parent_mode):
    """{cell_id -> [x_dig for c2=0,1,2]} for one parent net_mode, built straight
    from the ground-truth `_m_c2_hx_v2025` (via `rid2cell`), so the F4 digits are
    generated, not transcribed. Verified equal to the LUT's documented cells."""
    from hhg9.h9.addressing import _m_c2_hx_v2025, H9_RA
    rid2cell = H9_RA.rid2cell
    out = {}
    for c2_entries in _m_c2_hx_v2025[parent_mode]:
        for region, hexdigs in c2_entries:
            out[int(rid2cell[region])] = list(hexdigs)
    return out


# text halos so labels read on any (dark) fill — keeps the big-white/small-dark
# visual language but makes both legible over the dark high-trit faces.
_HALO_W = [pe.withStroke(linewidth=2.4, foreground='#1a1a1a')]   # white text on dark
_HALO_D = [pe.withStroke(linewidth=2.0, foreground='white')]     # dark text on dark


def _limits_both(pad=0.03):
    """Drawing extent covering BOTH concentric parents (the hexagram overlay)."""
    pts = np.vstack([H9P.hx[m, c2] for m in (0, 1) for c2 in range(3)]
                    + [H9P.sv[m] for m in (0, 1)])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def _f4_tiling(ax, mode, *, ghost, skip_ids=frozenset()):
    """Draw one parent's x_dig hexagon tiling, concentric at the origin.

    ghost=True  -> the *other* mode behind the foreground: faded fills, a faint
                   c2 outline, NO x_digs (they change with parent context, so
                   they would only confuse), and c-cell ids only for regions NOT
                   already labelled by the foreground (`skip_ids`) — so the 3
                   c-regions unique to this mode pop out of the background.
    ghost=False -> the foreground: solid fills, white x_digs, bold c2 spokes,
                   and every c-cell id."""
    lut = _xdig_lut(mode)
    fa = 0.30 if ghost else 1.0
    for path, cmode, origin, scale in region_grid(0, mode):
        cell = int(np.atleast_1d(path)[-1])
        xdigs = lut[cell]
        for c2 in range(3):
            xd = xdigs[c2]
            # x_dig 0-5 are full hexagons; 6/7/8 are the half-hex 'wings' that
            # straddle the parent boundary, so draw those as their d_cell half.
            base = H9P.hh if xd >= 6 else H9P.hx
            hexv = origin + base[cmode, c2] * scale
            _poly(ax, hexv, F4_FILL[xd // 3], ec='none' if ghost else '#222',
                  lw=0.8, z=1 if ghost else 2, alpha=fa)
            if not ghost:
                cx, cy = _ctr(hexv)
                ax.text(cx, cy, str(xd), ha='center', va='center', fontsize=14,
                        fontweight='bold', color='white', zorder=6,
                        path_effects=_HALO_W)
        # c_cell / region id at the region (t_cell) centre = shared vertex of its
        # 3 hexes. Drawn once per region; ghost skips ids the foreground owns.
        if ghost and cell in skip_ids:
            continue
        rx, ry = origin
        ax.text(rx, ry, f'{cell:02X}', ha='center', va='center', fontsize=8,
                color='#111', zorder=7, path_effects=_HALO_D)
    # c2 (half-hex) structure: faint grey for the ghost parent, bold black spokes
    # for the foreground (kept under the labels at z=4).
    for c2 in range(3):
        _poly(ax, H9P.hh[mode, c2], 'none',
              ec='#9aa0a6' if ghost else 'k',
              lw=1.4 if ghost else 2.4, z=1 if ghost else 4, alpha=fa)


def figure_f4():
    """F4 — the x-layer: the 9 x_dig (0-8) hexagon children of a parent, in the
    high-trit colouring (class = x_dig // 3: 0-2 / 3-5 / 6-8 splits). Each panel
    overlays BOTH parents concentrically: the foreground mode solid (with x_digs
    + c-cell ids), the other mode ghosted behind (faded, no x_digs). The two
    triangles form a hexagram whose central hexagon is the 6 c-regions shared by
    both modes (identical cells), while each mode contributes 3 unique outer
    c-regions — those ghost ids pop out of the background. The same core hexes
    carry different x_digs under each parent (e.g. 0/6/3 vs 1/7/5 around 0x35),
    which is why the ghost's digits are suppressed. Geometry from H9P; digits
    from the ground-truth `_m_c2_hx_v2025`."""
    fg_cells = {m: {int(np.atleast_1d(p)[-1]) for p, _, _, _ in region_grid(0, m)}
                for m in (0, 1)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2),
                             gridspec_kw={'wspace': 0.03})
    for ax, mode in zip(axes, (0, 1)):
        _f4_tiling(ax, 1 - mode, ghost=True, skip_ids=fg_cells[mode])
        _f4_tiling(ax, mode, ghost=False)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'parent mode {mode} (foreground)', fontsize=11)
    (x0, x1), (y0, y1) = _limits_both()
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    handles = [Patch(facecolor=F4_FILL[i], edgecolor='#222',
                     label=f'x_dig {3*i}-{3*i+2}' + ('  (splits)' if i == 2 else ''))
               for i in range(3)]
    handles.append(Patch(facecolor=F4_FILL[0], edgecolor='none', alpha=0.30,
                         label='other mode (ghost)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('F4 — the x-layer: shared c-regions across both parent modes',
                 fontsize=13)
    # tight_layout is incompatible with equal-aspect axes (and would reset wspace);
    # set margins explicitly and let bbox_inches='tight' crop the outer border.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.03)
    out = str(_OUT / 'ex0400_anatomy_f4.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def figure_f3(root_mode=0, depth_i=0, depth_j=1):
    """The §10.0 anatomy figure: 5 columns (t,t+d,d,d+x,x) x 3 rows
    (level i, i+j overlay, level j)."""
    cols = [('t', 't'), ('t+d', 'td'), ('d', 'd'), ('d+x', 'dx'), ('x', 'x')]
    xlim, ylim = _limits(root_mode)
    d_i = supercells(depth_i, root_mode)
    d_j = supercells(depth_j, root_mode)
    lw = [2.4, 1.2]
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for c, (title, show) in enumerate(cols):
        for layer, g in enumerate(show):
            # row 0 — level i (d_i)
            _draw_layer(axes[0, c], d_i, g, lw=lw[0], layer=2*layer)
            # row 1 — level i + level j
            _draw_layer(axes[1, c], d_i, g, lw=lw[0], layer=2*layer+1)
            _draw_layer(axes[1, c], d_j, g, lw=lw[1], layer=2*layer)
            # row 2 — level j
            _draw_layer(axes[2, c], d_j, g, lw=lw[1], layer=2*layer)

    # Symbolic levels only — the hierarchy is fractal, so no absolute depth is
    # implied: i is any level, j is one (or more) refinements finer.
    row_lbl = ['level i', 'level i + level j', 'level j']
    for r in range(3):
        for c, (title, _) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.axis('off')
            if r == 0:
                ax.set_title(title, fontsize=12)
        axes[r, 0].text(-0.04, 0.5, row_lbl[r], transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10)
    fig.suptitle('F3 — Hex9 anatomy: t → d → x (assembly) × refinement',
                 fontsize=15)
    fig.tight_layout()
    out = str(_OUT / 'ex0400_anatomy_f3.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, (fn, name) in enumerate([
        (panel_tcells, 't'), (panel_dcells, 'd'),
        (panel_xcells, 'x'), (panel_children, 'kids')]):
        for row, mode in enumerate((0, 1)):
            ax = axes[row, col]
            fn(ax, mode)
            ax.set_aspect('equal')
            ax.autoscale_view()
            ax.axis('off')
    fig.suptitle('Hex9 anatomy primitives — diagnostic sheet', fontsize=14)
    fig.tight_layout()
    out = str(_OUT / 'ex0400_anatomy_diag.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
    figure_f3()
    figure_f4()
    figure_f7()
