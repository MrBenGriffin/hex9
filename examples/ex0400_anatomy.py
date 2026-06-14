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
from matplotlib.patches import Polygon as MplPolygon, Patch

from hhg9.h9 import H9C, H9R, H9P
from hhg9.h9.polygon import region_grid

_OUT = Path(__file__).resolve().parent / 'output'

# Canonical colour scheme (shared across F3/F4/F7 per paper_figures.md).
MODE0 = '#bcd9ec'   # Down / V   (net_mode 0)
MODE1 = '#cfe6c3'   # Up   / Λ   (net_mode 1)
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


def _draw_layer(ax, cells, show, lw=1.0, stroke=None):
    """Draw one column's layer over a set of (origin, scale, mode) regions.

    Columns advance the assembly: t (triangles, mode two-tone) -> d (half-hexes
    filled by their 2-1 majority mode) -> x (hexagons filled solid as their two
    mode-halves: near = this region's d, far = the adjacent opposite-mode d, so
    the shifted-aperture-9 packing reads when fine hexes nest in a coarse one).
    `lw` scales all stroke widths; `stroke` (if set) overrides every boundary
    colour with a single hue (used to make the fine layer one neutral colour in
    the i+j overlay row).
    """
    t_ec = stroke or EDGE
    d_ec = stroke or DCOL
    x_ec = stroke or XCOL
    for origin, scale, mode in cells:
        if show in ('t', 'td'):
            for c2 in range(3):
                for ti in range(3):
                    tri = origin + H9P.tx[mode, c2, ti] * scale
                    _poly(ax, tri, MODECOL[_tri_mode(tri)], ec=t_ec,
                          lw=0.7 * lw, z=2)
            if show == 'td':
                for c2 in range(3):
                    _poly(ax, origin + H9P.hh[mode, c2] * scale, 'none',
                          ec=d_ec, lw=1.9 * lw, z=4)
        elif show == 'd':
            for c2 in range(3):
                dm = _d_mode(mode, c2)
                _poly(ax, origin + H9P.hh[mode, c2] * scale, MODECOL[dm],
                      ec=d_ec, lw=1.9 * lw, z=3)
        elif show in ('dx', 'x'):
            for c2 in range(3):
                dm = _d_mode(mode, c2)
                # solid fill: far (opposite-mode) half, then near half on top
                _poly(ax, _far_half(origin, scale, mode, c2), MODECOL[1 - dm],
                      ec='none', alpha=1.0, z=2)
                _poly(ax, origin + H9P.hh[mode, c2] * scale, MODECOL[dm],
                      ec='none', alpha=1.0, z=2)
                if show == 'dx':
                    _poly(ax, origin + H9P.hh[mode, c2] * scale, 'none',
                          ec=d_ec, lw=1.1 * lw, z=4)
                _poly(ax, origin + H9P.hx[mode, c2] * scale, 'none',
                      ec=x_ec, lw=1.9 * lw, z=5)


# common drawing extent (root mode 0, padded to include hexagon overhang)
def _limits(mode=0, pad=0.08):
    pts = [H9P.hx[mode, c2] for c2 in range(3)] + [H9P.sv[mode]]
    P = np.vstack(pts)
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def _coarse_boundaries(ax, cells, show):
    """Wide outlines of one column's coarse (level i) layer, hued by PARENT mode
    (so the overlay shows p_mo as well as the fine c_mo fills), drawn over the
    thin grey fine (level j) layer in the i+j overlay row. The region triangle
    is shown only for the t / t+d columns (it is noise in the d / x columns)."""
    for origin, scale, mode in cells:
        col = PMO[mode]
        if show in ('t', 'td'):
            _poly(ax, origin + H9P.sv[mode] * scale, 'none', ec=col, lw=2.6, z=9)
        if show == 't':
            for c2 in range(3):
                for ti in range(3):
                    _poly(ax, origin + H9P.tx[mode, c2, ti] * scale, 'none',
                          ec=col, lw=1.6, z=8)
        elif show in ('td', 'd'):
            for c2 in range(3):
                _poly(ax, origin + H9P.hh[mode, c2] * scale, 'none',
                      ec=col, lw=2.6, z=8)
        elif show in ('dx', 'x'):
            for c2 in range(3):
                _poly(ax, origin + H9P.hx[mode, c2] * scale, 'none',
                      ec=col, lw=2.8, z=8)


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
                    fontsize=7, color='#333', zorder=6)
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


def figure_f3(root_mode=0, depth_i=0, depth_j=1):
    """The §10.0 anatomy figure: 5 columns (t,t+d,d,d+x,x) x 3 rows
    (level i, i+j overlay, level j)."""
    cols = [('t', 't'), ('t+d', 'td'), ('d', 'd'), ('d+x', 'dx'), ('x', 'x')]
    xlim, ylim = _limits(root_mode)
    coarse = supercells(depth_i, root_mode)
    fine = supercells(depth_j, root_mode)

    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for c, (title, show) in enumerate(cols):
        # row 0 — level i (coarse)
        _draw_layer(axes[0, c], coarse, show)
        # row 1 — i+j overlay: fine layer thin + neutral grey, coarse wide + p_mo hue
        _draw_layer(axes[1, c], fine, show, lw=0.45, stroke=JCOL)
        _coarse_boundaries(axes[1, c], coarse, show)
        # row 2 — level j (fine): thinner strokes than the coarse top row
        _draw_layer(axes[2, c], fine, show, lw=0.55)

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
    figure_f7()
