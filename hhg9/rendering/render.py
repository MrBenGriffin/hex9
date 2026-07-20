# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 hex composition renderer.

Provides :func:`plot_hex` for rendering a list of
:class:`~hhg9.rendering.composition.ComposedLayer` objects produced by
:class:`~hhg9.rendering.composition.Compositor`.

Rendering modes (driven by ``cl.spec.kind`` and ``cl.spec.lut``):

- ``'fill'`` + ``lut`` is ndarray  → categorical fill (integer codes → RGB via LUT)
- ``'fill'`` + ``lut`` is dict     → per-hex text label from a label dict
- ``'fill'`` + ``lut`` is None, ``values`` shape ``(H,)``   → shade/continuous fill
- ``'fill'`` + ``lut`` is None, ``values`` shape ``(H, C)`` → direct RGB/RGBA per-hex
- ``'outline'``                    → outline only; address labels if present

Per-layer style overrides via ``spec.style``:
    ``lw``, ``ec``, ``fc``, ``alpha``, ``label_color``, ``label_size``,
    ``shade_strength``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from hhg9.rendering.composition import ComposedLayer
from hhg9.h9.grid import hex_props


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_area(m2: float) -> str:
    """Format an area in m² to the most readable unit (km², ha, or m²)."""
    if m2 >= 1e6:
        return f'{m2 / 1e6:.3g} km²'
    if m2 >= 1e4:
        return f'{m2 / 1e4:.3g} ha'
    if m2 >= 100:
        return f'{m2:.0f} m²'
    return f'{m2:.3g} m²'


def _format_len(m: float) -> str:
    """Format a length in metres to the most readable unit (km, m, cm, µm)."""
    if m >= 1e3:
        return f'{m / 1e3:.3g} km'
    if m >= 1:
        return f'{m:.3g} m'
    if m >= 1e-3:
        return f'{m * 1e3:.3g} cm'
    return f'{m * 1e6:.3g} µm'


def _north_quarter_turns(north_dir) -> int:
    """Quarter-turns that bring ``north_dir`` to within ±45° of up.

    Returns ``k`` in 0..3; the scene is then turned ``k`` × 90° clockwise.
    Snapping to a quarter turn keeps the whole operation exact: coordinates
    become swaps and sign flips, and a raster backdrop is reindexed by
    :func:`numpy.rot90` with no resampling and no interpolation loss. The
    residual (up to 45°) is what the north arrow then reports.
    """
    nd = np.asarray(north_dir, dtype=float).ravel()
    if nd.size < 2 or not np.any(nd):
        return 0
    theta = np.degrees(np.arctan2(nd[0], nd[1]))   # 0 = up, +90 = right
    return int(np.round(theta / 90.0)) % 4


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def plot_hex(
        composed: list[ComposedLayer],
        save_path: str | None = None,
        bbox: tuple | None = None,
        draw_bbox: bool = True,
        nodata: int = 0,
        shade_strength: float = 0.65,
        title: str | None = None,
        description: str | None = None,
        show_legend: bool = True,
        show_north: bool = True,
        north_dir: NDArray | None = None,
        rotate_north: bool = False,
        backdrop: NDArray | None = None,
        dpi: int = 500,
) -> tuple:
    """Render a list of :class:`~hhg9.rendering.composition.ComposedLayer` objects.

    Args:
        composed:       Ordered list of :class:`~hhg9.rendering.composition.ComposedLayer`
                        objects to render (bottom-first).
        save_path:      If given, save the figure to this path and close it.
        bbox:           Display bounding box as ``(top, left, bottom, right)``
                        in n_oct units.  Auto-derived from the first layer if
                        omitted.
        draw_bbox:      Draw a thin rectangle at the bbox boundary.
        nodata:         Integer code treated as transparent in categorical fills.
        shade_strength: Default alpha scale for shade (continuous) fills.
        title:          Title line shown in the legend panel.
        description:    Optional description line shown below the title.
        show_legend:    Draw the legend panel (prefix, area, reference hex).
        show_north:     Draw the north arrow.
        north_dir:      North direction vector in n_oct space.  Falls back to
                        straight up ``[0, 1]`` when ``None``.
        rotate_north:   Turn the whole scene by a multiple of 90° so that
                        ``north_dir`` lands within ±45° of up.  Snapping to a
                        quarter turn keeps it exact — coordinates are swaps and
                        sign flips, and the backdrop is reindexed by
                        :func:`numpy.rot90` with no resampling.  The north
                        arrow then shows the residual tilt.  Note the plotted
                        aspect swaps for odd quarter-turns.
        backdrop:       Optional ``(px_h, px_w, C)`` image array (float 0–1 or
                        uint8) to display behind all hex layers.  Produced by
                        :func:`~hhg9.rendering.composition.make_backdrop`.
        dpi:            Figure resolution (default 500).

    Returns:
        ``(fig, ax)`` — the matplotlib Figure and Axes.  If *save_path* is
        given the figure is also saved and closed before returning.
    """
    if not composed:
        raise ValueError("plot_hex: composed must be a non-empty list")

    adj = 72 / dpi

    # --- display bounds (needed before figsize) ---
    if bbox is not None:
        tp, lt, bt, rt = bbox
    else:
        p0 = np.asarray(composed[0].verts.coords, dtype=np.float64).reshape(-1, 6, 2)
        lt = float(p0[:, :, 0].min())
        rt = float(p0[:, :, 0].max())
        bt = float(p0[:, :, 1].min())
        tp = float(p0[:, :, 1].max())

    # --- optional quarter-turn so north lands within ±45° of up ---
    # Exact by construction: a k*90° turn is a coordinate swap plus sign flips,
    # and np.rot90 reindexes the backdrop without resampling it.
    turns = _north_quarter_turns(north_dir) if rotate_north else 0
    cx, cy = (lt + rt) / 2.0, (bt + tp) / 2.0

    def _turn(xy):
        """Turn points k*90° anticlockwise about the pre-rotation bbox centre.

        ``turns`` counts how far north sits clockwise of up, so undoing it
        turns the scene the other way.
        """
        if not turns:
            return np.asarray(xy, dtype=np.float64)
        p = np.asarray(xy, dtype=np.float64)
        x, y = p[..., 0] - cx, p[..., 1] - cy
        for _ in range(turns):
            x, y = -y, x
        return np.stack([x + cx, y + cy], axis=-1)

    if turns:
        corners = _turn(np.array([[lt, bt], [rt, bt], [rt, tp], [lt, tp]]))
        lt, rt = float(corners[:, 0].min()), float(corners[:, 0].max())
        bt, tp = float(corners[:, 1].min()), float(corners[:, 1].max())
        if backdrop is not None:
            # np.rot90 turns anticlockwise for row-0-at-top, but imshow draws
            # this with origin='lower', which inverts that sense -- so the sign
            # is negated to match the anticlockwise coordinate turn. Verified by
            # round-tripping pixel centres through both paths.
            backdrop = np.rot90(backdrop, -turns)
        if north_dir is not None:
            nd = np.asarray(north_dir, dtype=float).ravel()[:2]
            x, y = nd[0], nd[1]
            for _ in range(turns):
                x, y = -y, x
            north_dir = np.array([x, y])

    bw, bh = rt - lt, tp - bt

    # Match figure size to backdrop pixels when available; otherwise fall back
    # to bbox aspect ratio at a fixed 15-inch height.
    if backdrop is not None:
        bd_h, bd_w = backdrop.shape[:2]
        fig_w, fig_h = bd_w / dpi, bd_h / dpi
    elif bh > 0:
        fig_h = 15.0
        fig_w = fig_h * (bw / bh)
    else:
        fig_w = fig_h = 15.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    pad_x = 0.02 * bw if bw > 0 else 1.0
    pad_y = 0.02 * bh if bh > 0 else 1.0
    ax.set_xlim(lt - pad_x, rt + pad_x)
    ax.set_ylim(bt - pad_y, tp + pad_y)

    # --- pixel backdrop (drawn first, behind all hex layers) ---
    if backdrop is not None:
        bd = backdrop
        if bd.dtype == np.uint8:
            bd = bd.astype(np.float64) / 255.0
        ax.imshow(bd, extent=[lt, rt, bt, tp],
                  origin='lower', aspect='auto', zorder=0, interpolation='bilinear')

    # --- hex layers ---
    for i, cl in enumerate(composed):
        polys = _turn(cl.verts.coords).reshape(-1, 6, 2)
        h = polys.shape[0]
        if h == 0:
            continue

        s = cl.spec.style
        face_colors: Any = 'none'
        antialiased = s.get('aa', True)
        edge_colors = s.get('ec', '#00000080')
        line_widths = s.get('lw', adj)

        match cl.spec.kind:
            case 'fill':
                lut = cl.spec.lut
                if isinstance(lut, dict):
                    # Values are keys into a label dict — render text inside each hex
                    for hx, v in zip(polys, cl.values):
                        if v == 0:
                            continue
                        entry = lut[v]
                        label = entry['desc'] if isinstance(entry, dict) else str(entry)
                        min_x, min_y = np.min(hx[:, 0]), np.min(hx[:, 1])
                        max_x, max_y = np.max(hx[:, 0]), np.max(hx[:, 1])
                        ax.text(min_x + (max_x - min_x) * 0.23,
                                min_y + (max_y - min_y) * 0.01,
                                label, fontsize=s.get('label_size', 18 * adj),
                                ha='left', va='bottom', zorder=100,
                                color=s.get('label_color', 'white'), clip_on=True)

                elif lut is not None:
                    # Categorical fill: integer codes → RGB via LUT array
                    line_widths = s.get('lw', adj / 3)
                    codes = np.clip(np.asarray(cl.values).astype(np.int32),
                                   0, lut.shape[0] - 1)
                    rgb = lut[codes].astype(np.float64) / 255.0
                    alpha = np.ones((h, 1), dtype=np.float64)
                    if nodata is not None:
                        alpha[codes == int(nodata)] = 0.0
                    face_colors = np.concatenate([rgb, alpha], axis=1)

                else:
                    vals = np.asarray(cl.values)
                    if vals.ndim == 2:
                        ok = np.isfinite(np.sum(vals, axis=1))
                        # Direct RGB or RGBA per-hex colour
                        # (e.g. from a KDTree image sampler or a colormap)
                        line_widths = s.get('lw', adj / 9)
                        if vals.dtype == np.uint8:
                            rgb = vals.astype(np.float64) / 255.0
                        else:
                            rgb = np.clip(vals.astype(np.float64), 0.0, 1.0)
                        if rgb.shape[1] >= 4:
                            face_colors = rgb[:, :4]
                        else:
                            a = np.full((h, 1), s.get('alpha', 1.0))
                            face_colors = np.concatenate([rgb[:, :3], a], axis=1)
                        face_colors[~ok] = [1., 1., 1., 0.0]
                        edge_colors = s.get('ec', face_colors)

                    else:
                        # Continuous / shade fill: values expected in [0, 1];
                        # rendered as a dark overlay (alpha ∝ darkness)
                        line_widths = s.get('lw', adj / 9)
                        sh = vals.astype(np.float64)
                        alpha = np.zeros(h, dtype=np.float64)
                        ok = np.isfinite(sh)
                        ss = float(np.clip(
                            s.get('shade_strength', shade_strength), 0.0, 1.0))
                        alpha[ok] = ss * (1.0 - np.clip(sh[ok], 0.0, 1.0))
                        face_colors = np.zeros((h, 4), dtype=np.float64)
                        face_colors[:, 3] = alpha

            case 'outline':
                has_labels = cl.labels is not None
                line_widths = s.get('lw', adj * (2 if has_labels else 1))
                edge_colors = s.get('ec', '#ffffff80' if has_labels else '#00000080')
                face_colors = np.zeros((h, 4), dtype=np.float64)

        ax.add_collection(PolyCollection(
            polys, linewidths=line_widths,
            facecolors=face_colors, edgecolors=edge_colors,
            zorder=10 + i, antialiased=antialiased))

        if cl.labels is not None:
            ctrs = polys.mean(axis=1)
            label_color = s.get('label_color', '#ffffff80')
            label_size = s.get('label_size', 5)
            for ctr, label in zip(ctrs, cl.labels):
                ax.text(ctr[0], ctr[1], label,
                        fontsize=label_size, ha='center', va='center',
                        color=label_color, zorder=10 + i + 100, clip_on=True)

    if bbox is not None and draw_bbox:
        ax.add_patch(Rectangle((lt, bt), bw, bh, fill=False, linewidth=1.0))

    # --- annotations: legend panel + reference hex highlight + north arrow ---
    if show_legend:
        ref_levels = [cl.spec.level for cl in composed if cl.spec.reference]
        prefix = next(
            (cl.prefix for cl in composed if cl.labels is not None and cl.prefix), '')
        lines = []
        if title:
            lines.append(title)
        if description:
            lines.append(description)
        if prefix:
            lines.append(f'Prefix: {prefix}')
        for ref_level in ref_levels[::-1]:
            props = hex_props(ref_level)
            lines.append(
                f'L{ref_level} hex ≈ {_format_area(props[0])}; {_format_len(props[1])}/side')
        if lines:
            ax.text(lt + 0.02 * bw, bt + 0.02 * bh, '\n'.join(lines),
                    fontsize=7, color='white', va='bottom', ha='left',
                    zorder=300, clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.4', fc='#00000090', ec='none'))

        # Highlight one reference hex near the bottom-left corner so readers
        # can visually identify the hex size shown in the legend.
        for ref_level in ref_levels:
            ref_cl = next((cl for cl in composed if cl.spec.level == ref_level), None)
            if ref_cl is not None:
                ref_polys = np.asarray(
                    ref_cl.verts.coords, dtype=np.float64).reshape(-1, 6, 2)
                ref_ctrs = np.asarray(ref_cl.ctrs.coords, dtype=np.float64)
                target = np.array([lt + 0.05 * bw, bt + 0.08 * bh])
                bl = int(np.argmin(np.sum((ref_ctrs - target) ** 2, axis=1)))
                ax.add_collection(PolyCollection(
                    ref_polys[bl:bl + 1], linewidths=3.0,
                    facecolors='#ffffff25', edgecolors='#ffff0060', zorder=250))

    if show_north:
        nd = (np.asarray(north_dir, dtype=float)
              if north_dir is not None else np.array([0.0, 1.0]))
        norm = np.linalg.norm(nd)
        if norm > 0:
            nd = nd / norm
        nx = rt - 0.05 * bw
        ny = tp - 0.065 * bh
        shaft = 0.03 * bh
        base = np.array([nx, ny]) - nd * shaft
        tip = np.array([nx, ny]) + nd * shaft
        ax.annotate('', xy=(tip[0], tip[1]), xytext=(base[0], base[1]),
                    arrowprops=dict(arrowstyle='->', color='white',
                                   lw=1.0, mutation_scale=10),
                    zorder=300)
        ax.text(tip[0] + nd[0] * 0.008 * bh,
                tip[1] + nd[1] * 0.008 * bh, 'N',
                fontsize=7, color='white', ha='center', va='center',
                fontweight='bold', zorder=300)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        print(f'fig saved at {save_path}')

    return fig, ax
