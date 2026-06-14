# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
H9 Layer Composition.

Provides a declarative pipeline for composing multi-layer H9 hex renders
over a polygon region.

**Core types:**

* :class:`LayerSpec` — Declares what a layer is (level, kind, source, style, labels).
* :class:`ComposedLayer` — The geometry + data produced for one :class:`LayerSpec`.
* :class:`Compositor` — Runs specs against a polygon, returns :class:`ComposedLayer` list.

**Typical usage:**

.. code-block:: python

    specs = [
        LayerSpec(level=4, kind='outline', style={'lw': 1.5, 'ec': 'grey'}),
        LayerSpec(level=6, kind='fill',    source=my_lookup, labels=True,
                  style={'alpha': 0.8}),
    ]
    compositor = Compositor(reg, b_oct, n_oct, specs)
    layers = compositor.run(polygon_n)   # polygon_n: Points in n_oct

Each :class:`ComposedLayer` then carries everything a renderer needs:
vertices in n_oct, centroids, short address labels, the common prefix, and
per-hex values from ``source``.

**Source callable contract:**

    ``source(ctrs: Points) -> NDArray[float64]``

    *ctrs* are hex centroids in b_oct (with ``components`` set to octant ids).
    The return array must have length equal to the number of hexes.

**Clipping:**

The compositor does **not** clip geometry to the polygon boundary.  The
scanline fill in :func:`~hhg9.h9.grid.poly_net_field` handles interior
sampling; edge hexes are controlled by ``threshold``.  Final polygon masking
is a rendering concern left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from hhg9 import Points
from hhg9.h9.addressing import tail_unpack_reversible, tail_key_from_reversible
from hhg9.h9.binning import hex_reduce, hex_parents, ctr_from_pars
from hhg9.h9.grid import poly_net_field, hex_verts_in_noct


# ---------------------------------------------------------------------------
# Backdrop rasteriser
# ---------------------------------------------------------------------------

def make_backdrop(
        reg,
        b_oct,
        n_oct,
        bbox_n: tuple,
        px_w: int,
        px_h: int,
        sampler: Callable[[Points], NDArray],
) -> NDArray:
    """
    Rasterise a full-resolution backdrop image in n_oct pixel space.

    Creates a ``(px_h × px_w)`` pixel grid spanning *bbox_n*, projects each
    pixel from n_oct to b_oct (filtering out off-net pixels), calls *sampler*
    for valid pixels, and assembles a ``(px_h, px_w, C)`` image array.

    The *sampler* contract is identical to :attr:`LayerSpec.source`: it
    receives :class:`~hhg9.Points` in ``b_oct`` (with components set to octant
    ids) and must return an ``(N, C)`` array — typically ``float32`` or
    ``uint8`` RGB/RGBA.  The same sampler function can therefore be used both
    as a :class:`LayerSpec` source (per-hex) and here (per-pixel).

    Args:
        reg:     Registrar used for n_oct → b_oct projection.
        b_oct:   Barycentric-octahedral domain.
        n_oct:   Octahedral-net domain currently in use.
        bbox_n:  Display bounding box ``(top, left, bottom, right)`` in n_oct
                 units — same format as the *bbox* parameter of
                 :func:`~hhg9.rendering.render.plot_hex`.
        px_w:    Output image width in pixels.
        px_h:    Output image height in pixels.
        sampler: Callable ``(pts: Points) -> NDArray`` — see above.

    Returns:
        ``(px_h, px_w, C)`` array with the same dtype as the sampler output.
        Off-net pixels are filled with zeros (transparent for RGBA float).
    """
    tp, lt, bt, rt = bbox_n
    xs = np.linspace(lt, rt, px_w)
    ys = np.linspace(bt, tp, px_h)
    gx, gy = np.meshgrid(xs, ys)
    coords_n = np.stack([gx.ravel(), gy.ravel()], axis=1)

    pts_n = Points(coords_n, domain=n_oct)
    pts_b = reg.project(pts_n, [n_oct, b_oct])
    from hhg9.base.points import OID_INVALID
    valid = pts_b.oid != OID_INVALID

    pts_b_valid = Points(
        pts_b.coords[valid],
        domain=b_oct,
        oid=pts_b.oid[valid],
    )

    colours = np.asarray(sampler(pts_b_valid))          # (V, C)
    C = colours.shape[1]
    out = np.zeros((px_h * px_w, C), dtype=colours.dtype)
    out[valid] = colours
    return out.reshape(px_h, px_w, C)


# n_oct physical scale: earth equatorial circumference in n_oct = 4√2,
# so 1 metre ≈ 4√2 / (2π R_earth) n_oct units along the equator.
_N_OCT_PER_METRE = 4.0 * np.sqrt(2.0) / (2.0 * np.pi * 6_378_137.0)  # ≈ 1.412e-7


def estimate_backdrop_size(
        bbox_n: tuple,
        target_m: float = 1000.0,
        scale: float = 1.0,
) -> tuple[int, int]:
    """
    Estimate ``(px_w, px_h)`` for :func:`make_backdrop` from a target ground resolution.

    Uses the n_oct physical scale (1 m ≈ 4√2 / (2π R_earth) ≈ 1.412e-7 n_oct
    along the equator) to convert *target_m* metres-per-pixel into an appropriate
    pixel grid size for the given *bbox_n*.

    Args:
        bbox_n:   ``(top, left, bottom, right)`` n_oct bounding box — same format
                  as the *bbox_n* parameter of :func:`make_backdrop`.
        target_m: Target ground resolution in metres per pixel (default: 1 km).
        scale:    Output scale factor relative to *target_m*. ``1.0`` matches
                  the target; ``0.5`` halves resolution; ``2.0`` doubles it.

    Returns:
        ``(px_w, px_h)`` as ints, suitable to pass directly to
        :func:`make_backdrop`.

    Note:
        The n_oct scale constant is derived from the equatorial circumference and
        is a good approximation across the whole net.  When a source GeoTIFF is
        available, prefer :func:`hhg9.geo.gdal.estimate_backdrop_size` which
        matches the actual source pixel density via the inverse GeoTransform.
    """
    tp, lt, bt, rt = bbox_n
    n_oct_per_px = _N_OCT_PER_METRE * target_m
    dx, dy = rt - lt, tp - bt
    d_pp_x = dx / n_oct_per_px
    d_pp_y = dy / n_oct_per_px
    px_w = max(1, int(np.ceil(dx / n_oct_per_px * scale)))
    px_h = max(1, int(np.ceil(dy / n_oct_per_px * scale)))
    return px_w, px_h


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """
    Declarative descriptor for one layer in a hex composition.

    Attributes:
        level:     H9 hex level (depth).
        kind:      Visual representation — ``'fill'`` or ``'outline'``.
        source:    Optional callable ``(ctrs: Points) -> NDArray`` that maps
                   hex centroids (b_oct) to per-hex values.  ``None`` means
                   geometry-only.
        lut:       Optional colour look-up table passed through to the renderer
                   unchanged.  Convention: ``ndarray (256, 3) uint8`` for
                   categorical fills; ``dict`` for label-mapped fills; ``None``
                   for continuous/shade fills.
        style:     Additional rendering hints passed through to the caller
                   (e.g. ``{'ec': 'black', 'lw': 0.5, 'label_size': 6}``).
        labels:    If ``True``, the compositor populates
                   :attr:`ComposedLayer.labels` with shortened hex addresses.
        threshold: Minimum sub-sample count for a hex to be included.
                   ``1`` keeps all candidates; higher values trim edge hexes.
        reference: True, identifies the layer as the reference layer.
    """
    level: int
    kind: str = 'fill'                          # 'fill' | 'outline'
    source: Optional[Callable[[Points], NDArray]] = None
    lut: Any = None
    style: dict = field(default_factory=dict)
    labels: Any = None
    threshold: int = 1
    reference: bool = False


@dataclass
class ComposedLayer:
    """
    Geometry and data produced for one :class:`LayerSpec`.

    Attributes:
        spec:      The :class:`LayerSpec` that produced this layer.
        verts:     Hex polygon vertices in n_oct, shape ``(H*6, 2)``.
        ctrs:      Hex centroids in n_oct, shape ``(H, 2)``.
        prefix:    Common address prefix shared by all hexes (hex string).
        labels:    Per-hex address suffixes beyond ``prefix``, shape ``(H,)``,
                   or ``None`` if :attr:`LayerSpec.labels` is ``False``.
        values:    Per-hex values from the source callable, shape ``(H,)``,
                   or ``None`` if no source was provided.
        key_tails: Per-hex geometric type key (uint8), shape ``(H,)``.
                   Encodes c2 category and net_mode; used by GIS export.
        count:     Number of hexes in this layer.
    """
    spec: LayerSpec
    verts: Points
    ctrs: Points
    prefix: str
    labels: Optional[NDArray]
    values: Optional[NDArray]
    key_tails: Optional[NDArray] = None   # (H,) uint8 geometric type key

    @property
    def count(self) -> int:
        return len(self.ctrs.coords)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class Compositor:
    """
    Runs a list of :class:`LayerSpec` against a polygon and returns
    :class:`ComposedLayer` results.

    Args:
        reg:    Registrar used to project between domains.
        b_oct:  Barycentric-octahedral domain (``b_oct``).
        n_oct:  Octahedral-net domain (e.g. ``n_oct:butterfly``).
        specs:  Ordered list of :class:`LayerSpec` describing the composition.
    """

    def __init__(self, reg, b_oct, n_oct, specs: list[LayerSpec]):
        self.reg = reg
        self.b_oct = b_oct
        self.n_oct = n_oct
        self.specs = specs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, polygon_n: Points) -> list[ComposedLayer]:
        """
        Compose all layers for *polygon_n*.

        Args:
            polygon_n: Boundary polygon as :class:`~hhg9.Points` in n_oct.
                       Only the convex hull is used for the scanline fill.

        Returns:
            One :class:`ComposedLayer` per spec (in the same order as
            :attr:`specs`), skipping any spec that yields zero hexes.
        """
        if not self.specs:
            return []

        max_level = max(s.level for s in self.specs)

        # One scanline pass at the finest level needed
        scn = poly_net_field(polygon_n, max_level)

        # Project candidates to b_oct and drop any that missed all faces
        b_pts = self.reg.project(scn, [self.n_oct, self.b_oct])
        from hhg9.base.points import OID_INVALID
        valid = b_pts.oid != OID_INVALID
        b_pts = Points(
            b_pts.coords[valid],
            domain=self.b_oct,
            oid=b_pts.oid[valid],
        )

        results = []
        for spec in self.specs:
            layer = self._run_spec(b_pts, spec)
            if layer is not None:
                results.append(layer)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_spec(self, b_pts: Points, spec: LayerSpec) -> Optional[ComposedLayer]:
        """Bin b_pts to spec.level and build one ComposedLayer."""
        hex_num, hex_v, hex_inv, _ = hex_reduce(b_pts, spec.level)
        if hex_num == 0:
            return None

        counts = np.bincount(hex_inv, minlength=hex_num)
        keep = counts >= spec.threshold
        if not keep.any():
            return None

        hex_v_k = hex_v[keep]
        n_keep = int(keep.sum())

        tails = tail_key_from_reversible(hex_v_k[:, -1])
        hex_par, hex_oid, scale = hex_parents(self.b_oct, hex_v_k, n_keep)
        xpm, xc2, _, _ = tail_unpack_reversible(hex_v_k[:, -1])

        # Centroids in b_oct (carry octant components for source callables)
        ctrs_b = ctr_from_pars(self.b_oct, hex_par, hex_oid, scale, tails)

        # Vertices and centroids in n_oct
        verts_n = hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, scale, self.n_oct)
        ctrs_n = self.reg.project(ctrs_b, [self.b_oct, self.n_oct])

        # Common prefix from path digits (tail column excluded)
        prefix, short_addrs = _extract_prefix(hex_v_k)

        labels: Optional[NDArray] = None
        match spec.labels:
            case None | False:
                pass
            case 'layer':
                idx = spec.level - len(prefix)
                labels = np.array([a[idx] for a in short_addrs])
            case _:  # True/short/anything else.
                labels = np.array(short_addrs)

        values: Optional[NDArray] = None
        if spec.source is not None:
            values = np.asarray(spec.source(ctrs_b))

        if spec.reference:
            reference = spec.level

        return ComposedLayer(
            spec=spec,
            verts=verts_n,
            ctrs=ctrs_n,
            prefix=prefix,
            labels=labels,
            values=values,
            key_tails=tails,
        )


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------

def _extract_prefix(hex_v_k: NDArray) -> tuple[str, list[str]]:
    """
    Find the common address prefix shared by all hexes in *hex_v_k*.

    Only the path-digit columns are used (the tail column is excluded — it
    encodes c2/net_mode geometry, not address).  Column layout from
    ``hex_layer(..., TailStyle.reversible)``:  ``[octant, d1, d2, ..., dN, tail]``.

    Args:
        hex_v_k: ``(H, depth+2)`` hex address array (reversible tail in last col).

    Returns:
        ``(prefix_str, short_addrs)`` where *prefix_str* is the shared hex
        string and *short_addrs* is a list of per-hex suffixes (path digits only).
    """
    path = hex_v_k[:, :-1]              # drop tail column — path digits only

    matches = np.all(path == path[0], axis=0)
    prefix_len = int(np.argmin(matches)) if not matches.all() else path.shape[1]

    prefix_str = ''.join(f'{int(d):x}' for d in path[0, :prefix_len])
    sub = path[:, prefix_len:]
    short_addrs = [''.join(f'{int(d):x}' for d in row) for row in sub]
    return prefix_str, short_addrs
