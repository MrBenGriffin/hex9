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

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from hhg9 import Points
from hhg9.h9.addressing import tail_unpack_reversible, tail_key_from_reversible
from hhg9.h9.binning import hex_reduce, hex_parents, ctr_from_pars, hex_from_pars
from hhg9.h9.grid import poly_net_field, hex_verts_in_noct


# One spec, binned to b_oct geometry but not yet placed in the net.  Shared by
# the fixed-n_oct and dynamic-local-net placement paths.
BinnedSpec = namedtuple(
    'BinnedSpec',
    'spec hex_v_k hex_par hex_oid scale ctrs_b prefix labels values key_tails')


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


def _coherent_mask(polys: NDArray, factor: float = 3.0) -> NDArray:
    """Boolean per-hex mask: True where a placed hex (N,6,2) has a compact
    footprint.  A hex straddling an unclosed local-net seam is placed with verts
    in far-apart octants, giving an edge far longer than the layer's typical hex
    edge — those are flagged False.  Robust to a minority of such slivers via the
    median edge length.
    """
    if len(polys) == 0:
        return np.zeros(0, dtype=bool)
    edges = np.linalg.norm(polys - polys[:, [1, 2, 3, 4, 5, 0]], axis=2)   # (N,6)
    med = float(np.median(edges))
    if med == 0:
        return np.ones(len(polys), dtype=bool)
    return edges.max(axis=1) <= factor * med


def _point_in_tri(pts: NDArray, tri: NDArray) -> NDArray:
    """Boolean mask: which *pts* (N,2) lie inside triangle *tri* (3,2). Winding-free."""
    a, b, c = tri
    v0, v1 = b - a, c - a
    v2 = pts - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    den = d00 * d11 - d01 * d01
    if den == 0:
        return np.zeros(len(pts), dtype=bool)
    d20 = v2 @ v0
    d21 = v2 @ v1
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    eps = 1e-9
    return (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)


def make_local_backdrop(
        reg,
        b_oct,
        layout: dict,
        bbox: tuple,
        px_w: int,
        px_h: int,
        sampler: Callable[[Points], NDArray],
) -> NDArray:
    """Rasterise a backdrop in a *dynamic local net* frame (companion to
    :func:`make_backdrop`, which works in the fixed n_oct net).

    Each pixel of the *bbox* grid is mapped to the octant whose placed triangle
    contains it, inverted to b_oct via that octant's ``(matrix, offset)``, and
    passed to *sampler*.  Pixels outside every placed octant are left zero.

    Args:
        reg:     Registrar (unused directly; kept for signature symmetry / future use).
        b_oct:   Barycentric-octahedral domain.
        layout:  The ``layout`` dict from :meth:`OctahedralNet.local_layout`.
        bbox:    ``(top, left, bottom, right)`` in local-net units.
        px_w:    Output width in pixels.
        px_h:    Output height in pixels.
        sampler: Callable ``(pts: Points) -> NDArray`` — same contract as
                 :attr:`LayerSpec.source` / :func:`make_backdrop`.
    """
    from hhg9.h9 import H9P, H9O
    from hhg9.base.points import OID_INVALID

    tp, lt, bt, rt = bbox
    gx, gy = np.meshgrid(np.linspace(lt, rt, px_w), np.linspace(bt, tp, px_h))
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    n = pix.shape[0]
    b_xy = np.zeros((n, 2))
    oids = np.full(n, OID_INVALID, dtype=np.uint8)

    for oid, (mtx, off) in layout.items():
        side = H9O.oid_str[oid]
        tri = np.asarray(H9P.sv[b_oct.sides[side].mode]) @ mtx + off    # placed corners
        inside = _point_in_tri(pix, tri) & (oids == OID_INVALID)
        if not inside.any():
            continue
        b_xy[inside] = (pix[inside] - off) @ np.linalg.inv(mtx)
        oids[inside] = oid

    valid = oids != OID_INVALID
    colours = np.asarray(sampler(Points(b_xy[valid], domain=b_oct, oid=oids[valid])))
    C = colours.shape[1]
    out = np.zeros((n, C), dtype=colours.dtype)
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

    def __init__(self, reg, b_oct, n_oct, specs: list[LayerSpec], local: bool = False):
        self.reg = reg
        self.b_oct = b_oct
        self.n_oct = n_oct
        self.specs = specs
        # local=True places hexes on a *dynamic local net* (n_oct.local_layout),
        # hinged from a fundamental octant so a bounded region spanning a seam is
        # laid flat tear-free.  local=False (default) uses the fixed n_oct net.
        self.local = local
        self.local_residual = None      # tree seam-closure after a local run()
        self.local_cut = None           # cut_residual: >0 ⇒ region wraps a cone vertex
        self.layout = None              # LocalLayout from the last local run()
        self.local_dropped = 0          # hexes dropped for straddling an open seam

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

        if self.local:
            return self._run_local(b_pts)

        results = []
        for spec in self.specs:
            layer = self._run_spec(b_pts, spec)
            if layer is not None:
                results.append(layer)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bin_spec(self, b_pts: Points, spec: LayerSpec) -> Optional[BinnedSpec]:
        """Bin b_pts to spec.level -> b_oct geometry + addresses (no placement)."""
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
        # Centroids in b_oct (carry octant components for source callables)
        ctrs_b = ctr_from_pars(self.b_oct, hex_par, hex_oid, scale, tails)

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

        return BinnedSpec(spec, hex_v_k, hex_par, hex_oid, scale,
                          ctrs_b, prefix, labels, values, tails)

    def _compose(self, bs: BinnedSpec, verts_n: Points, ctrs_n: Points) -> ComposedLayer:
        return ComposedLayer(
            spec=bs.spec, verts=verts_n, ctrs=ctrs_n, prefix=bs.prefix,
            labels=bs.labels, values=bs.values, key_tails=bs.key_tails)

    def _run_spec(self, b_pts: Points, spec: LayerSpec) -> Optional[ComposedLayer]:
        """Build one ComposedLayer in the fixed n_oct net."""
        bs = self._bin_spec(b_pts, spec)
        if bs is None:
            return None
        xc2, _, xpm = tail_unpack_reversible(bs.hex_v_k[:, -1])   # (c2, r_mo, p_mo)
        verts_n = hex_verts_in_noct(bs.hex_par, bs.hex_oid, xpm, xc2, bs.scale, self.n_oct)
        ctrs_n = self.reg.project(bs.ctrs_b, [self.b_oct, self.n_oct])
        return self._compose(bs, verts_n, ctrs_n)

    def _run_local(self, b_pts: Points) -> list[ComposedLayer]:
        """Build all ComposedLayers on a dynamic local net.

        Bins every spec to b_oct geometry (``hex_from_pars`` reflects seam-straddle
        verts into the neighbour octant), gathers the octants actually used, builds
        one :meth:`OctahedralNet.local_layout` for them, and places every layer
        through it so the region lies flat and seam-continuous.
        """
        binned = [b for b in (self._bin_spec(b_pts, s) for s in self.specs) if b]
        if not binned:
            return []

        geos, octs = [], set()
        for bs in binned:
            verts_b = hex_from_pars(self.b_oct, bs.hex_par, bs.hex_oid,
                                    bs.scale, bs.hex_v_k[:, -1])
            octs.update(int(o) for o in np.unique(verts_b.oid))
            geos.append((bs, verts_b))

        layout = self.n_oct.local_layout(sorted(octs))
        self.layout = layout
        self.local_residual = layout.residual
        self.local_cut = layout.cut_residual

        results, dropped = [], 0
        for bs, verts_b in geos:
            polys = self.n_oct.place(
                verts_b.coords, verts_b.oid, layout.layout).reshape(-1, 6, 2)
            ctrs = self.n_oct.place(bs.ctrs_b.coords, bs.ctrs_b.oid, layout.layout)

            # Hex-level holonomy guard: a hex straddling an UNCLOSED seam has its
            # verts placed in octants that ended up far apart in the local net, so
            # its footprint is not compact (a long sliver).  Drop those — they
            # cannot be drawn coherently in this frame.
            keep = _coherent_mask(polys)
            dropped += int((~keep).sum())
            polys, ctrs = polys[keep], ctrs[keep]

            labels = bs.labels[keep] if bs.labels is not None else None
            values = bs.values[keep] if bs.values is not None else None
            key_tails = bs.key_tails[keep] if bs.key_tails is not None else None
            results.append(ComposedLayer(
                spec=bs.spec,
                verts=Points(polys.reshape(-1, 2), domain=self.n_oct),
                ctrs=Points(ctrs, domain=self.n_oct),
                prefix=bs.prefix, labels=labels, values=values, key_tails=key_tails))
        self.local_dropped = dropped
        return results


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
