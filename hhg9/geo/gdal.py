# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
GDAL/OGR adapters for Hex9.

Provides a lightweight bridge between GDAL raster datasets and the Hex9
domain/projection system:

* :class:`Wkt`                — a :class:`~hhg9.base.Domain` backed by a WKT
                                spatial reference (typically from a GeoTIFF).
* :class:`Wkt_4978`           — a :class:`~hhg9.base.Projection` connecting
                                EPSG:4978 (``c_ell``, WGS84 ECEF XYZ) to any
                                :class:`Wkt` domain.
* :func:`sample_gdal`         — sample a single-band GDAL raster at arbitrary
                                projected coordinates.
* :func:`wkt_geotiff_meta`    — open a GeoTIFF, extract bbox and optionally
                                register its CRS — no pixel data loaded.
* :func:`estimate_backdrop_size` — estimate ``(px_w, px_h)`` for
                                :func:`make_backdrop` to match source TIF
                                resolution without over/under-sampling.
* :func:`make_geotiff_sampler` — build a :func:`make_backdrop`-compatible
                                sampler from an open GDAL dataset.
* :func:`load_wkt_geotiff`    — open a GeoTIFF and load the full pixel array
                                (calls :func:`wkt_geotiff_meta` internally).

Requires ``osgeo`` (GDAL ≥ 3).  Install via::

    conda install -c conda-forge gdal
"""

import numpy as np
from numpy.typing import NDArray

try:
    from osgeo import gdal, osr
except ImportError as _e:
    raise ImportError(
        "hhg9.geo.gdal requires osgeo (GDAL/OGR). "
        "Install via:  conda install -c conda-forge gdal"
    ) from _e

from hhg9 import Points, Registrar
from hhg9.base import Domain, Projection


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class Wkt(Domain):
    """
    A Hex9 domain backed by an OGR/OSR WKT spatial reference.

    Typically constructed by passing the WKT string returned by
    ``gdal.Dataset.GetProjection()``.

    Args:
        registrar: The :class:`~hhg9.Registrar` to register this domain with.
        name:      Unique domain name (default ``'g_wkt'``).
        _wkt:      WKT projection string.
    """

    def valid(self, arr) -> NDArray:
        return super().valid(arr)

    def __init__(self, registrar, name: str = 'g_wkt', _wkt: str = ''):
        self.sr = osr.SpatialReference()
        self.sr.ImportFromWkt(_wkt)
        axes = self.sr.GetAxesCount()
        super().__init__(registrar, name, axes)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class Wkt_4978(Projection):
    """
    Bidirectional projection between EPSG:4978 (``c_ell``) and a
    :class:`Wkt` domain.

    EPSG:4978 is WGS84 geocentric Cartesian XYZ (metres), which is the
    ``c_ell`` domain in Hex9.  This projection lets the registrar chain
    from any Hex9 domain all the way out to an arbitrary GeoTIFF CRS.

    Args:
        registrar: The :class:`~hhg9.Registrar` to register this projection with.
        wkt:       The target :class:`Wkt` domain.
    """

    def __init__(self, registrar: Registrar, wkt: Wkt):
        wkt_name = wkt.name
        super().__init__(registrar, f'ell_{wkt_name[-3:]}', 'c_ell', wkt_name)
        self.esr = osr.SpatialReference()
        self.esr.ImportFromEPSG(4978)
        self.wsr = wkt.sr
        # Force traditional GIS axis order (easting, northing) = (lon, lat) on
        # the WKT side so TransformPoints output matches GeoTransform convention
        # (coords[:,0] = easting/lon).  Without this, GDAL 3 may use the CRS
        # native order (e.g. lat/lon for EPSG:4326), causing a 90° swap when
        # coords are fed to the inverse GeoTransform in sample_gdal /
        # make_geotiff_sampler.
        wsr_trad = self.wsr.Clone()
        wsr_trad.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self._fwd = osr.CoordinateTransformation(self.esr, wsr_trad)
        self._bak = osr.CoordinateTransformation(wsr_trad, self.esr)

    def forward(self, pts: Points) -> Points:
        """ECEF (c_ell) → Wkt CRS."""
        res = np.atleast_2d(self._fwd.TransformPoints(pts.coords))
        return Points(coords=res[:, :self.fwd_cs.axes], domain=self.fwd_cs,
                      samples=pts.samples)

    def backward(self, pts: Points) -> Points:
        """Wkt CRS → ECEF (c_ell)."""
        res = np.atleast_2d(self._bak.TransformPoints(pts.coords))
        return Points(coords=res[:, :self.rev_cs.axes], domain=self.rev_cs,
                      samples=pts.samples)


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------

def estimate_backdrop_size(
        ds,
        bbox_n: tuple,
        reg,
        chain_n_to_wkt: list,
        scale: float = 1.0,
        grid: int = 4,
) -> tuple:
    """
    Estimate ``(px_w, px_h)`` for :func:`~hhg9.rendering.composition.make_backdrop`
    that matches the source GeoTIFF resolution.

    Projects a ``grid × grid`` lattice of points from n_oct into the TIF's CRS
    and measures the local n_oct → WKT scale at each cell.  The *maximum*
    density across all cells is used so that no part of the n_oct bbox is
    under-sampled, even where the projection compresses the most.

    Args:
        ds:              Open :class:`gdal.Dataset` (from :func:`wkt_geotiff_meta`).
        bbox_n:          ``(tp, lt, bt, rt)`` n_oct bounding box.
        reg:             :class:`~hhg9.Registrar`.
        chain_n_to_wkt:  Domain chain from n_oct to the TIF's :class:`Wkt`
                         domain, e.g. ``[n_oct, b_oct, c_oct, c_ell, wkt]``.
        scale:           Output scale factor relative to source resolution.
                         ``1.0`` matches source; ``0.5`` halves it; ``2.0``
                         doubles it.
        grid:            Number of sample rows/cols in the lattice (default 4).

    Returns:
        ``(px_w, px_h)`` as ints — suitable to pass directly to
        :func:`~hhg9.rendering.composition.make_backdrop`.

    Example::

        ds, bbox_wkt, wkt = wkt_geotiff_meta('landcover.tif', reg=rg, name='g_lc')
        px_w, px_h = estimate_backdrop_size(ds, bbox_n, rg,
                                            [n_oct, b_oct, c_oct, c_ell, wkt])
        backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h,
                                 make_geotiff_sampler(ds, rg, [b_oct, c_oct, c_ell, wkt]))
    """
    tp, lt, bt, rt = bbox_n

    # Project a (grid+1)×(grid+1) lattice from n_oct to WKT
    xs = np.linspace(lt, rt, grid + 1)
    ys = np.linspace(bt, tp, grid + 1)
    gx, gy = np.meshgrid(xs, ys)
    pts_n = Points(
        np.stack([gx.ravel(), gy.ravel()], axis=1),
        domain=chain_n_to_wkt[0],
    )
    wkt_pts = reg.project(pts_n, chain_n_to_wkt)
    coords_wkt = wkt_pts.coords

    if isinstance(ds, gdal.Dataset):
        rx, ry = ds.RasterXSize, ds.RasterYSize
        inv_gt = gdal.InvGeoTransform(ds.GetGeoTransform())
        if not inv_gt:
            raise RuntimeError("estimate_backdrop_size: failed to invert GeoTransform")
        c0, c1, c2, c3, c4, c5 = inv_gt
        # Map WKT coords → source pixel coordinates via inverse GeoTransform.
        # WKT axis-ordering differences (GDAL 3 may return lat/lon or lon/lat).
        src_x = c0 + coords_wkt[:, 0] * c1 + coords_wkt[:, 1] * c2
        src_y = c3 + coords_wkt[:, 0] * c4 + coords_wkt[:, 1] * c5
    else:
        p_ppx = reg.project(wkt_pts, [wkt_pts.domain, 'n_plt'])
        rx, ry = ds
        src_x = coords_wkt[:, 0]
        src_y = coords_wkt[:, 1]

    # Discard off-net and out-of-bounds points
    valid = (
        np.isfinite(src_x) & np.isfinite(src_y) &
        (src_x >= 0) & (src_x <= rx) &
        (src_y >= 0) & (src_y <= ry)
    )
    if not valid.any():
        raise ValueError(
            "estimate_backdrop_size: no grid points project within the source TIF"
        )

    # Source pixel range spanned by the n_oct bbox → directly gives output size
    px_w = max(1, int(np.ceil((src_x[valid].max() - src_x[valid].min()) * scale)))
    px_h = max(1, int(np.ceil((src_y[valid].max() - src_y[valid].min()) * scale)))
    return px_w, px_h


def _open_ds(path: str, caller: str):
    """Open a GDAL dataset or raise FileNotFoundError."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"{caller}: could not open {path!r}")
    return ds


def _bbox_from_ds(ds) -> tuple:
    """Return ``(top, left, bottom, right)`` in the dataset's own CRS."""
    gt = ds.GetGeoTransform()
    return (
        gt[3],
        gt[0],
        gt[3] + gt[5] * ds.RasterYSize,
        gt[0] + gt[1] * ds.RasterXSize,
    )


def _register_wkt(ds, reg, name: str):
    """Create and register Wkt + Wkt_4978 from an open dataset."""
    wkt_dom = Wkt(reg, name, ds.GetProjection())
    Wkt_4978(reg, wkt_dom)
    return wkt_dom


def wkt_geotiff_meta(
        path: str,
        reg=None,
        name: str = 'g_wkt',
) -> tuple:
    """
    Open a georeferenced GeoTIFF and extract its spatial metadata — no pixel
    data is loaded into memory.

    Use this when you intend to stream pixels via :func:`sample_gdal` or
    :func:`make_geotiff_sampler` rather than loading the full array.

    Args:
        path: GeoTIFF file path.
        reg:  Optional :class:`~hhg9.Registrar`.  If provided, registers a
              :class:`Wkt` domain and :class:`Wkt_4978` projection so the
              TIF's CRS joins the H9 ``reg.project()`` pipeline.
        name: Domain name for registration (default ``'g_wkt'``).  Use a
              distinct name per file when loading multiple TIFs.

    Returns:
        ``(ds, bbox, domain_or_wkt)`` where:

        * *ds*     — open :class:`gdal.Dataset` (keep alive for sampling).
        * *bbox*   — ``(top, left, bottom, right)`` in the raster's CRS units.
        * *domain_or_wkt* — :class:`Wkt` domain if *reg* was supplied,
          otherwise the raw WKT projection string.

    Example::

        ds, bbox, wkt = wkt_geotiff_meta('landcover.tif', reg=rg, name='g_lc')
        sampler = make_geotiff_sampler(ds, rg, [b_oct, c_oct, c_ell, wkt])
        backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, sampler)
    """
    ds = _open_ds(path, 'wkt_geotiff_meta')
    bbox = _bbox_from_ds(ds)
    if reg is not None:
        return ds, bbox, _register_wkt(ds, reg, name)
    return ds, bbox, ds.GetProjection()


def make_geotiff_sampler(ds, reg, chain: list):
    """
    Build a :func:`~hhg9.rendering.composition.make_backdrop`-compatible
    sampler that reads all bands from an open GDAL dataset.

    The returned callable accepts :class:`~hhg9.Points` in the *first* domain
    of *chain* and returns an ``(N, C)`` float32 array in [0, 1].

    Args:
        ds:    Open :class:`gdal.Dataset` (from :func:`wkt_geotiff_meta`).
        reg:   :class:`~hhg9.Registrar` used to project coordinates.
        chain: Domain chain ending at the TIF's :class:`Wkt` domain,
               e.g. ``[b_oct, c_oct, c_ell, wkt]``.

    Returns:
        Callable ``(ctrs: Points) -> NDArray`` compatible with
        :func:`~hhg9.rendering.composition.make_backdrop` and
        :class:`~hhg9.rendering.composition.LayerSpec` ``source``.

    Example::

        ds, bbox, wkt = wkt_geotiff_meta('landcover.tif', reg=rg, name='g_lc')
        sampler = make_geotiff_sampler(ds, rg, [b_oct, c_oct, c_ell, wkt])
        backdrop = make_backdrop(rg, b_oct, n_oct, bbox_n, px_w, px_h, sampler)
    """
    inv_gt = gdal.InvGeoTransform(ds.GetGeoTransform())
    if not inv_gt:
        raise RuntimeError("make_geotiff_sampler: failed to invert GeoTransform")
    c0, c1, c2, c3, c4, c5 = inv_gt
    n_bands = ds.RasterCount
    scale = 1.0 / 255.0 if ds.GetRasterBand(1).DataType == gdal.GDT_Byte else 1.0

    def sampler(ctrs) -> NDArray:
        coords = reg.project(ctrs, chain).coords
        px = c0 + coords[:, 0] * c1 + coords[:, 1] * c2
        py = c3 + coords[:, 0] * c4 + coords[:, 1] * c5
        valid = (px >= 0) & (px < ds.RasterXSize) & (py >= 0) & (py < ds.RasterYSize)
        cols = np.clip(np.floor(px).astype(np.int64), 0, ds.RasterXSize - 1)
        rows = np.clip(np.floor(py).astype(np.int64), 0, ds.RasterYSize - 1)
        out = np.zeros((len(coords), n_bands), dtype=np.float32)
        for b in range(n_bands):
            data = ds.GetRasterBand(b + 1).ReadAsArray()
            out[:, b] = data[rows, cols] * scale
        out[~valid] = 0.0
        return out

    return sampler


def load_wkt_geotiff(
        path: str,
        reg=None,
        name: str = 'g_wkt',
) -> tuple:
    """
    Load a georeferenced GeoTIFF into memory and optionally register its CRS.

    Convenience wrapper around :func:`wkt_geotiff_meta` that also reads all
    pixel data into a ``(H, W, C)`` uint8 array.  Use :func:`wkt_geotiff_meta`
    instead when you plan to stream pixels via :func:`make_geotiff_sampler`.

    Args:
        path: GeoTIFF file path.
        reg:  Optional :class:`~hhg9.Registrar` — registers :class:`Wkt` +
              :class:`Wkt_4978` if supplied.
        name: Domain name for registration (default ``'g_wkt'``).

    Returns:
        ``(array, bbox, domain_or_wkt)`` where *array* is ``(H, W, C)`` uint8.
    """
    ds, bbox, domain = wkt_geotiff_meta(path, reg, name)
    bands = [ds.GetRasterBand(b + 1).ReadAsArray() for b in range(ds.RasterCount)]
    arr = np.stack(bands, axis=-1)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float64), 0.0, 1.0) * 255).astype(np.uint8)
    return arr, bbox, domain


def sample_gdal(ds, coords: np.ndarray):
    """
    Sample a single-band GDAL raster at projected (x, y) coordinates.

    Out-of-bounds coordinates are clamped to the raster edge for the read,
    but flagged ``False`` in the returned *valid* mask so callers can
    substitute NaN or nodata as appropriate.

    Args:
        ds:     Open :class:`gdal.Dataset` (single band).
        coords: ``(N, 2)`` array of (x, y) in the dataset's own CRS.

    Returns:
        vals:   ``(N,)`` raster values at each point.
        pix_id: ``(N,)`` flat pixel index (useful for de-duplication).
        valid:  ``(N,)`` bool mask — ``True`` where coords are within bounds.
    """
    inv_gt = gdal.InvGeoTransform(ds.GetGeoTransform())
    if not inv_gt:
        raise RuntimeError("sample_gdal: failed to invert GeoTransform")
    c0, c1, c2, c3, c4, c5 = inv_gt
    px = c0 + coords[:, 0] * c1 + coords[:, 1] * c2
    py = c3 + coords[:, 0] * c4 + coords[:, 1] * c5
    valid = (px >= 0) & (px < ds.RasterXSize) & (py >= 0) & (py < ds.RasterYSize)
    cols = np.clip(np.floor(px).astype(np.int64), 0, ds.RasterXSize - 1)
    rows = np.clip(np.floor(py).astype(np.int64), 0, ds.RasterYSize - 1)
    pix_id = rows * ds.RasterXSize + cols
    min_col, max_col = int(cols.min()), int(cols.max())
    min_row, max_row = int(rows.min()), int(rows.max())
    data = ds.GetRasterBand(1).ReadAsArray(
        min_col, min_row,
        max_col - min_col + 1,
        max_row - min_row + 1,
    )
    return data[rows - min_row, cols - min_col], pix_id, valid
