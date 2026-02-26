# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Not projecting to - pretending it's Bary..
26 December 2026 0.1.0a4
"""
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy.typing import NDArray
from matplotlib import font_manager
from osgeo import gdal, osr
import numpy as np

from hhg9 import Points, Registrar
from hhg9.base import Domain, Projection
from hhg9.formats import OctahedralH9
from hhg9.h9 import H9P, H9K
from hhg9.h9.polygon import hex_poly_layer
# from hhg9.h9.polygon import tri_grid
import json
from geographiclib.geodesic import Geodesic
from PIL import Image  # Pillow for clean image saving
geod = Geodesic.WGS84

def densify_poly_geodesic_by_step(poly_latlon: np.ndarray,
                                  max_step_m: float = 2_000.0,
                                  max_per_edge: int = 512,
                                  closed: bool = True) -> np.ndarray:
    """
    poly_latlon: (m,2) in (lat, lon), ordered along the boundary.
    Returns boundary samples (lat, lon).

    Notes:
      - Samples include each edge start, exclude each edge end (avoids duplicate vertices).
      - If closed and last vertex repeats first, the duplicate is ignored.
    """
    poly = np.asarray(poly_latlon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"poly_latlon must have shape (m,2); got {poly.shape}")
    if poly.shape[0] < 2:
        return poly.copy()

    if closed and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    m = poly.shape[0]
    last = m if closed else (m - 1)

    out = []
    for i in range(last):
        lat1, lon1 = poly[i]
        lat2, lon2 = poly[(i + 1) % m]

        inv = geod.Inverse(float(lat1), float(lon1), float(lat2), float(lon2))
        s12 = float(inv["s12"])
        line = geod.Line(float(lat1), float(lon1), float(inv["azi1"]))

        nseg = int(np.ceil(s12 / max_step_m)) if max_step_m > 0 else 1
        nseg = max(1, min(nseg, max_per_edge))

        s = np.linspace(0.0, s12, nseg, endpoint=False)
        for si in s:
            r = line.Position(float(si))
            out.append((r["lat2"], r["lon2"]))

    return np.array(out, dtype=np.float64)


def densify_quad_geodesic_by_step(corners_latlon: np.ndarray,
                                  max_step_m: float = 2_000.0,
                                  max_per_edge: int = 512) -> np.ndarray:
    """Convenience wrapper for densifying a 4-corner geodesic quad."""
    return densify_poly_geodesic_by_step(
        corners_latlon,
        max_step_m=max_step_m,
        max_per_edge=max_per_edge,
        closed=True,
    )


def mean_u8(inv_hex, counts, pts):
    """Mean reducer for uint8 samples, with per-pixel de-dupe."""
    h = counts.shape[0]
    out = np.empty(h, dtype=np.float64)
    if pts.samples is None:
        return out

    order = np.argsort(inv_hex)
    inv_s = inv_hex[order]
    v_s = pts.samples[order].astype(np.float64)
    p_s = pts.px_id[order].astype(np.int64, copy=False)

    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    ends = np.r_[starts[1:], inv_s.size]

    out[:] = np.nan
    for s, e in zip(starts, ends):
        k = inv_s[s]
        ps = p_s[s:e]
        vs = v_s[s:e]
        ok = (ps != -1) & np.isfinite(vs)
        if not np.any(ok):
            out[k] = np.nan
            continue
        _, uidx = np.unique(ps[ok], return_index=True)
        out[k] = float(np.mean(vs[ok][uidx]))
    return out


def mode_u8(inv_hex, counts, pts):
    h = counts.shape[0]
    out = np.empty(h, dtype=np.float64)
    if pts.samples is None:
        return out
    order = np.argsort(inv_hex)
    inv_s = inv_hex[order]
    v_s = pts.samples[order].astype(np.uint8)
    p_s = pts.px_id[order].astype(np.int64, copy=False)
    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    ends = np.r_[starts[1:], inv_s.size]

    out[:] = np.nan
    for s, e in zip(starts, ends):
        k = inv_s[s]
        ps = p_s[s:e]
        vs = v_s[s:e]
        ok = (ps != -1)
        if not np.any(ok):
            out[k] = np.nan
            continue
        _, uidx = np.unique(ps[ok], return_index=True)
        vsu = vs[ok][uidx]
        bc = np.bincount(vsu, minlength=256)
        out[k] = np.argmax(bc)
    return out


def make_priority_mode_u8(
    *,
    priority: list[int] | None = None,
    weights: NDArray | None = None,
    min_support: int = 1,
    support_map: dict[int, int] | None = None,
    fallback_to_mode: bool = True,
):
    """Factory for a categorical reducer that prefers certain classes.

    Useful for thin linear features (roads/rivers) that otherwise get lost under a
    strict majority vote.

    Parameters
    ----------
    priority:
        List of class codes, ordered from highest priority to lowest.
        Any class present in the pixel set for a hex will win over lower
        priority classes, subject to `min_support`.
    weights:
        Optional explicit weights array of shape (256,) where larger means more
        preferred. If provided, it overrides `priority`.
    min_support:
        Minimum number of (de-duped) pixels required for a preferred class to
        win. If the top-priority present class has fewer than `min_support`
        pixels, we either fall back to normal mode or return that class anyway
        depending on `fallback_to_mode`.
    support_map:
        Optional dict mapping class code -> minimum support (pixel count) required
        for that specific class to win. This lets you demand more evidence for
        classes that tend to "bleed" (e.g. rivers/roads).
    fallback_to_mode:
        If True, fall back to majority mode when preferred class is too sparse.

    Returns
    -------
    reducer callable with signature (inv_hex, counts, pts) -> float64(H,)
    """

    if weights is not None:
        w = np.asarray(weights)
        if w.shape != (256,):
            raise ValueError(f'weights must have shape (256,), got {w.shape}')
        w = w.astype(np.float64, copy=False)
    else:
        w = np.zeros((256,), dtype=np.float64)
        if priority:
            # Highest priority gets the largest weight.
            # Use a strictly decreasing sequence so any presence wins.
            n = len(priority)
            for rank, code in enumerate(priority):
                c = int(code)
                if 0 <= c < 256:
                    w[c] = float(n - rank)

    min_support = int(min_support)
    if min_support < 1:
        min_support = 1

    # Per-class support overrides
    s_map = {}
    if support_map:
        for k, v in support_map.items():
            c = int(k)
            if 0 <= c < 256:
                s_map[c] = max(1, int(v))

    def _reducer(inv_hex, counts, pts):
        h = counts.shape[0]
        out = np.empty(h, dtype=np.float64)
        if pts.samples is None:
            return out

        order = np.argsort(inv_hex)
        inv_s = inv_hex[order]
        v_s = pts.samples[order].astype(np.uint8)
        p_s = pts.px_id[order].astype(np.int64, copy=False)

        starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
        ends = np.r_[starts[1:], inv_s.size]

        out[:] = np.nan
        for s, e in zip(starts, ends):
            k = inv_s[s]
            ps = p_s[s:e]
            vs = v_s[s:e]
            ok = (ps != -1)
            if not np.any(ok):
                out[k] = np.nan
                continue

            # De-dupe per pixel
            _, uidx = np.unique(ps[ok], return_index=True)
            vsu = vs[ok][uidx]

            bc = np.bincount(vsu, minlength=256)
            if not np.any(bc):
                out[k] = np.nan
                continue

            present = bc > 0

            # If we have weights/priorities, pick the highest-weight class that is present.
            if np.any(w[present] > 0):
                # Mask out absent classes by setting weight to -inf.
                ww = np.where(present, w, -np.inf)
                choice = int(np.argmax(ww))
                need = s_map.get(choice, min_support)
                if bc[choice] >= need or (not fallback_to_mode):
                    out[k] = float(choice)
                    continue

            # Fallback: majority mode
            out[k] = float(np.argmax(bc))

        return out

    return _reducer


def sample_gdal(ds, coords):
    """Sample band 1 at map coordinates.

    ds: gdal.Open(...) dataset
    coords: (N,2) array in the same CRS/axis order as the dataset.

    Returns:
      vals: sampled values (dtype follows raster where sensible)
      pix_id: per-point pixel id (row*w+col), or -1 if out-of-bounds
    """
    gt = ds.GetGeoTransform()

    inv_res = gdal.InvGeoTransform(gt)
    if isinstance(inv_res, (tuple, list)) and len(inv_res) == 2 and isinstance(inv_res[0], (bool, int)):
        ok = bool(inv_res[0])
        inv_gt = inv_res[1]
    else:
        ok = inv_res is not None
        inv_gt = inv_res
    if not ok or inv_gt is None:
        raise RuntimeError("Failed to invert GeoTransform")

    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must have shape (N,2); got {coords.shape}")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())

    if srs.IsGeographic():
        # coords are (lat, lon) -> geotransform wants (x,y)=(lon,lat)
        y = coords[:, 0].astype(np.float64, copy=False)  # lat
        x = coords[:, 1].astype(np.float64, copy=False)  # lon
    else:
        # projected: coords already (x,y)
        x = coords[:, 0].astype(np.float64, copy=False)
        y = coords[:, 1].astype(np.float64, copy=False)

    # map -> pixel (vectorised affine)

    px = inv_gt[0] + inv_gt[1] * x + inv_gt[2] * y  # col (fractional)
    py = inv_gt[3] + inv_gt[4] * x + inv_gt[5] * y  # row (fractional)
    cols = np.floor(px).astype(np.int64)
    rows = np.floor(py).astype(np.int64)

    w = int(ds.RasterXSize)
    h = int(ds.RasterYSize)
    inb = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)

    # Heuristic axis-order fix: if swapping x/y yields materially better in-bounds fraction,
    # assume coordinates are (y,x) relative to dataset.
    if inb.size and (inb.mean() < 0.90):
        px2 = inv_gt[0] + inv_gt[1] * y + inv_gt[2] * x
        py2 = inv_gt[3] + inv_gt[4] * y + inv_gt[5] * x
        cols2 = np.floor(px2).astype(np.int64)
        rows2 = np.floor(py2).astype(np.int64)
        inb2 = (cols2 >= 0) & (cols2 < w) & (rows2 >= 0) & (rows2 < h)
        if inb2.mean() > inb.mean() + 0.05:
            cols, rows, inb = cols2, rows2, inb2

    pix_id = rows * w + cols
    pix_id = np.where(inb, pix_id, -1).astype(np.int64, copy=False)

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    # Choose an out-of-bounds fill value that won't poison reducers.
    # - For integer rasters (e.g., NLCD uint8), use nodata if present, else 0.
    # - For floating rasters, use NaN.
    if nodata is None:
        fill = np.nan
    else:
        fill = nodata

    # If integer data and nodata is None, use 0 (common for categorical masks)
    dt = band.DataType
    is_int = dt in (gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_UInt16, gdal.GDT_Int32, gdal.GDT_UInt32)
    if is_int and nodata is None:
        fill = 0

    vals = np.full((coords.shape[0],), fill, dtype=np.float64 if (not is_int and np.isnan(fill)) else np.float64)

    if not np.any(inb):
        return vals, pix_id

    cols_in = cols[inb]
    rows_in = rows[inb]
    min_col, max_col = int(cols_in.min()), int(cols_in.max())
    min_row, max_row = int(rows_in.min()), int(rows_in.max())
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    data_subset = band.ReadAsArray(min_col, min_row, width, height)

    vals_in = data_subset[rows_in - min_row, cols_in - min_col]

    idx = np.flatnonzero(inb)  # indices into the full N-length output arrays

    if nodata is None:
        # All in-bounds samples are considered valid.
        vals[idx] = vals_in
        return vals, pix_id

    # Mark nodata samples as invalid: keep fill in `vals`, but also set pix_id=-1
    # so downstream reducers can drop them via (px_id != -1).
    vok = (vals_in != nodata)
    if np.any(vok):
        vals[idx[vok]] = vals_in[vok]

    if np.any(~vok):
        pix_id[idx[~vok]] = -1

    return vals, pix_id


def create_label_lut(file_name: str):
    """Generate lut from json data"""
    fp = open(file_name)
    lut = json.load(fp)  # Load the legend
    return {int(k): v for k, v in lut.items()}


def create_nlcd_lut():
    """Generate the NLCD lut from json data"""
    # Initialize a 256-row table to handle uint8 data safely (or 96 for specific NLCD)
    # Shape (256, 3) for RGB
    lut = np.zeros((256, 3), dtype=np.uint8)
    fp = open('../assets/nlcd_legend.json')
    data = json.load(fp)  # Load the legend

    for item in data['nlcd_legend']:
        idx = item['id']
        hex_val = item['color'].lstrip('#')
        rgb = tuple(int(hex_val[i:i + 2], 16) for i in (0, 2, 4)) # Convert Hex to RGB tuple
        lut[idx] = rgb
    return lut


class Wkt(Domain):
    """
    Custom wkt domain
    The default name is g_wkt.
    # Keep the prefix stable for readability; keep the digest short but deterministic.
    """

    def valid(self, arr) -> NDArray:
        return super().valid(arr)

    def __init__(self, registrar, _wkt='', axes=3):
        from hashlib import blake2b
        self.sr = osr.SpatialReference()
        self.sr.ImportFromWkt(_wkt)  # here we define it via the Wkt in the file.
        wkt_hash = blake2b(_wkt.encode('utf-8'), digest_size=8).hexdigest()
        name = f'g_wkt:{wkt_hash}'
        # Ensure lon/lat (traditional GIS) axis order even under GDAL 3+/PROJ EPSG rules.
        self.sr.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
        axes = self.sr.GetAxesCount()
        super().__init__(registrar, name, axes)


class Wkt_4978(Projection):
    """Custom wkt projection to c_ell"""
    def __init__(self, registrar: Registrar, _wkt: Wkt):
        wkt_name = _wkt.name
        name = f'ell_{wkt_name[-3:]}'
        super().__init__(registrar, name, 'c_ell', wkt_name)
        self.esr = osr.SpatialReference()  # spatial reference for EPSG:4978
        self.esr.ImportFromEPSG(4978)      # WGS84 ECEF XYZ (m); equivalent to c_ell
        self.wsr = _wkt.sr                 # Some other wtk based projection.
        self.esr.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
        self.wsr.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
        self._fwd = osr.CoordinateTransformation(self.esr, self.wsr)  # fwd ell→wkt
        self._bak = osr.CoordinateTransformation(self.wsr, self.esr)  # bak wkt→ell

    def forward(self, pts: Points) -> Points:
        """ECEF;4978 → Wkt"""
        # x, y, z = pts.coords[..., 0], pts.coords[..., 1], pts.coords[..., 2]
        res = np.atleast_2d(self._fwd.TransformPoints(pts.coords))
        if res.shape[1] > self.fwd_cs.axes:
            res = res[:, :self.fwd_cs.axes]
        _result = Points(coords=res, domain=self.fwd_cs, samples=pts.samples)
        return _result

    def backward(self, pts: Points) -> Points:
        """Wkt → ECEF;4978"""
        res = np.atleast_2d(self._bak.TransformPoints(pts.coords))
        if res.shape[1] > self.rev_cs.axes:
            res = res[:, :self.rev_cs.axes]
        _result = Points(coords=res, domain=self.rev_cs, samples=pts.samples)
        return _result


def tri_grid_clipped(level, mode, bbox, h9p=H9P):
    """Return a set of barycentric xy cell centroids for level hex_layer, mode M, clipped by a TLBR bounding box"""
    tp, lt, bt, rt = bbox
    from hhg9.h9 import H9C, H9R

    # Initialize arrays with the starting triangle
    origins = np.array([[0.0, 0.0]])  # Shape (hex_layer, 2)
    modes = np.array([mode], dtype=int)  # Shape (hex_layer,)
    scale = 1.0

    # Pre-fetch lookup tables as NumPy arrays for faster indexing
    # We assume H9C.off_xy and H9C.mode are indexable by the child indices k
    off_xy_lut = np.array(H9C.off_xy)
    mode_lut = np.array(H9C.mode)
    mo_reg_lut = np.array([H9R.downs, H9R.ups])

    for _ in range(level):
        # 1. Get the child indices for every current triangle
        # ks shape: (hex_layer, 9)
        ks = mo_reg_lut[modes]

        # 2. Calculate child origins using broadcasting
        # (hex_layer, 1, 2) + (hex_layer, 9, 2) * scale -> (hex_layer, 9, 2)
        child_offsets = off_xy_lut[ks] * scale
        origins = (origins[:, np.newaxis, :] + child_offsets).reshape(-1, 2)

        # 3. Update modes and scale
        modes = mode_lut[ks].flatten()
        scale /= 3.0

        # 4. Vectorized Clipping (Culling)
        # Compute all vertices: (N_total, 3_vertices, 2_coords)
        # h9p.sv[modes] shape: (N_total, 3, 2)
        vertices = origins[:, np.newaxis, :] + (h9p.sv[modes] * scale)

        # Calculate BBox for every triangle in the batch
        a_lt = np.min(vertices[:, :, 0], axis=1)
        a_rt = np.max(vertices[:, :, 0], axis=1)
        a_bt = np.min(vertices[:, :, 1], axis=1)
        a_tp = np.max(vertices[:, :, 1], axis=1)

        # Keep only triangles that intersect the bbox
        mask = ~((a_rt < lt) | (rt < a_lt) | (a_tp < bt) | (tp < a_bt))

        origins = origins[mask]
        modes = modes[mask]

        # Optimization: If no triangles left, exit early
        if len(origins) == 0:
            break

    return origins


def plot_hex(overlays, xy, name='hex', bbox=None, nodata=0, shade_strength=0.68):
    """Plot one or more hex layers.
    """
    wd, ht = xy
    tp, lt, bt, rt = bbox  # This is in domain size! Not inch size!
    extent = (lt, rt, bt, tp)
    dpi = 100
    adj = (72 / dpi)
    fig = plt.figure(figsize=(wd/100, ht/100), dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # <- instead of add_subplot
    ax.set_facecolor((0, 0, 0, 0))  # transparent
    fig.patch.set_alpha(0.0)
    ax.set_axis_off()

    if not overlays:
        raise ValueError("plot_hex: overlays must be a non-empty list")

    # Draw each overlay as its own PolyCollection (base first, shade later).
    for overlay, item in enumerate(overlays):
        if (not isinstance(item, (list, tuple))) or len(item) != 3:
            raise ValueError("plot_hex: each overlay must be [hx_pts, values, lut]")
        hx_pts, values, lut = item

        polys = np.asarray(hx_pts.coords, dtype=np.float64).reshape(-1, 6, 2)
        h = polys.shape[0]
        if h == 0:
            continue

        face_colors = 'none'
        # Default to no edges; enable explicitly per-overlay to avoid grey "veil" from dense outlines.
        edge_colors = 'none'
        line_widths = 0.0

        if values is not None and lut is not None:
            if isinstance(lut, dict):
                pixel_height = 18  # pixels
                # This is an id to label thing
                font_height = pixel_height * adj  # adjust to dpi.
                line_widths = 1.25 * adj
                edge_colors = '#00000080'

                # Define a high zorder so text is ALWAYS on top of every polygon hex_layer
                text_zorder = 100

                for hx, v in zip(polys, values):
                    if v == 0:
                        continue

                    kind = lut[v]
                    # Handle both simple string lookups or complex dict lookups
                    label = kind['desc'] if isinstance(kind, dict) else str(kind)

                    # 1. Get the Bounds
                    min_x = np.min(hx[:, 0])
                    min_y = np.min(hx[:, 1])
                    max_x = np.max(hx[:, 0])
                    max_y = np.max(hx[:, 1])

                    # 2. Calculate Margin (10% of width)
                    margin_x = (max_x - min_x) * 0.23
                    margin_y = (max_y - min_y) * 0.01

                    # 3. Plot at the COORDINATE (Min + Margin)
                    ax.text(
                        min_x + margin_x,
                        min_y + margin_y,
                        label,
                        fontsize=font_height,
                        ha='left',
                        va='bottom',
                        zorder=text_zorder,  # Critical: Ensure text renders above polygons
                        color='white',
                        clip_on=True
                    )
            else:
                # Land Usage
                codes = values.astype(np.int32, copy=False)
                codes[codes == 0] = 11
                if nodata is not None:
                    codes[codes == int(nodata)] = 11
                codes = np.clip(codes, 0, lut.shape[0] - 1)
                rgb = lut[codes].astype(np.float64) / 255.0  # (H,3)
                alpha = np.ones((codes.shape[0], 1), dtype=np.float64)
                # alpha[codes == int(nodata)] = 0.0
                face_colors = np.concatenate([rgb, alpha], axis=1)  # (H,4)
                ds = face_colors.reshape([ht, wd, -1])
                plt.imshow(ds, extent=extent, origin='upper', interpolation='nearest')
                continue

        elif values is not None and lut is None:
            h = values.shape[0]
            sh = np.asarray(values / 255.0, dtype=np.float64)
            alpha = np.zeros((h,), dtype=np.float64)
            ok = np.isfinite(sh)
            ss = float(np.clip(shade_strength, 0.0, 1.0))
            alpha[ok] = ss * (1.0 - sh[ok])

            face_colors = np.zeros((h, 4), dtype=np.float64)
            face_colors[ok, 3] = alpha[ok]
            ds = face_colors.reshape([ht, wd, -1])
            plt.imshow(ds, extent=extent, origin='upper', interpolation='nearest')
            continue

        else:
            # Pure overlay with no values: draw visible outlines.
            line_widths = 2.0 * adj
            # White edges so they show over dark hillshade.
            edge_colors = np.tile(np.array([1.0, 1.0, 1.0, 0.35], dtype=np.float64), (h, 1))
            face_colors = 'none'
            if lut is not None and isinstance(lut, list): # not a lut.
                text_z_order = 1000
                pixel_height = 120  # labels
                font_height = pixel_height * adj  # adjust to dpi.
                for hx, txt in zip(polys, lut):
                    if txt == '':
                        continue
                    font_to_use = 'SimpleDragonRunes'
                    # label = ''.join(['ᛃᛐᛚᛩᚦᚳᚴᛊᚲ'[ord(ch) - ord('0')] for ch in txt])
                    label = txt
                    # 1. Get the Bounds
                    min_x = np.min(hx[:, 0])
                    min_y = np.min(hx[:, 1])
                    max_x = np.max(hx[:, 0])
                    max_y = np.max(hx[:, 1])

                    # 2. Calculate Margin (10% of width)
                    margin_x = (max_x - min_x) * 0.25
                    margin_y = (max_y - min_y) * 0.01

                    # 3. Plot at the COORDINATE (Min + Margin)
                    ax.text(
                        min_x + margin_x,
                        min_y + margin_y,
                        label,
                        fontfamily=font_to_use,
                        fontsize=font_height,
                        ha='left',
                        va='bottom',
                        zorder=text_z_order,  # Ensure text renders above polygons
                        color="#F0F000", alpha=0.8,
                        clip_on=True
                    )

        # Matplotlib can show faint seams between adjacent polygons due to antialiasing
        # and sub-pixel coverage (especially with alpha). Disable AA for filled layers.
        aa = not (overlay in (0, 1))
        rast = (overlay in (0, 1))
        ax.add_collection(
            PolyCollection(
                polys,
                linewidths=line_widths,
                facecolors=face_colors,
                edgecolors=edge_colors,
                zorder=20 + overlay,
                antialiased=aa,
                rasterized=rast,
                snap=True
            )
        )

    # View/Image limits
    tp, lt, bt, rt = bbox
    xlim0, xlim1 = lt, rt
    ylim0, ylim1 = bt, tp
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    canvas.draw()
    out_path = f"output/ex0255_ch_{name}.png"
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4) uint8
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    print(f"fig saved at {out_path}")
    plt.close(fig)


def pixel_to_map(gt, px, py):
    x = gt[0] + px*gt[1] + py*gt[2]
    y = gt[3] + px*gt[4] + py*gt[5]
    return x, y


def geotiff_to_points(ds: gdal.Dataset, band=1, stride=1) -> [Points, Wkt, tuple]:
    b = ds.GetRasterBand(band)
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize

    cols = (np.arange(0, w, stride, dtype=np.float64) + 0.5)
    rows = (np.arange(0, h, stride, dtype=np.float64) + 0.5)
    cc, rr = np.meshgrid(cols, rows)  # (H,W)

    x = gt[0] + cc * gt[1] + rr * gt[2]
    y = gt[3] + cc * gt[4] + rr * gt[5]

    coords = np.stack([x, y], axis=-1).reshape(-1, 2)   # (N,2) in dataset CRS
    vals = b.ReadAsArray(0, 0, w, h)[::stride, ::stride].reshape(-1)

    # Domain choice:
    wkt = Wkt(rg, ds.GetProjection())  # construct the domain.
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    print("IsGeographic:", bool(srs.IsGeographic()))
    print("gt:", gt)
    print("px_size:", gt[1], gt[5])
    pts = Points(coords=coords, domain=wkt, samples=vals)
    return pts, wkt, (w, h)


if __name__ == '__main__':
    font_path = "src/SimpleDragonRunes-Regular.ttf"
    font_manager.fontManager.addfont(font_path)
    alien_name = font_manager.FontProperties(fname=font_path).get_name()
    # plt.rcParams["font.family"] = alien_name

    print('Initialising')
    gdal.UseExceptions()
    hy = H9K.lattice.V
    rg = Registrar()
    n_layout = 'diamonds'  # Keep north-pole north.
    b_oct = rg.domain('b_oct')
    b_oct.register_format(OctahedralH9(rg))
    b_oct.set_warp('src/l4_polished.npz')
    g_gcd = rg.domain('g_gcd')
    p_pix = rg.domain('p_pix')
    nlcd_lut = create_nlcd_lut()
    print('Loading and registering geotifs')

    l_pts = None
    hx = 0
    ix, iy = 0, 0
    if True:  # Load Land Usage...
        src_file = 'src/use_region10_pc_sea11.tif'  # land usage.
        lu = gdal.Open(src_file)
        lup, l_wkt, (ix, iy) = geotiff_to_points(lu)
        ldx = ix/iy
        hx = ldx*hy
        dx = (np.linspace(-hx, hx, ix, dtype=np.float64))
        dy = (np.linspace(-hy, hy, iy, dtype=np.float64))
        cc, rr = np.meshgrid(dy, dx)  # (H,W)
        coords = np.stack([cc, rr], axis=-1).reshape(-1, 2)   # (N,2) in dataset CRS
        l_pts = Points(coords=coords, domain=b_oct, components=(1, 1, 1), samples=lup.samples)
    s_pts = None
    if True:  # Load Land Usage...
        src_file = 'src/r10_hillshade_masked.tif'  # land usage.
        lu = gdal.Open(src_file)
        lup, l_wkt, (lx, ly) = geotiff_to_points(lu)
        ldx = lx/ly
        hx = ldx*hy
        dx = (np.linspace(-hx, hx, lx, dtype=np.float64))
        dy = (np.linspace(-hy, hy, ly, dtype=np.float64))
        cc, rr = np.meshgrid(dy, dx)  # (H,W)
        coords = np.stack([cc, rr], axis=-1).reshape(-1, 2)   # (N,2) in dataset CRS
        s_pts = Points(coords=coords, domain=b_oct, components=(1, 1, 1), samples=lup.samples)
        # Prevent hillshade from darkening open water (NLCD code 11)
        if l_pts is not None and s_pts is not None:
            sh = np.asarray(s_pts.samples, dtype=np.float64)
            sh = np.where(l_pts.samples == 11, np.nan, sh)
            s_pts = Points(coords=s_pts.coords, domain=s_pts.domain, components=s_pts.components, samples=sh)

    # BBox is (top, left, bottom, right)
    bbox = (hy, -hx, -hy, hx)  # TLBR
    for layer in [5]:  # range(8):
        print(f'Generating layer {layer}...')

        ovr_delta = -3
        print(f'{layer}: Over-grid at {layer + ovr_delta}')
        ovr_pts = tri_grid_clipped(level=(layer + ovr_delta) + 2, mode=0, bbox=bbox)
        ovr_bt = Points(ovr_pts, b_oct, components=(1, 1, 1))
        ovr_3, _ = hex_poly_layer(ovr_bt, layers=layer + ovr_delta)
        ovr_hx = ovr_3.coords.reshape(-1, 6, 2)
        ovr_ct = np.mean(ovr_hx, axis=1)
        ovr_cp = Points(ovr_ct, b_oct, components=(1, 1, 1))
        lab_3a = f'{ovr_cp:h9.k{layer + ovr_delta + 0}}'.split('\n')

        # Over-grid L-1 (coarser)
        ovr_delta = -2
        print(f'{layer}: Over-grid at {layer + ovr_delta}')
        ovr_pts = tri_grid_clipped(level=(layer + ovr_delta) + 2, mode=0, bbox=bbox)
        ovr_bt = Points(ovr_pts, b_oct, components=(1, 1, 1))
        ovr_2, _ = hex_poly_layer(ovr_bt, layers=layer + ovr_delta)
        ovr_hx = ovr_2.coords.reshape(-1, 6, 2)
        ovr_ct = np.mean(ovr_hx, axis=1)
        ovr_cp = Points(ovr_ct, b_oct, components=(1, 1, 1))
        lab_2a = f'{ovr_cp:h9.k{layer + ovr_delta + 0}}'.split('\n')

        ovr_delta = -1
        print(f'{layer}: Over-grid at {layer + ovr_delta}')
        ovr_pts = tri_grid_clipped(level=(layer + ovr_delta) + 2, mode=0, bbox=bbox)
        ovr_bt = Points(ovr_pts, b_oct, components=(1, 1, 1))
        ovr_1, _ = hex_poly_layer(ovr_bt, layers=layer + ovr_delta)
        ovr_hx = ovr_1.coords.reshape(-1, 6, 2)
        ovr_ct = np.mean(ovr_hx, axis=1)
        ovr_cp = Points(ovr_ct, b_oct, components=(1, 1, 1))
        lab_1a = f'{ovr_cp:h9.k{layer + ovr_delta + 0}}'.split('\n')

        print(f'{layer}: Projecting layers')
        overlays = [[l_pts, l_pts.samples, nlcd_lut], [s_pts, s_pts.samples, None], [ovr_1, None, None], [ovr_2, None, lab_2a], [ovr_3, None, None]]

        print(f'{layer}: Plotting b_oct')
        plot_hex(overlays, (ix, iy), f'L{layer:02d}', bbox=bbox)
