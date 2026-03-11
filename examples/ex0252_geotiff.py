# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Showing 2 layers of hexagons:  land-usage, hill-shading
Last Tested
26 December 2026 0.1.0a4 (passed)
"""
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from osgeo import gdal, osr
import numpy as np
import math
from hhg9 import Points, Registrar
from hhg9.base import Domain, Projection
from hhg9.algorithms.distance import wgs84_offset, wgs84_area
from hhg9.h9 import H9P
from hhg9.h9.polygon import hex_poly_layer
import json
from geographiclib.geodesic import Geodesic

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
        _, uidx = np.unique(p_s[s:e], return_index=True)   # de-dupe same raster pixel
        out[k] = float(np.mean(v_s[s:e][uidx]))
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
        _, uidx = np.unique(p_s[s:e], return_index=True)
        bc = np.bincount(v_s[s:e][uidx], minlength=256)
        # bc = np.bincount(v_s[s:e], minlength=256)
        out[k] = np.argmax(bc)
    return out


def sample_gdal(ds, coords):
    """
    ds: gdal.Open(src_file) object
    coords: NumPy array of shape (hex_layer, 2) in the same CRS as the TIFF
    """
    gt = ds.GetGeoTransform()   # Affine Transform [UL_x, res_x, rot_x, UL_y, rot_y, res_y]
    inv_gt = gdal.InvGeoTransform(gt)  # API change! Get the Inverse Transform to go from Map -> Pixel
    if not inv_gt:
        raise RuntimeError("Failed to invert GeoTransform")
    c0, c1, c2, c3, c4, c5 = inv_gt  # Extract the coefficients for readability

    px = c0 + coords[:, 0] * c1 + coords[:, 1] * c2  # Pixel X = c0 + (Map X * c1) + (Map Y * c2)
    py = c3 + coords[:, 0] * c4 + coords[:, 1] * c5  # Pixel Y = c3 + (Map X * c4) + (Map Y * c5)

    cols = np.floor(px).astype(np.int64)
    rows = np.floor(py).astype(np.int64)
    # cols = np.clip(cols, 0, ds.RasterXSize - 1)
    # rows = np.clip(rows, 0, ds.RasterYSize - 1)

    pix_id = rows * ds.RasterXSize + cols

    min_col, max_col = cols.min(), cols.max()  # bounds of area of interest. cols
    min_row, max_row = rows.min(), rows.max()  # bounds of area of interest. rows
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    band = ds.GetRasterBand(1)
    data_subset = band.ReadAsArray(min_col, min_row, width, height)  # no need to load CONUS.
    vals = data_subset[rows - min_row, cols - min_col]
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
    This is fine while one is using only one Wkt defined domain.
    """

    def valid(self, arr) -> NDArray:
        return super().valid(arr)

    def __init__(self, registrar, name='g_wkt', _wkt='', axes=3):
        self.sr = osr.SpatialReference()
        self.sr.ImportFromWkt(_wkt)  # here we define it via the Wkt in the file.
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


def plot_pts(level, pts, bbox):
    """Draw points in bbox"""
    tp, lt, bt, rt = bbox
    fig = plt.figure(figsize=(17, 10), dpi=300, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)  # (*nrows*, *ncols*, *index*)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lt, rt)  #
    ax.set_ylim(bt, tp)  #
    ax.set_axis_off()
    ax.scatter(pts[:, 0], pts[:, 1], ec='none', marker='.', s=10)
    fig.savefig(f"output/ex0250_grid_{level}.jpg", dpi=200)


def plot_hex(overlays, name='hex', bbox=None, draw_bbox=True, nodata=0, shade_strength=0.65):
    """Plot one or more hex layers.
    """
    dpi = 500
    adj = (72 / dpi)
    fig = plt.figure(figsize=(14, 17), dpi=dpi, frameon=False)
    ax = fig.add_subplot(111)

    if not overlays:
        raise ValueError("plot_hex: overlays must be a non-empty list")

    # Draw each overlay as its own PolyCollection (base first, shade later).
    for i, item in enumerate(overlays):
        if (not isinstance(item, (list, tuple))) or len(item) != 3:
            raise ValueError("plot_hex: each overlay must be [hx_pts, values, lut]")
        hx_pts, values, lut = item

        polys = np.asarray(hx_pts.coords, dtype=np.float64).reshape(-1, 6, 2)
        h = polys.shape[0]
        if h == 0:
            continue

        face_colors = 'none'
        edge_colors = '#00000080'
        line_widths = 0.0

        if values is not None and lut is not None:
            if isinstance(lut, dict):
                pixel_height = 18  # pixels
                # This is an id to label thing
                font_height = pixel_height * adj  # adjust to dpi.
                line_widths = 1 * adj
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
                # Categorical base hex_layer
                line_widths = 1/3 * adj
                v = np.asarray(values)
                if v.shape != (h,):
                    raise ValueError(f"plot_hex: values must have shape ({h},), got {v.shape}")

                codes = v.astype(np.int32, copy=False)
                codes = np.clip(codes, 0, lut.shape[0] - 1)
                rgb = lut[codes].astype(np.float64) / 255.0  # (H,3)

                alpha = np.ones((h, 1), dtype=np.float64)
                if nodata is not None:
                    alpha[codes == int(nodata)] = 0.0
                face_colors = np.concatenate([rgb, alpha], axis=1)  # (H,4)

        elif values is not None and lut is None:
            line_widths = 1/9 * adj
            # Shade hex_layer: darken only (preserves hue better than greyscale compositing)
            sh = np.asarray(values, dtype=np.float64)
            if sh.shape != (h,):
                raise ValueError(f"plot_hex: shade values must have shape ({h},), got {sh.shape}")

            alpha = np.zeros((h,), dtype=np.float64)
            ok = np.isfinite(sh)
            sh_clip = np.zeros((h,), dtype=np.float64)
            sh_clip[ok] = np.clip(sh[ok], 0.0, 1.0)

            ss = float(np.clip(shade_strength, 0.0, 1.0))
            alpha[ok] = ss * (1.0 - sh_clip[ok])

            # RGBA: black with varying alpha
            face_colors = np.zeros((h, 4), dtype=np.float64)
            face_colors[:, 3] = alpha

        else:
            line_widths = adj
            face_colors = np.zeros((h, 4), dtype=np.float64)

        ax.add_collection(
            PolyCollection(
                polys,
                linewidths=line_widths,
                facecolors=face_colors,
                edgecolors=edge_colors,
                zorder=10 + i,
                antialiased=True
            )
        )

    # View limits
    if bbox is not None:
        tp, lt, bt, rt = bbox
        pad_x = 0.02 * (rt - lt) if rt > lt else 1.0
        pad_y = 0.02 * (tp - bt) if tp > bt else 1.0
        ax.set_xlim(lt - pad_x, rt + pad_x)
        ax.set_ylim(bt - pad_y, tp + pad_y)

        if draw_bbox:
            ax.add_patch(Rectangle((lt, bt), rt - lt, tp - bt, fill=False, linewidth=1.0))
    else:
        # Derive bounds from the *first* overlay
        hx0, _, _ = overlays[0]
        p0 = np.asarray(hx0.coords, dtype=np.float64).reshape(-1, 6, 2)
        xmin = float(np.min(p0[:, :, 0]))
        xmax = float(np.max(p0[:, :, 0]))
        ymin = float(np.min(p0[:, :, 1]))
        ymax = float(np.max(p0[:, :, 1]))
        pad_x = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
        pad_y = 0.02 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0252_sn_{name}.png", dpi=dpi)
    print(f"fig saved at output/ex0252_sn_{name}.png")


if __name__ == '__main__':
    print('Initialising')
    gdal.UseExceptions()
    rg = Registrar()
    b_oct = rg.domain('b_oct')
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')  # EPSG:4978
    b_oct = rg.domain('b_oct')
    c_oct = rg.domain('c_oct')
    n_oct = rg.domain(f'n_oct:diamonds')  # Keep north pole north.

    nlcd_lut = create_nlcd_lut()
    # Center      (120° 6'54.88"W, 38°56'53.96"N) 38.9483222222, -120.1152444444
    # Upper Left  (120°13'43.75"W, 38°59'46.23"N)  38.996175, -120.2288194444
    # Lower Right (120° 0' 6.58"W, 38°54' 1.31"N)  38.9003638889, -120.0018277778
    # src/CA_SierraNevada_hs.tif
    print('Loading and registering geotifs')

    # We are going to load up a tif.  It's a proxy, so not hogging memory here.
    src_file = 'src/Annual_NLCD_LndCov_2024_CU_C1V1_crop.tif'  # https://www.mrlc.gov/ land usage.
    ds = gdal.Open(src_file)
    l_wkt = Wkt(rg, 'l_wkt', ds.GetProjection())  # construct the domain.
    lu_wkt = Wkt_4978(rg, l_wkt)  # construct the projection
    # Emerald Bay, Lake Tahoe
    # Center      (120° 6'54.88"W, 38°56'53.96"N)  38.9483222222, -120.1152444444
    # Upper Left  (120°13'43.75"W, 38°59'46.23"N)  38.996175, -120.2288194444
    # Lower Right (120° 0' 6.58"W, 38°54' 1.31"N)  38.9003638889, -120.0018277778
    hs_ds = gdal.Open("src/CA_SierraNevada_hs.tif")  # Grab the hillshade for this area.
    s_wkt = Wkt(rg, 's_wkt', hs_ds.GetProjection())  # construct the domain.
    hs_wkt = Wkt_4978(rg, s_wkt)  # construct the projection
    # Let's ensure that hhg9 and gdal/osr are speaking the same language!
    # Here we are using A given point at the centre of the map.
    print('Determining area of presentation')
    focus = "CaSN"
    size, size_str = 1_000, '1k'        # width/height of map, in metres.
    # 38.948322, -120.115244 is the centre of the hillshade - much smaller area.
    # 20_000 = max extend e/w, 10_000 n/s
    rcc = Points(np.atleast_2d([38.9517340730661, -120.1119544660363]), g_gcd)  # Trailhead.
    # rcc = Points(np.atleast_2d([38.9483222222, -120.1152444444]), g_gcd)
    centres = rcc.coords  # shape (n, 2) in (lat, lon)
    diag = (size / 2.0) * math.sqrt(2.0)  # centre → corner
    az = np.array([45.0, 135.0, 225.0, 315.0], dtype=np.float64)
    n = centres.shape[0]
    centres_rep = np.repeat(centres, az.size, axis=0)  # (n*4, 2)
    az_rep = np.tile(az, n)  # (n*4,)
    dist_rep = np.full(n * az.size, diag, dtype=np.float64)
    corners = wgs84_offset(centres_rep, az_rep, dist_rep)  # (n*4, 2)
    quad = Points(np.atleast_2d(corners), g_gcd)
    qx = densify_quad_geodesic_by_step(quad.coords, 50)
    # print(corners)
    q_dens = Points(qx, g_gcd)
    bhx = rg.project(q_dens, [g_gcd, b_oct])
    # calculate our b_oct limits accordingly. Nice and square here.. :-/
    lt = float(bhx.coords[:, 0].min())
    rt = float(bhx.coords[:, 0].max())
    bt = float(bhx.coords[:, 1].min())
    tp = float(bhx.coords[:, 1].max())
    cmp, mo = bhx.cm()
    if np.unique(cmp).size > 1:
        raise NotImplementedError('Need to improve the mode, and components part here.')
    bbox = (tp, lt, bt, rt)  # TLBR
    for layer in [12]:  # range(8, 11):
        print(f'Generating layer {layer}...')

        # grab land-usage via l_wkt
        print(f'{layer}: land-usage')
        pts = tri_grid_clipped(level=layer+2, mode=int(mo[0]), bbox=bbox)
        bt = Points(pts, b_oct, components=bhx.components[0])
        smp = rg.project(bt, [b_oct, c_oct, c_ell, l_wkt])
        bt.samples, px_id = sample_gdal(ds, smp.coords)
        bt.px_id = px_id   # de-dupes
        lu_hx_pts, lu_vals = hex_poly_layer(bt, layers=layer, reducer=mode_u8)

        # grab hillshade via s_wkt at L+1 (denser)
        print(f'{layer}: hillshade')
        shade_delta = 1
        shade_pts = tri_grid_clipped(level=(layer+shade_delta)+2, mode=int(mo[0]), bbox=bbox)
        shade_bt = Points(shade_pts, b_oct, components=bhx.components[0])
        shade_smp = rg.project(shade_bt, [b_oct, c_oct, c_ell, s_wkt])
        shade_bt.samples, shade_px_id = sample_gdal(hs_ds, shade_smp.coords)  # see below
        shade_bt.px_id = shade_px_id
        sh_hx_pts, shade_v = hex_poly_layer(shade_bt, layers=layer + shade_delta, reducer=mean_u8)
        sh_vals = np.asarray(shade_v, dtype=np.float64) / 255.0
        sh_vals = np.clip(sh_vals, 0.0, 1.0)

        # Over-grid L-1 (coarser)
        ovr_delta = -1
        print(f'{layer}: Over-grid at {layer + ovr_delta}')
        ovr_pts = tri_grid_clipped(level=(layer + ovr_delta) + 2, mode=int(mo[0]), bbox=bbox)
        ovr_bt = Points(ovr_pts, b_oct, components=bhx.components[0])
        ovr_hx_pts, ovr_v = hex_poly_layer(ovr_bt, layers=layer + ovr_delta, reducer=mode_u8)

        print(f'{layer}: Projecting layers')
        use_n = rg.project(lu_hx_pts, [b_oct, n_oct])
        shd_n = rg.project(sh_hx_pts, [b_oct, n_oct])
        ovr_n = rg.project(ovr_hx_pts, [b_oct, n_oct])

        nbx = rg.project(bhx, [b_oct, n_oct])
        nlt = float(nbx.coords[:, 0].min())
        nrt = float(nbx.coords[:, 0].max())
        nbt = float(nbx.coords[:, 1].min())
        ntp = float(nbx.coords[:, 1].max())
        bbox_n = ntp, nlt, nbt, nrt
        overlays = [[use_n, lu_vals, nlcd_lut], [shd_n, sh_vals, None], [ovr_n, None, None]]
        print(f'{layer}: Plotting.')
        plot_hex(overlays, f'{size_str}L{layer:02d}_{focus}', bbox=bbox_n)

