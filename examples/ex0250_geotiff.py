# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Given a point in the continental US (or anywhere for which you have geotiff
land usage that uses NLCD categories) and generate a hexgrid plot of that area.

This example requires osgeo, gdal in order to read geotiffs
While hhg9 does not depend upon such tools, with GIS it's not
a bad idea to have these libraries, alongside the GIS stack it does use
(geographiclib, proj)

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
"""
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import numpy as np
import math
from hhg9 import Points, Registrar
from hhg9.h9 import H9P
from hhg9.h9.polygon import hex_poly_layer
import json
from geographiclib.geodesic import Geodesic
from hhg9.algorithms.distance import wgs84_offset, densify_quad_geodesic_by_step
from osgeo import gdal
from hhg9.geo.gdal import Wkt, Wkt_4978

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


def mode_u8(inv_hex, counts, pts):
    h = counts.shape[0]
    out = np.empty(h, dtype=np.float64)
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


def sample_nlcd_gdal(ds, coords):
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
    cols = np.clip(cols, 0, ds.RasterXSize - 1)
    rows = np.clip(rows, 0, ds.RasterYSize - 1)

    pix_id = rows * ds.RasterXSize + cols

    min_col, max_col = cols.min(), cols.max()  # bounds of area of interest. cols
    min_row, max_row = rows.min(), rows.max()  # bounds of area of interest. rows
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    band = ds.GetRasterBand(1)
    data_subset = band.ReadAsArray(min_col, min_row, width, height)  # no need to load CONUS.
    vals = data_subset[rows - min_row, cols - min_col]
    return vals, pix_id


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


def tri_grid_clipped(level, mode, bbox, h9p=H9P):
    """Return a set of barycentric xy cell centroids for layer hex_layer, mode M, clipped by a TLBR bounding box"""
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
    # mo_reg_lut: mapping mode -> 9 child indices [k1, k2, ... k9]
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

def plot_hex(pts, name='hex', bbox=None, draw_bbox=True, values=None, lut=None, nodata=0):
    """Plot hex polygons.
    Args:
        pts: Points containing hex vertices, shaped (H*6,2) in `pts.coords`.
        name: output filename suffix.
        bbox: optional TLBR tuple (tp, lt, bt, rt) for view limits.
        draw_bbox: draw bbox rectangle if bbox provided.
        values: optional per-hex categorical values (H,) (e.g. NLCD class codes).
        lut: optional uint8 LUT of shape (256,3) mapping class -> RGB.
        nodata: class code treated as nodata (alpha=0). Set to None to disable.
    """
# def plot_hex(pts, name='hex', bbox=None, draw_bbox=True):
    fig = plt.figure(figsize=(17, 10), dpi=500, frameon=False)
    ax = fig.add_subplot(111)

    polys = np.asarray(pts.coords, dtype=np.float64).reshape(-1, 6, 2)
    h = polys.shape[0]
    if h == 0:
        print(f"plot_hex: no polygons for {name}")
        return

    face_colors = 'none'
    edge_colors = 'black'
    if (values is not None) and (lut is not None):
        v = np.asarray(values)
        if v.shape != (h,):
            raise ValueError(f"plot_hex: values must have shape ({h},), got {v.shape}")

        # Map class codes -> RGBA in [0,1]
        codes = v.astype(np.int32, copy=False)
        codes = np.clip(codes, 0, lut.shape[0] - 1)
        rgb = lut[codes].astype(np.float64) / 255.0  # (H,3)
        alpha = np.ones((h, 1), dtype=np.float64)
        if nodata is not None:
            alpha[codes == int(nodata)] = 0.0
        face_colors = np.concatenate([rgb, alpha], axis=1)  # (H,4)
        edge_colors = 'none'  # cleaner for categorical fills; change to 'black' if you want outlines

    ax.add_collection(PolyCollection(polys, linewidths=0.2, facecolors=face_colors, edgecolors=edge_colors))

    if bbox is not None:
        tp, lt, bt, rt = bbox
        pad_x = 0.02 * (rt - lt) if rt > lt else 1.0
        pad_y = 0.02 * (tp - bt) if tp > bt else 1.0
        ax.set_xlim(lt - pad_x, rt + pad_x)
        ax.set_ylim(bt - pad_y, tp + pad_y)

        if draw_bbox:
            ax.add_patch(Rectangle((lt, bt), rt - lt, tp - bt, fill=False, linewidth=1.0))
    else:
        xmin = float(np.min(polys[:, :, 0])); xmax = float(np.max(polys[:, :, 0]))
        ymin = float(np.min(polys[:, :, 1])); ymax = float(np.max(polys[:, :, 1]))
        pad_x = 0.02 * (xmax - xmin) if xmax > xmin else 1.0
        pad_y = 0.02 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0250_hx_{name}.png", dpi=500)
    print(f"fig saved at output/ex0250_hx_{name}.png")


if __name__ == '__main__':
    gdal.UseExceptions()
    rg = Registrar()

    # Example Usage with the vectorized grid:
    # nlcd_values: a numpy array of NLCD digits retrieved from your raster
    # colors: the resulting array of RGB values for every triangle/point
    nlcd_lut = create_nlcd_lut()
    # grid_colors = nlcd_lut[nlcd_values]

    # We are going to load up a tif.  It's a proxy, so not hogging memory here.
    # 2016 Alaska
    src_file = '../experimental/personal/src/NLCD_2016_Land_Cover_AK_20200724.img'  # Alaska only!
    # 2023 Conus for the USA continent
    # src_file = 'src/USA_NLCD_LndCov_2023_CU_C1V1.tif'  # https://www.mrlc.gov/ land usage.
    ds = gdal.Open(src_file)
    g_wkt = Wkt(rg, 'g_wkt', ds.GetProjection())
    p_wkt = Wkt_4978(rg, g_wkt)

    # Let's ensure that hhg9 and gdal/osr are speaking the same language!
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')  # EPSG:4978
    b_oct = rg.domain('b_oct')

    c_oct = rg.domain('c_oct')
    # 57.1255027778, -156.8763111111 APNP (alaska national park)
    # 70.1500, -148.4500 Tundra Mosaic NEEDS Alaska!
    # 37.42222892438389, -122.08531190998633 Googleplex
    # 37.815782046081814, -122.04091803913096 Las Trampas Ca.
    # 29.9511, -90.0715  New Orleans, LA
    # 37.5407, -77.4360  City-Centre of Richmond, Virginia
    # 44.4280, -110.5885 Yellowstone area, WY (44.4280, -110.5885)
    # 25.4687, -80.4776 Everglades edge, FL (Homestead)
    # 33.84118870718737, -109.96478418965168 Whiteriver
    # Here we are using A given point at the centre of the map.
    focus = "APNP"
    size, size_str = 1_000.0, '1k'        # width/height of map, in metres.
    rcc = Points(np.atleast_2d([57.1255027778, -156.8763111111]), g_gcd)
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
    for layer in range(12, 14):
        pts = tri_grid_clipped(level=layer+3, mode=int(mo[0]), bbox=bbox)
        bt = Points(pts, b_oct, components=bhx.components[0])
        smp = rg.project(bt, [b_oct, c_oct, c_ell, g_wkt])
        bt.samples, px_id = sample_nlcd_gdal(ds, smp.coords)
        bt.px_id = px_id   # de-dupes
        hx_pts, vals = hex_poly_layer(bt, layers=layer, reducer=mode_u8)
        plot_hex(hx_pts, f'{size_str}{layer}{focus}', bbox=bbox, values=vals, lut=nlcd_lut)
