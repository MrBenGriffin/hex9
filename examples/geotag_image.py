# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Georeference a plain photo (JPG/PNG) by *placing* it at a point with a known
real-world ground size, writing a GeoTIFF that QGIS — and ex0098's
make_bm_source — can read.

This is NOT EXIF geotagging.  EXIF GPS records where the *camera* stood (one
point); it cannot drape the image over the ground.  Here we fabricate a proper
geotransform: centre lon/lat + ground width (metres) + optional heading.  Use
this when the subject is too fine to have map features for GCP matching (a fly
on a lawn, a coin, a leaf) — you know roughly where it is and how big it is.

Output CRS is EPSG:4326 (lon/lat).  For a footprint of centimetres-to-metres the
local equirectangular approximation (metres→degrees at the centre latitude) is
far below pixel error, and 4326 drops straight into the ex0098 sampler chain.

Usage:
    python geotag_image.py fly.jpg src/fly.tif --width-m 0.20
    python geotag_image.py fly.jpg src/fly.tif --width-m 0.20 --heading-deg 35
    python geotag_image.py fly.jpg src/fly.tif --width-m 0.20 --lon 1.6403 --lat 49.0988

--width-m   ground width the photo's full pixel width spans, in metres.
--height-m  ground height; omit to derive from image aspect ratio.
--heading-deg  compass bearing the photo's "up" (top edge) points toward, cw
            from north.  0 = north-up.  A handheld macro is usually arbitrary;
            eyeball it or leave 0 and rotate later in QGIS.
--lon/--lat centre; defaults to the ex0098 POI.
"""
import argparse
import math

import numpy as np
from osgeo import gdal, osr
from PIL import Image

# Default centre = the ex0098 point of interest ("box on lawn").
#DEF_LON, DEF_LAT = 1.6402506890084543, 49.09879599485989
DEF_LON, DEF_LAT = 1.640250018, 49.098794825,

def geotag(in_path, out_path, *, lon, lat, width_m, height_m=None,
           heading_deg=0.0):
    """Write `in_path` as a georeferenced GeoTIFF placed at (lon, lat)."""
    img = np.asarray(Image.open(in_path).convert('RGB'))
    h_px, w_px = img.shape[:2]
    if height_m is None:
        height_m = width_m * (h_px / w_px)

    # Metres per pixel along the image's own axes.
    mppx = width_m / w_px
    mppy = height_m / h_px

    # Local metres-per-degree at the centre latitude (equirectangular).
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

    # Image axes → ENU.  row increases downward; "up" (row-) points along the
    # heading.  right(col+) = heading rotated 90° cw; down(row+) = -up.
    th = math.radians(heading_deg)
    # right (col+) in (East, North):
    rE, rN = math.cos(th), -math.sin(th)
    # down  (row+) in (East, North):
    dE, dN = -math.sin(th), -math.cos(th)

    # Affine: x=lon, y=lat as functions of (col, row).
    gt1 = mppx * rE / m_per_deg_lon   # d lon / d col
    gt2 = mppy * dE / m_per_deg_lon   # d lon / d row
    gt4 = mppx * rN / m_per_deg_lat   # d lat / d col
    gt5 = mppy * dN / m_per_deg_lat   # d lat / d row
    # Pin the image centre (cx, cy) to (lon, lat).
    cx, cy = w_px / 2.0, h_px / 2.0
    gt0 = lon - cx * gt1 - cy * gt2
    gt3 = lat - cx * gt4 - cy * gt5

    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(out_path, w_px, h_px, 3, gdal.GDT_Byte,
                    options=['COMPRESS=DEFLATE', 'PHOTOMETRIC=RGB'])
    ds.SetGeoTransform((gt0, gt1, gt2, gt3, gt4, gt5))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(3):
        ds.GetRasterBand(b + 1).WriteArray(img[:, :, b])
    ds.FlushCache()
    ds = None

    corner = (gt0, gt3)
    print(f'  wrote {out_path}  ({w_px}×{h_px}px, {width_m:.3f}×{height_m:.3f} m)')
    print(f'  centre lon/lat = {lon:.7f}, {lat:.7f}   heading = {heading_deg}°')
    print(f'  ul corner lon/lat = {corner[0]:.7f}, {corner[1]:.7f}')


if __name__ == '__main__':
    gdal.UseExceptions()

    fly_lon, fly_lat = 1.640250000, 49.0987953450  # 49.0987953200 +100 +100 +50
    geotag('src/fly2.png', 'src/fly_1mm.tif', width_m=0.257212475633528, lon=fly_lon, lat=fly_lat)

    # c_lon, c_lat = 1.640250018, 49.098794825,
    # geotag('src/171c.png', 'src/171c.tif', heading_deg=0, width_m=15, lon=c_lon, lat=c_lat)
    # geotag(a.input, a.output, lon=a.lon, lat=a.lat,
    #        width_m=a.width_m, height_m=a.height_m, heading_deg=a.heading_deg)
