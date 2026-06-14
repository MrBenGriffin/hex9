from osgeo import ogr
from hhg9.geo import gdal

if __name__ == '__main__':
    ogr.UseExceptions()
    src = ogr.Open('output/ex0302_uk.gpkg')
    drv = ogr.GetDriverByName('GPKG')
    dst = drv.CreateDataSource('output/h9_lines.gpkg')
    # Buffer widths in degrees (~lon); tune to taste
    buffers = {'h9_L04': 0.001, 'h9_L05': 0.0005}

    for lname in ['h9_L04', 'h9_L05']:
        lyr = src.GetLayerByName(lname)
        buf = buffers[lname]
        out = dst.CreateLayer(lname, srs=lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)
        for feat in lyr:
            ring = feat.GetGeometryRef().GetGeometryRef(0)
            line = ogr.Geometry(ogr.wkbLineString)
            for k in range(ring.GetPointCount()):
                line.AddPoint_2D(ring.GetX(k), ring.GetY(k))
            f = ogr.Feature(out.GetLayerDefn())
            f.SetGeometry(line.Buffer(buf))
            out.CreateFeature(f)
    dst.FlushCache()

# Then burn (run from project root):
# cp output/ex0302_uk_osm.tif output/ex0302_overlay.tif
# -b flags required: without them gdal_rasterize only burns band 1 (Red), leaving G+B → cyan
# Black:
# gdal_rasterize -b 1 -b 2 -b 3 -burn 0   -burn 0   -burn 0   -l h9_L04 output/h9_lines.gpkg output/ex0302_overlay.tif
# gdal_rasterize -b 1 -b 2 -b 3 -burn 0   -burn 0   -burn 0   -l h9_L05 output/h9_lines.gpkg output/ex0302_overlay.tif
# Dark green:
# gdal_rasterize -b 1 -b 2 -b 3 -burn 0   -burn 100 -burn 0   -l h9_L04 output/h9_lines.gpkg output/ex0302_overlay.tif
# gdal_rasterize -b 1 -b 2 -b 3 -burn 0   -burn 80  -burn 0   -l h9_L05 output/h9_lines.gpkg output/ex0302_overlay.tif
