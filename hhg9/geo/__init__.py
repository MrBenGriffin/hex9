# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Optional geographic I/O adapters for Hex9.

Submodules require external libraries (e.g. GDAL/OGR) that are not part of
the core Hex9 dependencies.  Import only what you need:

    from hhg9.geo.gdal import (Wkt, Wkt_4978, sample_gdal,
                               estimate_backdrop_size,
                               wkt_geotiff_meta, make_geotiff_sampler, load_wkt_geotiff)
    from hhg9.geo.export import HexLayer, from_composed, layers_to_gpkg, load_gpkg
    from hhg9.geo.export import backdrop_to_geotiff, net_to_geotiff, load_net_geotiff

:mod:`hhg9.geo.gdal`
    GDAL/OGR domain and projection bridge + raster sampler.

:mod:`hhg9.geo.export`
    Pure-data :class:`~hhg9.geo.export.HexLayer` container and writers/readers
    for GeoPackage (vector) and GeoTIFF (raster).  No matplotlib dependency.

:mod:`hhg9.geo.vector`
    Format-agnostic vector geometry utilities: seam-splitting projected paths,
    geodesic path projection, Tissot circle generation, graticule generation.
"""
