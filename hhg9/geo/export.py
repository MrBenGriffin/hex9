# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
GIS export for Hex9 — GeoPackage (vector) and GeoTIFF (raster).

Provides a pure-data container (:class:`HexLayer`) and writers/readers for
standard GIS formats, with no matplotlib dependency.

**Public API:**

* :class:`HexLayer`           — pure-data container for one H9 level's geometry and attributes.
* :func:`from_composed`       — build a :class:`HexLayer` from a rendering-pipeline
                                :class:`~hhg9.rendering.composition.ComposedLayer`.
* :func:`layers_to_gpkg`      — write a list of :class:`HexLayer` to a GeoPackage file.
* :func:`load_gpkg`           — load a GeoPackage written by :func:`layers_to_gpkg`.
* :func:`backdrop_to_geotiff` — export a plate-carrée image to a georeferenced GeoTIFF.
* :func:`net_to_geotiff`      — export an n_oct-space backdrop to GeoTIFF with H9 metadata.
* :func:`load_net_geotiff`    — load a GeoTIFF written by :func:`net_to_geotiff`.

Requires ``osgeo`` (GDAL ≥ 3).  Install via::

    conda install -c conda-forge gdal
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    ogr.UseExceptions()
except ImportError as _e:
    raise ImportError(
        "hhg9.geo.export requires osgeo (GDAL/OGR ≥ 3). "
        "Install via:  conda install -c conda-forge gdal"
    ) from _e


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class HexLayer:
    """
    Pure-data container for one H9 level's geometry and attributes.

    Coordinate conventions
    ~~~~~~~~~~~~~~~~~~~~~~
    ``crs='g_gcd'``
        ``polys`` and ``ctrs`` use ``[lat, lon]`` order (GeneralGCD convention).
        OGR geometry is written as ``(lon, lat)`` per GIS convention.

    ``crs='n_oct'``
        ``polys`` and ``ctrs`` use ``[x, y]`` in n_oct lattice units.
        OGR geometry is written as ``(x, y)`` directly; no EPSG CRS is set.

    Attributes:
        level:            H9 level depth.
        name:             OGR layer name, e.g. ``'h9_L05'``.
        crs:              ``'g_gcd'`` or ``'n_oct'``.
        polys:            ``(H, 6, 2)`` float64 polygon vertex coordinates.
        ctrs:             ``(H, 2)`` float64 centroid coordinates.
        addresses:        ``(H,)`` str — full H9 address per hex.  Empty strings
                          when the source :class:`~hhg9.rendering.composition.LayerSpec`
                          did not have ``labels`` enabled.
        parent_addresses: ``(H,)`` str — address of the parent hex (``addr[:-1]``).
        key_tails:        ``(H,)`` uint8 — geometric type key (c2 category + net_mode).
        fields:           ``{name: NDArray}`` per-hex attribute arrays.
                          Each value is ``(H,)`` or ``(H, K)``; 2-D arrays are
                          written to GeoPackage as ``{name}_0``, ``{name}_1``, …
    """
    level:            int
    name:             str
    crs:              str
    polys:            NDArray          # (H, 6, 2)
    ctrs:             NDArray          # (H, 2)
    addresses:        NDArray          # (H,) str
    parent_addresses: NDArray          # (H,) str
    key_tails:        NDArray          # (H,) uint8
    fields:           dict             # {name: NDArray}


# ---------------------------------------------------------------------------
# from_composed adapter
# ---------------------------------------------------------------------------

def from_composed(cl, reg, crs: str = 'g_gcd', fields=None) -> HexLayer:
    """
    Build a :class:`HexLayer` from a rendering-pipeline
    :class:`~hhg9.rendering.composition.ComposedLayer`.

    Args:
        cl:     :class:`~hhg9.rendering.composition.ComposedLayer` from
                :class:`~hhg9.rendering.composition.Compositor`.
        reg:    The :class:`~hhg9.Registrar` used to set up the composition.
        crs:    Output coordinate system — ``'g_gcd'`` for WGS84 lat/lon
                (EPSG:4326) or ``'n_oct'`` for raw n_oct pixel space.
        fields: Optional ``{name: NDArray}`` mapping for per-hex attributes.
                If *None* and ``cl.values`` is not None, wrapped automatically
                as ``{'values': cl.values}``.  Pass ``{}`` to suppress.

    Returns:
        :class:`HexLayer` ready for :func:`layers_to_gpkg`.

    Note:
        Address columns require that the source
        :class:`~hhg9.rendering.composition.LayerSpec` was created with
        ``labels=True``.  If ``cl.labels`` is ``None``, empty string arrays
        are written to the GeoPackage.
    """
    n_oct = cl.ctrs.domain

    if crs == 'g_gcd':
        b_oct = reg.domain('b_oct')
        c_oct = reg.domain('c_oct')
        c_ell = reg.domain('c_ell')
        g_gcd = reg.domain('g_gcd')
        chain = [n_oct, b_oct, c_oct, c_ell, g_gcd]
        polys = reg.project(cl.verts, chain).coords.reshape(-1, 6, 2)
        ctrs  = reg.project(cl.ctrs,  chain).coords
        # Drop hexes whose projected vertices contain NaN/inf (seam vertices
        # that received OID_INVALID during n_oct→b_oct binning produce
        # degenerate AK results; those hexes are silently excluded).
        valid = (np.all(np.isfinite(polys.reshape(len(ctrs), -1)), axis=1) &
                 np.all(np.isfinite(ctrs), axis=1))
        if not valid.all():
            polys = polys[valid]
            ctrs  = ctrs[valid]
    elif crs == 'n_oct':
        polys = cl.verts.coords.reshape(-1, 6, 2)
        ctrs  = cl.ctrs.coords
        valid = np.ones(len(ctrs), dtype=bool)
    else:
        raise ValueError(f"crs must be 'g_gcd' or 'n_oct', got {crs!r}")

    H = len(ctrs)

    # Addresses: prefix + per-hex label suffix
    if cl.labels is not None:
        addresses = np.array([cl.prefix + str(lbl) for lbl in cl.labels])[valid]
    else:
        addresses = np.full(H, '', dtype=object)

    parent_addresses = np.array([addr[:-1] if addr else '' for addr in addresses])

    # Geometric type key
    if hasattr(cl, 'key_tails') and cl.key_tails is not None:
        key_tails = np.asarray(cl.key_tails, dtype=np.uint8)[valid]
    else:
        key_tails = np.zeros(H, dtype=np.uint8)

    # Fields
    if fields is None:
        flds = {'values': cl.values[valid] if cl.values is not None else None}
        if flds['values'] is None:
            flds = {}
    else:
        flds = {k: v[valid] if hasattr(v, '__len__') else v for k, v in fields.items()}

    level  = cl.spec.level
    suffix = '' if crs == 'g_gcd' else '_n'
    name   = f'h9_L{level:02d}{suffix}'

    return HexLayer(
        level=level,
        name=name,
        crs=crs,
        polys=polys,
        ctrs=ctrs,
        addresses=addresses,
        parent_addresses=parent_addresses,
        key_tails=key_tails,
        fields=flds,
    )


# ---------------------------------------------------------------------------
# GeoPackage writer — internal helpers
# ---------------------------------------------------------------------------

def _ogr_field_type(arr: NDArray) -> int:
    """Infer OGR field type from numpy dtype."""
    if np.issubdtype(arr.dtype, np.integer):
        return ogr.OFTInteger
    return ogr.OFTReal


def _add_layer_fields(ogr_lyr, layer: HexLayer) -> None:
    """Create fixed + dynamic OGR field definitions on *ogr_lyr*."""
    for fname, ftype in [
        ('h9_addr',   ogr.OFTString),
        ('h9_level',  ogr.OFTInteger),
        ('h9_parent', ogr.OFTString),
        ('h9_key',    ogr.OFTInteger),
    ]:
        ogr_lyr.CreateField(ogr.FieldDefn(fname, ftype))

    for key, val in layer.fields.items():
        arr = np.asarray(val)
        if arr.ndim == 1:
            ogr_lyr.CreateField(ogr.FieldDefn(key, _ogr_field_type(arr)))
        elif arr.ndim == 2:
            for k in range(arr.shape[1]):
                ogr_lyr.CreateField(ogr.FieldDefn(f'{key}_{k}', _ogr_field_type(arr[:, k])))
        else:
            raise ValueError(
                f"Field {key!r}: expected 1-D or 2-D array, got shape {arr.shape}"
            )


def _write_layer_features(ogr_lyr, layer: HexLayer) -> None:
    """Write one OGR feature per hex into *ogr_lyr*."""
    use_gcd  = (layer.crs == 'g_gcd')
    lyr_defn = ogr_lyr.GetLayerDefn()
    H        = len(layer.ctrs)

    # Flatten 2-D field arrays into per-column entries
    field_arrs: dict[str, NDArray] = {}
    for key, val in layer.fields.items():
        arr = np.asarray(val)
        if arr.ndim == 1:
            field_arrs[key] = arr
        elif arr.ndim == 2:
            for k in range(arr.shape[1]):
                field_arrs[f'{key}_{k}'] = arr[:, k]

    ogr_lyr.StartTransaction()
    for i in range(H):
        feat = ogr.Feature(lyr_defn)

        # Polygon geometry — close ring by repeating vertex 0
        ring   = ogr.Geometry(ogr.wkbLinearRing)
        verts6 = layer.polys[i]         # (6, 2)
        for v in verts6:
            if use_gcd:
                ring.AddPoint_2D(float(v[1]), float(v[0]))   # (lon, lat)
            else:
                ring.AddPoint_2D(float(v[0]), float(v[1]))   # (x, y)
        v0 = verts6[0]
        if use_gcd:
            ring.AddPoint_2D(float(v0[1]), float(v0[0]))
        else:
            ring.AddPoint_2D(float(v0[0]), float(v0[1]))

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feat.SetGeometry(poly)

        # Fixed attributes
        feat.SetField('h9_addr',   str(layer.addresses[i]))
        feat.SetField('h9_level',  int(layer.level))
        feat.SetField('h9_parent', str(layer.parent_addresses[i]))
        feat.SetField('h9_key',    int(layer.key_tails[i]))

        # Dynamic attributes
        for fname, arr in field_arrs.items():
            v = arr[i]
            if np.issubdtype(arr.dtype, np.integer):
                feat.SetField(fname, int(v))
            else:
                feat.SetField(fname, float(v))

        ogr_lyr.CreateFeature(feat)

    ogr_lyr.CommitTransaction()


# ---------------------------------------------------------------------------
# GeoPackage writer — public
# ---------------------------------------------------------------------------

def layers_to_gpkg(
        layers: list[HexLayer],
        path: str,
        layout: str = 'butterfly',
) -> None:
    """
    Write a list of :class:`HexLayer` objects to a single GeoPackage file.

    One OGR polygon layer is created per :class:`HexLayer`.  An auxiliary
    ``h9_metadata`` attribute table is appended to support round-tripping
    via :func:`load_gpkg`.

    Args:
        layers: Ordered list of :class:`HexLayer` to export.
        path:   Output GeoPackage path (``*.gpkg``).  Overwritten if it exists.
        layout: H9 net layout name stored in ``h9_metadata`` (default
                ``'butterfly'``).
    """
    if os.path.exists(path):
        os.remove(path)

    driver = ogr.GetDriverByName('GPKG')
    ds     = driver.CreateDataSource(path)

    # WGS84 spatial reference for g_gcd layers.
    # Derive from EPSG:4978 (ECEF) via CloneGeogCS() so that AUTHORITY["EPSG","4326"]
    # nodes are present — ImportFromEPSG(4326) alone can omit them in some GDAL builds,
    # causing QGIS to report "Custom CRS" and preventing Mollweide reprojection.
    sr_ecef = osr.SpatialReference()
    sr_ecef.ImportFromEPSG(4978)
    sr_4326 = osr.SpatialReference()
    sr_4326.ImportFromEPSG(4326)

    for layer in layers:
        sr = None
        match layer.crs:
            case 'g_gcd':
                sr = sr_4326
            case 'c_ell':
                sr = sr_ecef
            case _:
                pass
        ogr_lyr = ds.CreateLayer(layer.name, srs=sr, geom_type=ogr.wkbPolygon)
        _add_layer_fields(ogr_lyr, layer)
        _write_layer_features(ogr_lyr, layer)

    # h9_metadata — non-spatial attribute table
    meta_lyr = ds.CreateLayer('h9_metadata', geom_type=ogr.wkbNone)
    for fname, ftype in [
        ('layer_name', ogr.OFTString),
        ('layout',     ogr.OFTString),
        ('crs',        ogr.OFTString),
        ('level',      ogr.OFTInteger),
        ('bbox_n',     ogr.OFTString),
    ]:
        meta_lyr.CreateField(ogr.FieldDefn(fname, ftype))

    meta_defn = meta_lyr.GetLayerDefn()
    meta_lyr.StartTransaction()
    for layer in layers:
        sr = None
        match layer.crs:
            case 'g_gcd':
                sr = sr_4326.GetName()
            case 'c_ell':
                sr = sr_ecef.GetName()
            case _:
                sr = layer.crs
        feat = ogr.Feature(meta_defn)
        feat.SetField('layer_name', layer.name)
        feat.SetField('layout',     layout)
        feat.SetField('crs',        layer.crs)
        feat.SetField('level',      layer.level)
        feat.SetField('bbox_n',     '')
        meta_lyr.CreateFeature(feat)
    meta_lyr.CommitTransaction()

    ds.FlushCache()
    ds = None


# ---------------------------------------------------------------------------
# GeoPackage loader
# ---------------------------------------------------------------------------

def load_gpkg(path: str) -> list[HexLayer]:
    """
    Load a GeoPackage saved by :func:`layers_to_gpkg`.

    Args:
        path: GeoPackage file path.

    Returns:
        List of :class:`HexLayer` in original layer order.

    Note:
        Centroids are recomputed from OGR geometry and may differ from the
        original ``ctrs`` values by small floating-point amounts.
    """
    ds = ogr.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Could not open {path!r}")

    # Read h9_metadata table
    meta_map: dict[str, dict] = {}
    meta_lyr = ds.GetLayerByName('h9_metadata')
    if meta_lyr is not None:
        for feat in meta_lyr:
            lname = feat.GetField('layer_name')
            meta_map[lname] = {
                'layout': feat.GetField('layout'),
                'crs':    feat.GetField('crs'),
                'level':  feat.GetField('level'),
                'bbox_n': feat.GetField('bbox_n'),
            }

    results: list[HexLayer] = []
    fixed_names = {'h9_addr', 'h9_level', 'h9_parent', 'h9_key'}

    for i in range(ds.GetLayerCount()):
        ogr_lyr = ds.GetLayerByIndex(i)
        lname   = ogr_lyr.GetName()
        if lname == 'h9_metadata':
            continue

        meta    = meta_map.get(lname, {})
        crs     = meta.get('crs', 'g_gcd')
        level   = meta.get('level', 0) or 0
        use_gcd = (crs == 'g_gcd')

        lyr_defn = ogr_lyr.GetLayerDefn()
        dyn_names = [
            lyr_defn.GetFieldDefn(j).GetName()
            for j in range(lyr_defn.GetFieldCount())
            if lyr_defn.GetFieldDefn(j).GetName() not in fixed_names
        ]

        polys_list: list[NDArray] = []
        ctrs_list:  list[NDArray] = []
        addrs:   list[str] = []
        parents: list[str] = []
        keys:    list[int] = []
        dyn_data: dict[str, list] = {n: [] for n in dyn_names}

        for feat in ogr_lyr:
            geom = feat.GetGeometryRef()
            ring = geom.GetGeometryRef(0)
            n_pts = ring.GetPointCount()
            pts = []
            for k in range(n_pts - 1):   # skip closing repeat
                x, y = ring.GetX(k), ring.GetY(k)
                pts.append([y, x] if use_gcd else [x, y])
            polys_list.append(np.array(pts, dtype=np.float64))

            ctr = geom.Centroid()
            cx, cy = ctr.GetX(), ctr.GetY()
            ctrs_list.append([cy, cx] if use_gcd else [cx, cy])

            addrs.append(str(feat.GetField('h9_addr') or ''))
            parents.append(str(feat.GetField('h9_parent') or ''))
            keys.append(int(feat.GetField('h9_key') or 0))
            for n in dyn_names:
                dyn_data[n].append(feat.GetField(n))

        # Group name_0, name_1, ... back into 2-D arrays
        grouped: dict[str, NDArray] = {}
        consumed: set[str] = set()
        for n in dyn_names:
            if n in consumed:
                continue
            if n.endswith('_0'):
                base = n[:-2]
                k, cols = 0, []
                while f'{base}_{k}' in dyn_data:
                    cols.append(np.array(dyn_data[f'{base}_{k}']))
                    consumed.add(f'{base}_{k}')
                    k += 1
                if k > 1:
                    grouped[base] = np.stack(cols, axis=1)
                else:
                    grouped[n] = np.array(dyn_data[n])
                    consumed.add(n)
            else:
                grouped[n] = np.array(dyn_data[n])
                consumed.add(n)

        results.append(HexLayer(
            level=level,
            name=lname,
            crs=crs,
            polys=np.array(polys_list, dtype=np.float64),
            ctrs=np.array(ctrs_list,  dtype=np.float64),
            addresses=np.array(addrs),
            parent_addresses=np.array(parents),
            key_tails=np.array(keys, dtype=np.uint8),
            fields=grouped,
        ))

    return results


# ---------------------------------------------------------------------------
# GeoTIFF writers — internal helper
# ---------------------------------------------------------------------------

def _prepare_raster(backdrop: NDArray) -> tuple[NDArray, int, int, int]:
    """Normalise backdrop to uint8 ``(H, W, C)`` and return ``(arr, H, W, C)``."""
    arr = np.asarray(backdrop)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    H, W, C = arr.shape
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float64), 0.0, 1.0) * 255).astype(np.uint8)
    return arr, H, W, C


def _write_geotiff(arr: NDArray, gt: tuple, sr_wkt: Optional[str], path: str,
                   metadata: Optional[dict] = None, nodata: Optional[int] = None) -> None:
    """Write uint8 ``(H, W, C)`` array to a GeoTIFF."""
    H, W, C = arr.shape
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(path, W, H, C, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    if sr_wkt:
        ds.SetProjection(sr_wkt)
    if metadata:
        for domain, items in metadata.items():
            for key, val in items.items():
                ds.SetMetadataItem(key, val, domain)
    for b in range(C):
        band = ds.GetRasterBand(b + 1)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.WriteArray(arr[:, :, b])
    ds.FlushCache()
    ds = None


# ---------------------------------------------------------------------------
# GeoTIFF writers — public
# ---------------------------------------------------------------------------

def backdrop_to_geotiff(
        backdrop: NDArray,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        path: str,
) -> None:
    """
    Export a plate-carrée backdrop image to a georeferenced GeoTIFF (EPSG:4326).

    Args:
        backdrop: ``(H, W, C)`` float [0, 1] or uint8 image array.
                  Row 0 must correspond to *lat_max* (north / top edge).
                  Images loaded via ``matplotlib.image.imread`` satisfy this.
        lat_min:  Southern boundary latitude.
        lon_min:  Western boundary longitude.
        lat_max:  Northern boundary latitude.
        lon_max:  Eastern boundary longitude.
        path:     Output GeoTIFF path.
    """
    arr, H, W, _ = _prepare_raster(backdrop)
    lon_res = (lon_max - lon_min) / W
    lat_res = (lat_max - lat_min) / H
    gt = (lon_min, lon_res, 0.0, lat_max, 0.0, -lat_res)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    _write_geotiff(arr, gt, sr.ExportToWkt(), path)


def net_to_geotiff(
        backdrop: NDArray,
        bbox_n: tuple,
        path: str,
        layout: str = 'butterfly',
        nodata: int = 0,
) -> None:
    """
    Export an n_oct-space backdrop to a GeoTIFF with H9 metadata.

    The *backdrop* is typically the array returned directly by
    :func:`~hhg9.rendering.composition.make_backdrop`, whose row 0 is at the
    *bottom* of the bounding box (y = *bt*).  This function flips rows
    internally so that GDAL row 0 corresponds to the top (y = *tp*).

    H9-specific metadata is stored in the ``'H9'`` GDAL metadata domain so
    the file can be round-tripped with :func:`load_net_geotiff`.

    Args:
        backdrop: ``(H, W, C)`` float [0, 1] or uint8 array from
                  :func:`~hhg9.rendering.composition.make_backdrop`.
        bbox_n:   ``(tp, lt, bt, rt)`` bounding box in n_oct units —
                  same convention as :func:`~hhg9.rendering.composition.make_backdrop`.
        path:     Output GeoTIFF path.
        layout:   H9 net layout name stored in H9 metadata (default ``'butterfly'``).
        nodata:   No-data pixel value set on each band (default ``0``).
    """
    tp, lt, bt, rt = bbox_n
    arr, H, W, _ = _prepare_raster(backdrop)
    # make_backdrop row 0 = bottom; GeoTIFF expects row 0 = top
    arr = np.ascontiguousarray(arr[::-1])
    lon_res = (rt - lt) / W
    lat_res = (tp - bt) / H
    gt = (lt, lon_res, 0.0, tp, 0.0, -lat_res)

    meta = {'H9': {
        'LAYOUT': layout,
        'BBOX_N': f'{tp},{lt},{bt},{rt}',
    }}
    _write_geotiff(arr, gt, None, path, metadata=meta, nodata=nodata)


# ---------------------------------------------------------------------------
# GeoTIFF loader
# ---------------------------------------------------------------------------

def load_net_geotiff(path: str) -> tuple[NDArray, tuple, str]:
    """
    Load a GeoTIFF saved by :func:`net_to_geotiff`.

    For GeoTIFFs without H9 metadata (e.g. written by :func:`backdrop_to_geotiff`),
    ``layout`` is returned as ``'g_gcd'`` and ``bbox_n`` is derived from the
    GeoTransform.

    Args:
        path: GeoTIFF file path.

    Returns:
        ``(array, bbox_n, layout)`` where *array* is ``(H, W, C)`` uint8,
        *bbox_n* is ``(tp, lt, bt, rt)``, and *layout* is the H9 net name
        (or ``'g_gcd'``).
    """
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Could not open {path!r}")

    layout  = ds.GetMetadataItem('LAYOUT', 'H9')
    bbox_str = ds.GetMetadataItem('BBOX_N', 'H9')

    if layout and bbox_str:
        tp, lt, bt, rt = (float(v) for v in bbox_str.split(','))
        bbox_n = (tp, lt, bt, rt)
    else:
        layout = 'g_gcd'
        gt = ds.GetGeoTransform()
        lon_min, lon_res, _, lat_max, _, neg_lat_res = gt
        W, H = ds.RasterXSize, ds.RasterYSize
        lon_max = lon_min + lon_res * W
        lat_min = lat_max + neg_lat_res * H
        bbox_n  = (lat_max, lon_min, lat_min, lon_max)

    C    = ds.RasterCount
    bands = [ds.GetRasterBand(b + 1).ReadAsArray() for b in range(C)]
    arr  = np.stack(bands, axis=-1)   # (H, W, C)

    # net_to_geotiff flipped rows on write; restore to make_backdrop convention
    if layout != 'g_gcd':
        arr = np.ascontiguousarray(arr[::-1])

    return arr, bbox_n, layout
