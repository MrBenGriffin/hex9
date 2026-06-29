# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Tests for the GDAL-backed geo I/O (``hhg9.geo``):

  * export.py  — HexLayer → GeoPackage → HexLayer round-trip;
                 backdrop_to_geotiff → load_net_geotiff round-trip
  * vector.py  — tissot_circle, graticule_paths, split_path_at_seams

Skips cleanly when GDAL is not importable.  Files are written to pytest's
tmp_path so nothing lands in the project tree.
"""
import numpy as np
import pytest

pytest.importorskip("osgeo", reason="GDAL (osgeo) not installed")

from hhg9 import Registrar
from hhg9.geo.export import (
    HexLayer, layers_to_gpkg, load_gpkg, backdrop_to_geotiff, load_net_geotiff,
)
import hhg9.geo.vector as V
import hhg9.algorithms.distance as D


def _hexagon(clat, clon, r=0.3):
    a = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    return np.column_stack([clat + r * np.sin(a), clon + r * np.cos(a)])


# ---------------------------------------------------------------------------
# GeoPackage round-trip
# ---------------------------------------------------------------------------

def test_gpkg_roundtrip(tmp_path):
    polys = np.stack([_hexagon(51, 0), _hexagon(40, -74), _hexagon(-33, 151)])
    ctrs = polys.mean(axis=1)
    layer = HexLayer(
        level=5, name='h9_L05', crs='g_gcd', polys=polys, ctrs=ctrs,
        addresses=np.array(['abc', 'def', 'ghi'], dtype=object),
        parent_addresses=np.array(['ab', 'de', 'gh'], dtype=object),
        key_tails=np.array([0, 1, 2], dtype=np.uint8),
        fields={'values': np.array([1.5, 2.5, 3.5])},
    )
    path = str(tmp_path / "out.gpkg")
    layers_to_gpkg([layer], path)

    out = load_gpkg(path)
    assert len(out) == 1
    got = out[0]
    assert len(got.ctrs) == 3
    assert got.polys.shape == (3, 6, 2)
    assert got.addresses.tolist() == ['abc', 'def', 'ghi']
    np.testing.assert_allclose(got.fields['values'], [1.5, 2.5, 3.5])
    # centroids are recomputed from geometry but must stay close
    assert np.abs(got.ctrs - ctrs).max() < 0.1


# ---------------------------------------------------------------------------
# GeoTIFF round-trip
# ---------------------------------------------------------------------------

def test_backdrop_geotiff_roundtrip(tmp_path):
    img = (np.random.default_rng(0).random((10, 20, 3)) * 255).astype(np.uint8)
    path = str(tmp_path / "out.tif")
    backdrop_to_geotiff(img, lat_min=-10, lon_min=-20, lat_max=10, lon_max=20, path=path)

    arr, bbox, layout = load_net_geotiff(path)
    assert arr.shape == (10, 20, 3)
    assert layout == 'g_gcd'
    # bbox carries the four corner values
    assert set(np.round(bbox, 3)) == {10.0, -20.0, -10.0, 20.0}


# ---------------------------------------------------------------------------
# vector.py
# ---------------------------------------------------------------------------

def test_tissot_circle_radius():
    lat, lon, r = 51.5, -0.1, 100_000.0
    circ = V.tissot_circle(lat, lon, radius_m=r, n_pts=32)
    assert circ.shape == (32, 2)
    # every vertex is ~r metres from the centre (geodesic)
    centre = np.full_like(circ, [lat, lon])
    dists = D.ell_distance(centre, circ)
    np.testing.assert_allclose(dists, r, rtol=1e-3)


def test_graticule_paths_structure():
    paths = V.graticule_paths(lat_step=30.0, lon_step=30.0, n_pts=181)
    assert len(paths) > 0
    assert all(p.shape[1] == 2 for p in paths)
    # parallels use n_pts samples
    assert any(p.shape[0] == 181 for p in paths)


def test_split_path_at_seams():
    # a path with one large jump in the middle splits into two segments
    path = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    segs = V.split_path_at_seams(path, threshold=1.0)
    assert len(segs) == 2
    # a smooth path stays in one piece
    smooth = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    assert len(V.split_path_at_seams(smooth, threshold=1.0)) == 1


def test_clip_polygon_to_octants():
    pytest.importorskip("shapely", reason="shapely not installed")
    from shapely.geometry import Polygon

    # ring as (lat, lon); a box straddling the prime meridian (lon 0)
    ring = np.array([[10.0, -10.0], [10.0, 10.0], [30.0, 10.0], [30.0, -10.0]])
    pieces = V.clip_polygon_to_octants(ring)
    assert len(pieces) == 2                       # split at lon 0 into 2 octants

    src_area = Polygon(ring[:, ::-1]).area
    tot = 0.0
    for p in pieces:
        lat, lon = p[:, 0], p[:, 1]
        # each piece lies in a single octant band
        assert lat.min() >= 0.0 and lat.max() <= 90.0
        assert (lon.min() >= -90.0 and lon.max() <= 0.0) or \
               (lon.min() >= 0.0 and lon.max() <= 90.0)
        tot += Polygon(p[:, ::-1]).area
    # clipping conserves total area
    np.testing.assert_allclose(tot, src_area, rtol=1e-6)

    # a ring wholly inside one octant comes back as a single piece
    inside = np.array([[10.0, 10.0], [10.0, 40.0], [40.0, 40.0], [40.0, 10.0]])
    one = V.clip_polygon_to_octants(inside)
    assert len(one) == 1
    np.testing.assert_allclose(Polygon(one[0][:, ::-1]).area,
                               Polygon(inside[:, ::-1]).area, rtol=1e-6)


def test_clip_polygon_to_octants_antimeridian():
    # a small ring centred on the ±180 antimeridian must NOT become a
    # globe-spanning polygon — each piece stays contiguous (lon-span << 360).
    pytest.importorskip("shapely", reason="shapely not installed")
    circ = V.tissot_circle(20.0, 180.0, radius_m=500_000.0, n_pts=72)
    pieces = V.clip_polygon_to_octants(circ)
    assert len(pieces) >= 2                       # split across the antimeridian
    for p in pieces:
        span = p[:, 1].max() - p[:, 1].min()
        assert span < 90.0                        # contiguous, not wrapped
        # each piece sits against the ±180 meridian
        assert np.isclose(np.abs(p[:, 1]).max(), 180.0, atol=1e-6)


# ---------------------------------------------------------------------------
# gdal.py — WKT GeoTIFF loading / CRS registration
# ---------------------------------------------------------------------------

def _write_4326_tif(tmp_path):
    img = (np.random.default_rng(1).random((8, 16, 3)) * 255).astype(np.uint8)
    path = str(tmp_path / "wkt.tif")
    backdrop_to_geotiff(img, lat_min=-40, lon_min=-80, lat_max=40, lon_max=80, path=path)
    return path


def test_load_wkt_geotiff_without_registrar(tmp_path):
    from hhg9.geo.gdal import load_wkt_geotiff
    arr, bbox, wkt = load_wkt_geotiff(_write_4326_tif(tmp_path))
    assert arr.shape == (8, 16, 3)
    assert isinstance(wkt, str) and ("WGS" in wkt.upper() or "4326" in wkt)
    # bbox is (top, left, bottom, right)
    assert set(np.round(bbox, 3)) == {40.0, -80.0, -40.0, 80.0}


def test_load_wkt_geotiff_registers_domain(tmp_path):
    from hhg9.geo.gdal import load_wkt_geotiff, Wkt
    reg = Registrar()
    arr, bbox, dom = load_wkt_geotiff(_write_4326_tif(tmp_path), reg, name='g_lc')
    assert isinstance(dom, Wkt)
    assert dom.name == 'g_lc'
    assert reg.domain('g_lc') is dom         # joined the project's domain registry


def test_wkt_geotiff_meta_streams_without_pixels(tmp_path):
    from hhg9.geo.gdal import wkt_geotiff_meta
    ds, bbox, wkt = wkt_geotiff_meta(_write_4326_tif(tmp_path))
    assert ds.RasterCount == 3
    assert set(np.round(bbox, 3)) == {40.0, -80.0, -40.0, 80.0}
    assert isinstance(wkt, str)


# ---------------------------------------------------------------------------
# export.py — n_oct GeoTIFF + n_oct GeoPackage (no rendering pipeline needed)
# ---------------------------------------------------------------------------

def test_net_geotiff_roundtrip(tmp_path):
    from hhg9.geo.export import net_to_geotiff, load_net_geotiff
    img = (np.random.default_rng(2).random((20, 30, 3)) * 255).astype(np.uint8)
    bbox_n = (3.674, 0.0, 0.0, 4.95)             # (top, left, bottom, right) n_oct units
    path = str(tmp_path / "net.tif")
    net_to_geotiff(img, bbox_n, path, layout='mortar')
    arr, bbox, layout = load_net_geotiff(path)
    assert arr.shape == (20, 30, 3)
    assert layout == 'mortar'                     # H9 metadata round-trips
    assert set(np.round(bbox, 3)) == {3.674, 0.0, 4.95}


def test_gpkg_n_oct_crs_with_2d_fields(tmp_path):
    """n_oct-CRS layers (no EPSG) with a 2-D field column round-trip."""
    polys = np.random.default_rng(0).random((3, 6, 2)) * 4.0
    ctrs = polys.mean(axis=1)
    layer = HexLayer(
        level=3, name='h9_L03', crs='n_oct', polys=polys, ctrs=ctrs,
        addresses=np.array(['a', 'b', 'c'], dtype=object),
        parent_addresses=np.array(['', '', ''], dtype=object),
        key_tails=np.zeros(3, np.uint8),
        fields={'rgb': np.random.default_rng(1).integers(0, 255, (3, 3))},
    )
    path = str(tmp_path / "noct.gpkg")
    layers_to_gpkg([layer], path, layout='mortar')
    out = load_gpkg(path)
    assert out[0].crs == 'n_oct'
    assert 'rgb' in out[0].fields            # 2-D field split into rgb_0/1/2 and reassembled
    assert len(out[0].ctrs) == 3


# ---------------------------------------------------------------------------
# vector.py — path projection / tile vertices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg_mod():
    return Registrar()


def test_project_path_returns_noct_subpaths(reg_mod):
    n_oct = reg_mod.domain('n_oct:butterfly')
    parallel = np.column_stack([np.full(20, 10.0), np.linspace(-50, 50, 20)])
    segs = V.project_path(parallel, reg_mod, n_oct, max_step_m=200_000.0)
    assert len(segs) >= 1
    assert all(s.shape[1] == 2 for s in segs)


def test_project_fill_dissolves_joins(reg_mod):
    # A small circle straddling the prime-meridian octant boundary is split by
    # clip_polygon_to_octants into 2 pieces; in butterfly that boundary is a
    # *join*, so project_fill_to_noct must union them back into ONE whole
    # polygon (not leave a visible split chord).
    pytest.importorskip("shapely", reason="shapely not installed")
    n_oct = reg_mod.domain('n_oct:butterfly')
    circ = V.tissot_circle(30.0, 0.0, radius_m=500_000.0, n_pts=72)
    assert len(V.clip_polygon_to_octants(circ)) == 2          # split by the clip
    fills = V.project_fill_to_noct(circ, reg_mod, n_oct, max_step_m=20_000.0)
    assert len(fills) == 1                                    # join dissolved
    assert fills[0].geom_type == 'Polygon' and fills[0].area > 0


def test_project_fill_confined_to_net(reg_mod):
    # A large ring filling a southern octant (around a cone point) must NOT
    # bleed into the net's angle-deficit "impossible space": every fill polygon
    # stays within the union of face triangles.
    pytest.importorskip("shapely", reason="shapely not installed")
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    n_oct = reg_mod.domain('n_oct:butterfly')
    # robust net: buffer each face triangle before union (avoids dropped faces)
    net = unary_union([Polygon(t).buffer(1e-7)
                       for polys in n_oct.face_polys.values() for t in polys])
    ring = np.array([[-80.0, -175.0], [-80.0, -95.0],
                     [-5.0, -95.0], [-5.0, -175.0]])      # southern octant box
    fills = V.project_fill_to_noct(ring, reg_mod, n_oct, max_step_m=50_000.0)
    assert len(fills) >= 1                                 # shapely Polygons
    bleed = sum(pg.difference(net).area for pg in fills)
    assert bleed < 1e-3                                    # confined to the net


def test_project_fill_preserves_holes(reg_mod):
    # A polygon-with-hole (e.g. the ocean's continent holes) must keep its hole
    # through clip→project→confine→union so the interior is cut out.
    pytest.importorskip("shapely", reason="shapely not installed")
    from shapely.geometry import Polygon
    n_oct = reg_mod.domain('n_oct:butterfly')
    # shapely polygon in (lon, lat) within a single octant, with a hole
    outer = [(10, 10), (80, 10), (80, 80), (10, 80)]
    hole = [(30, 30), (60, 30), (60, 60), (30, 60)]
    poly = Polygon(outer, [hole])
    fills = V.project_fill_to_noct(poly, reg_mod, n_oct, max_step_m=100_000.0)
    assert len(fills) >= 1
    assert any(len(p.interiors) >= 1 for p in fills)        # hole carried through


def test_tile_vertices_noct_dedup(reg_mod):
    n_oct = reg_mod.domain('n_oct:butterfly')
    tv = V.tile_vertices_noct(n_oct)
    assert tv.ndim == 2 and tv.shape[1] == 2
    # deduplicated to 1e-6: no two corners coincide
    for i in range(len(tv)):
        d = np.linalg.norm(tv[i + 1:] - tv[i], axis=1)
        assert np.all(d >= 1e-6)
