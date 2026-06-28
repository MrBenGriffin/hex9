# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Coverage for the parametric raster / pixel domains:

  * PlatePixel       (p_pix)            — affine pixel↔world grid + raster I/O
  * PlatePixelCarree (lon/lat raster)   — windowing / crop helpers
  * OctahedralNet    (n_oct:<layout>)   — flattened-net geometry + b_oct↔n_oct
  * NetPixel         (n_pix)            — raster grid aligned to a net layout

These domains are parameterised (extent / image dims / layout) which makes them
fiddly to set up; the tests focus on the raster round-trip (adopt→image),
the pixel↔world sizing inverses, the windowing round-trip, and the
b_oct↔n_oct / p_pix↔g_gcd projection round-trips.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points


@pytest.fixture(scope="module")
def reg():
    return Registrar()


# ===========================================================================
# PlatePixel (p_pix)
# ===========================================================================

def test_p_pix_adopt_image_roundtrip(reg):
    pp = reg.domain('p_pix')
    img = np.random.default_rng(0).integers(0, 256, size=(6, 8, 3)).astype(np.uint8)
    pts = pp.adopt(img, extent=(-10, -5, 10, 5))
    assert pts.coords.shape == (48, 2)
    assert pts.samples.shape == (48, 3)
    # Dense round-trip through the affine grid is exact.
    out = pp.image(pts, dims=(8, 6))
    np.testing.assert_array_equal(out, img)


def test_p_pix_extent_derives_pixel_size(reg):
    from hhg9.domains.plate_pixel import PlatePixel
    pp = PlatePixel(reg, width=20, height=10, extent=(0.0, 0.0, 40.0, 10.0))
    assert pp.pixel_size == (2.0, 1.0)
    assert pp.origin == (0.0, 0.0)


def test_p_pix_invalid_extent_raises(reg):
    from hhg9.domains.plate_pixel import PlatePixel
    with pytest.raises(ValueError):
        PlatePixel(reg, extent=(10.0, 0.0, 0.0, 10.0))   # xmax <= xmin


def test_p_pix_adopt_rejects_non_image(reg):
    pp = reg.domain('p_pix')
    with pytest.raises(ValueError):
        pp.adopt(np.zeros((4, 4)))                       # 2D, not H×W×C


def test_p_pix_valid_in_extent(reg):
    from hhg9.domains.plate_pixel import PlatePixel
    pp = PlatePixel(reg, extent=(-10.0, -5.0, 10.0, 5.0))
    res = pp.valid(np.array([[0.0, 0.0], [100.0, 0.0], [-10.0, -5.0]]))
    assert res[0] and not res[1] and res[2]      # inside, outside, on min-corner


def test_p_pix_set_grid_from_extent_and_pixel_size(reg):
    from hhg9.domains.plate_pixel import PlatePixel
    pp = PlatePixel(reg).set_grid(extent=(0.0, 0.0, 100.0, 50.0), pixel_size=(2.0, 2.0))
    assert pp.width == 50 and pp.height == 25
    assert pp.extent == (0.0, 0.0, 100.0, 50.0)


def test_p_pix_to_gcd_roundtrip(reg):
    """p_pix in world (lon/lat) mode projects to g_gcd and back exactly."""
    pp = reg.domain('p_pix')
    pts = pp.adopt(np.zeros((180, 360, 3), np.uint8), extent=(-180, -90, 180, 90))
    g = reg.project(pts, ['p_pix', 'g_gcd'])
    back = reg.project(g, ['g_gcd', 'p_pix'])
    assert np.abs(back.coords - pts.coords).max() == 0.0


# ===========================================================================
# PlatePixelCarree (lon/lat raster windowing)
# ===========================================================================

@pytest.fixture(scope="module")
def carree(reg):
    from hhg9.domains.plate_pixel_carree import PlatePixelCarree
    return PlatePixelCarree.full_sphere(reg, 360, 180)


def test_carree_full_sphere_extent(carree):
    assert carree.extent == (-180.0, -90.0, 180.0, 90.0)
    assert carree.pixel_size == (1.0, 1.0)


def test_carree_window_extent_roundtrip(carree):
    win = carree.bounds_to_windows((-10, -5, 10, 5))
    assert len(win) == 1
    assert carree.window_to_extent(win[0]) == (-10.0, -5.0, 10.0, 5.0)


def test_carree_dateline_splits_into_two_windows(carree):
    wins = carree.bounds_to_windows((170, -5, -170, 5), wrap="split")
    assert len(wins) == 2


def test_carree_dateline_error_mode(carree):
    with pytest.raises(ValueError):
        carree.bounds_to_windows((170, -5, -170, 5), wrap="error")


def test_carree_bad_lat_bounds_raise(carree):
    with pytest.raises(ValueError):
        carree.bounds_to_windows((-10, 5, 10, -5))       # lat_max <= lat_min


def test_carree_crop_shape_matches_window(carree):
    img = np.zeros((180, 360, 3), np.uint8)
    crops = carree.crop(img, (-10, -5, 10, 5))
    (crop, extent), = crops
    i0, i1, j0, j1 = carree.bounds_to_windows((-10, -5, 10, 5))[0]
    assert crop.shape == (i1 - i0, j1 - j0, 3)
    assert extent == (-10.0, -5.0, 10.0, 5.0)


# ===========================================================================
# OctahedralNet (n_oct:<layout>)
# ===========================================================================

@pytest.fixture(scope="module")
def n_oct(reg):
    return reg.domain('n_oct:mortar')


def test_n_oct_geometry(n_oct):
    assert n_oct.wi > 0 and n_oct.he > 0
    assert n_oct.ratio() == pytest.approx(n_oct.wi / n_oct.he)
    w, h = n_oct.image_dims(100)
    assert w > 0 and h > 0


def test_b_oct_n_oct_roundtrip(reg, n_oct):
    g = reg.domain('g_gcd')
    ll = np.array([[51.5, -0.12], [40.7, -74.0], [-33.9, 151.2]])
    chain = ['g_gcd', 'b_oct', n_oct.name]
    fwd = reg.project(Points(ll, domain=g), chain)
    assert fwd.oid is not None and np.all((fwd.oid >= 0) & (fwd.oid <= 7))
    back = reg.project(fwd, chain[::-1])
    assert np.abs(back.coords - ll).max() < 1e-10


def test_n_oct_pt_face_classifies(reg, n_oct):
    """pt_face returns a face oid for in-net points and 255 for far-outside ones."""
    from hhg9.base.points import OID_INVALID
    g = reg.domain('g_gcd')
    inside = reg.project(Points(np.array([[20.0, 30.0]]), domain=g),
                         ['g_gcd', 'b_oct', n_oct.name])
    oids = n_oct.pt_face(inside.coords)
    assert oids[0] != OID_INVALID
    far = n_oct.pt_face(np.array([[-1000.0, -1000.0]]))
    assert far[0] == OID_INVALID


# ===========================================================================
# NetPixel (n_pix)
# ===========================================================================

def test_n_pix_sizing_inverse(reg):
    npx = reg.domain('n_pix')
    assert npx.ratio() == pytest.approx(npx.n_oct.wi / npx.n_oct.he)
    dims = npx.image_dims(64)
    assert dims[0] > 0 and dims[1] > 0
    # dim_from_image inverts image_dims back to the triangle side length.
    assert npx.dim_from_image(*dims) == pytest.approx(64.0)


def test_n_pix_adopt_and_image_shapes(reg):
    npx = reg.domain('n_pix')
    img = np.zeros((10, 12, 3), np.uint8)
    pts = npx.adopt(img)
    assert pts.coords.shape == (120, 2)
    assert pts.domain is npx
    out = npx.image(pts, dims=(12, 10))
    assert out.shape == (10, 12, 3)
