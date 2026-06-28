# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Vector geometry utilities for projected H9 output.

These tools operate on paths and polygons after projection into a display
coordinate system (typically n_oct) and are independent of the output format
(SVG, GeoPackage, Shapefile, etc.).

Functions
---------
split_path_at_seams
    Split a projected path wherever consecutive vertices jump across a net seam.
project_path
    Densify a lat/lon path geodesically, project through the H9 pipeline,
    and split at net seams in one call.
tissot_circle
    Generate a small geodesic circle (Tissot indicatrix) centred on a
    given lat/lon as a densified (N, 2) lat/lon array.
graticule_paths
    Generate a full lat/lon graticule as a list of (N, 2) lat/lon arrays.
load_land_polygons
    Load a land-polygon shapefile (Natural Earth, OSM) and return rings as
    (lat°, lon°) arrays ready for project_path.
clip_polygon_to_octants
    Clip a lat/lon polygon ring to the 8 octahedral octants (lat/lon
    rectangles) so each resulting piece lies within a single octant and
    projects without crossing a net seam — the robust replacement for the
    seam-split + snap heuristic when rendering *filled* polygons.
is_c2_split
    True when a net layout subdivides octants into c2 sub-triangles (rhombus),
    for which octant-level clipping is not a fine enough fundamental to close
    fills.  Render guards should check this.
tile_vertices_noct
    Return the unique tile-corner positions in n_oct space for any net layout.
    For simple layouts these are the 6 octahedral face corners; for c2-split
    layouts (e.g. rhombus) also includes c2 sub-triangle joints.
"""

import numpy as np
from hhg9.h9 import H9K
from hhg9.algorithms.distance import densify_poly_geodesic_by_step, wgs84_offset


# ---------------------------------------------------------------------------
# Seam splitting
# ---------------------------------------------------------------------------

def split_path_at_seams(path_noct: np.ndarray,
                        threshold: float = H9K.derived.W / 9) -> list:
    """Split a projected path at net seams.

    Consecutive vertices whose n_oct distance exceeds *threshold* are assumed
    to have crossed a net seam (a face boundary between disconnected net
    regions).  The path is split just before each such jump.

    Works for all net layouts including rhombus — no face-adjacency knowledge
    required.  Any genuine seam jump is always ≥ W (= √2, the face edge
    length) regardless of layout; valid steps between densified geographic
    points are many times smaller.

    Parameters
    ----------
    path_noct : (N, 2) ndarray
        Path vertices already projected into n_oct coordinates.
    threshold : float
        Maximum valid consecutive step.  Default W/9 ≈ 0.157 (L2 hex-edge
        scale) is safe for paths densified at ≤ ~500 km geodesic step.
        Use W/27 ≈ 0.052 (L3 scale) for finer densification.
        Pass ``H9K.derived.W / (3**layer)`` to match a specific H9 layer.

    Returns
    -------
    list of ndarray
        Sub-paths, each at least 2 vertices, with no seam-spanning edges.
        Empty input returns an empty list.
    """
    if len(path_noct) < 2:
        return [path_noct] if len(path_noct) else []
    lens = np.hypot(*np.diff(path_noct, axis=0).T)
    cuts = np.where(lens >= threshold)[0] + 1
    return [s for s in np.split(path_noct, cuts) if len(s) >= 2]


# ---------------------------------------------------------------------------
# Path projection
# ---------------------------------------------------------------------------

def project_path(latlon: np.ndarray,
                 reg,
                 n_oct,
                 max_step_m: float = 100_000.0,
                 closed: bool = False,
                 threshold: float = None) -> list:
    """Densify, project, and seam-split a lat/lon path.

    Parameters
    ----------
    latlon : (M, 2) ndarray
        Path vertices as (lat°, lon°).
    reg : Registrar
    n_oct : OctahedralNet
        Target display domain.
    max_step_m : float
        Maximum geodesic step for densification (metres).  Smaller values give
        smoother projected curves at the cost of more vertices.
    closed : bool
        Treat the path as a closed polygon (last edge wraps to first).
    threshold : float or None
        Seam-split threshold passed to :func:`split_path_at_seams`.
        ``None`` uses the default (W/9).

    Returns
    -------
    list of (K, 2) ndarray
        Projected sub-paths in n_oct coordinates, seam-split.
    """
    from hhg9 import Points

    dense = densify_poly_geodesic_by_step(latlon, max_step_m=max_step_m,
                                          closed=closed)
    if len(dense) < 2:
        return []

    b_oct = n_oct.b_oct
    pts_g = Points(dense, domain='g_gcd')
    pts_b = reg.project(pts_g, ['g_gcd', b_oct])
    ok = b_oct.valid(pts_b)
    if not np.any(ok):
        return []
    pts_b = pts_b.select(ok)
    dense  = dense[ok]

    pts_n  = reg.project(pts_b, [b_oct, n_oct])
    kw     = {} if threshold is None else {'threshold': threshold}
    return split_path_at_seams(pts_n.coords, **kw)


# ---------------------------------------------------------------------------
# Tissot indicatrix generation
# ---------------------------------------------------------------------------

def tissot_circle(lat: float, lon: float,
                  radius_m: float = 500_000.0,
                  n_pts: int = 64) -> np.ndarray:
    """Generate a small geodesic circle as a (N, 2) lat/lon array.

    The circle is approximated by *n_pts* vertices placed at *radius_m* metres
    from (lat, lon) along evenly-spaced azimuths.

    Parameters
    ----------
    lat, lon : float
        Centre in degrees.
    radius_m : float
        Circle radius in metres (default 500 km).
    n_pts : int
        Number of vertices.

    Returns
    -------
    (n_pts, 2) ndarray  [lat°, lon°]
    """
    azimuths = np.linspace(0.0, 360.0, n_pts, endpoint=False)
    pts = wgs84_offset(np.array([[lat, lon]]), azimuths, radius_m)
    return pts  # (n_pts, 2) [lat, lon]


# ---------------------------------------------------------------------------
# Land polygon loading
# ---------------------------------------------------------------------------

def load_shape_polygons(path: str,
                        interior: bool = False) -> list:
    """Load shape polygons from a shapefile and return as lat/lon arrays.

    Reads any Polygon/MultiPolygon shapefile (Natural Earth, OSM land-polygons,
    etc.) and returns a list of (N, 2) arrays in **(lat°, lon°)** order —
    the H9 convention — ready for :func:`project_path`.

    Natural Earth and OSM shapefiles store coordinates as (lon, lat), so this
    function flips the order automatically.

    Parameters
    ----------
    path : str
        Path to the ``.shp`` file.  Suggested sources in ``preparatory/landmasses/``:

        * ``ne_10m_land.shp``   — Natural Earth 10 m (~7 MB), good for paper figures
        * ``ne_50m_land.shp``   — Natural Earth 50 m (~1 MB), quick renders
        * ``land_polygons_1m.shp`` — OSM simplified (~1 m precision)
    interior : bool
        If True, also include interior rings (lakes / holes).  Default False.

    Returns
    -------
    list of (N, 2) ndarray  [lat°, lon°]
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    gdf   = gpd.read_file(path)
    rings = []

    for geom in gdf.geometry:
        if geom is None:
            continue
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if not isinstance(poly, Polygon):
                continue
            # exterior ring — flip (lon, lat) → (lat, lon)
            xy = np.array(poly.exterior.coords)
            rings.append(xy[:, ::-1])
            if interior:
                for ring in poly.interiors:
                    xy = np.array(ring.coords)
                    rings.append(xy[:, ::-1])

    return rings


# ---------------------------------------------------------------------------
# Geographic octant clipping
# ---------------------------------------------------------------------------

# The 8 octahedral octants are exactly the lat/lon rectangles split at the
# meridians {-180, -90, 0, 90, 180} and the equator.  This follows from the
# octahedral projection assigning octant identity from the sign of the ECEF
# coordinates (x=cos·cos(lon), y=cos·sin(lon), z=sin(lat)) — see
# hhg9/projections/ak_octahedral.py — so there is no theta rotation in
# geographic space.
_OCTANT_BOXES = [(lon0, lat0, lon0 + 90.0, lat0 + 90.0)
                 for lat0 in (-90.0, 0.0)
                 for lon0 in (-180.0, -90.0, 0.0, 90.0)]


def clip_polygon_to_octants(ring_latlon: np.ndarray) -> list:
    """Clip a lat/lon polygon ring to the 8 octahedral octants.

    Each returned sub-ring lies entirely within a single octant, so it
    projects through the H9 pipeline without crossing a net seam and is
    already closed along the octant edges that cut it (meridians ±90/0/180
    and the equator).  This is the robust replacement for the
    seam-split + nearest-vertex-snap heuristic when rendering *filled*
    polygons: clip in geographic space *before* projection.

    Closure direction (interior vs exterior) and concavity are handled by
    shapely's intersection — no manual winding logic required.

    Parameters
    ----------
    ring_latlon : (N, 2) ndarray  [lat°, lon°]
        A single polygon ring, as produced by :func:`load_shape_polygons`.

    Returns
    -------
    list of (M, 2) ndarray  [lat°, lon°]
        One sub-ring per octant the polygon touches.  A ring already inside
        a single octant returns a single piece.  Empty input or a degenerate
        ring returns an empty list.
    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box

    if len(ring_latlon) < 3:
        return []

    # ring is (lat, lon); shapely works in (x=lon, y=lat)
    poly = Polygon(ring_latlon[:, ::-1])
    if not poly.is_valid:
        poly = poly.buffer(0)          # repair self-intersections
    if poly.is_empty:
        return []

    pieces = []
    for bounds in _OCTANT_BOXES:
        clipped = poly.intersection(box(*bounds))
        if clipped.is_empty:
            continue
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            geoms = clipped.geoms
        else:
            geoms = [clipped]
        for geom in geoms:
            if not isinstance(geom, Polygon) or geom.is_empty:
                continue
            xy = np.asarray(geom.exterior.coords)
            pieces.append(xy[:, ::-1])  # back to (lat, lon)
    return pieces


# ---------------------------------------------------------------------------
# Tile vertex positions in n_oct
# ---------------------------------------------------------------------------

def is_c2_split(n_oct) -> bool:
    """True if the net layout subdivides each octant into c2 sub-triangles.

    c2-split layouts (e.g. ``rhombus``) introduce seams *inside* an octant's
    geographic extent, so octant-level geographic clipping
    (:func:`clip_polygon_to_octants`) is not a fine enough fundamental to close
    filled polygons there — the c2 sub-triangle is.  Callers rendering fills
    should guard on this until a c2/cell-level clip is available.

    Detected from ``n_oct.face_polys``: simple layouts contribute one triangle
    per octant, c2-split layouts contribute three.
    """
    return any(len(polys) > 1 for polys in n_oct.face_polys.values())


def tile_vertices_noct(n_oct) -> np.ndarray:
    """Return unique tile-corner positions in n_oct coordinates.

    Reads directly from ``n_oct.face_polys``, which is populated at construction
    time and kept in sync with any global theta rotation.

    For simple layouts (butterfly, mortar) each octant contributes one triangle
    → equivalent to the 6 octahedral face corners.  For c2-split layouts
    (rhombus) each split octant contributes three sub-triangles → the c2 joint
    vertices are also included, giving the correct snap targets for rhombus tile
    seams.

    Parameters
    ----------
    n_oct : OctahedralNet
        The display domain whose ``face_polys`` dict to read.

    Returns
    -------
    (V, 2) ndarray  —  n_oct coordinates of all tile corners, deduplicated to
    within 1e-6 absolute tolerance.
    """
    raw  = np.vstack([tri for polys in n_oct.face_polys.values() for tri in polys])
    keep = np.ones(len(raw), dtype=bool)
    for i in range(1, len(raw)):
        if np.any(np.linalg.norm(raw[:i] - raw[i], axis=1) < 1e-6):
            keep[i] = False
    return raw[keep]


# ---------------------------------------------------------------------------
# Graticule generation
# ---------------------------------------------------------------------------

def graticule_paths(lat_step: float = 30.0,
                    lon_step: float = 30.0,
                    lat_range: tuple = (-90.0, 90.0),
                    lon_range: tuple = (-180.0, 180.0),
                    n_pts: int = 361) -> list:
    """Generate a lat/lon graticule as a list of (N, 2) lat/lon arrays.

    Parameters
    ----------
    lat_step, lon_step : float
        Spacing of parallels and meridians in degrees.
    lat_range, lon_range : tuple (min, max)
        Range to cover.
    n_pts : int
        Number of sample points along each line.

    Returns
    -------
    list of (N, 2) ndarray  [lat°, lon°]
        Each array is one parallel or meridian, suitable for passing to
        :func:`project_path`.  Lines are NOT pre-densified geodesically —
        pass each through :func:`project_path` for proper handling.
    """
    paths = []
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    # Parallels
    lats = np.arange(lat_min, lat_max + lat_step * 0.5, lat_step)
    lons = np.linspace(lon_min, lon_max, n_pts)
    for lat in lats:
        paths.append(np.column_stack([np.full(n_pts, lat), lons]))

    # Meridians
    lons_m = np.arange(lon_min, lon_max + lon_step * 0.5, lon_step)
    lats_m = np.linspace(lat_min, lat_max, n_pts)
    for lon in lons_m:
        paths.append(np.column_stack([lats_m, np.full(n_pts, lon)]))

    return paths
