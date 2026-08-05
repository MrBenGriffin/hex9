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
load_shape_geoms
    Load polygons *with holes* from a shapefile as shapely Polygons (lon, lat),
    for hole-bearing fills (ocean cuts out continents; keeps endorheic seas).
clip_polygon_to_octants
    Clip a lat/lon polygon ring to the 8 octahedral octants (lat/lon
    rectangles) so each resulting piece lies within a single octant and
    projects without crossing a net seam — the robust replacement for the
    seam-split + snap heuristic when rendering *filled* polygons.
project_fill_to_noct
    Clip → project → union a lat/lon ring into n_oct fill polygons, dissolving
    contiguous octant *joins* while keeping true net *seams* split.  Use this
    (not raw clip+project) to render fills so polygons stay whole across joins.
is_c2_split
    True when a net layout subdivides octants into c2 sub-triangles (rhombus),
    for which octant-level clipping is not a fine enough fundamental to close
    fills.  Render guards should check this.
tile_vertices_noct
    Return the unique tile-corner positions in n_oct space for any net layout.
    For simple layouts these are the 6 octahedral face corners; for c2-split
    layouts (e.g. rhombus) also includes c2 sub-triangle joints.
tile_frame
    Develop one H9 hexagon tile (any level) flat: the 1–2 octant faces it spans
    unfolded (via local_layout) into a 2D frame, plus the hexagon polygon and a
    coarse geographic bbox.  The basis for printable per-tile vector maps.
project_fill_to_tile
    Clip+project a lat/lon polygon (holes preserved) into an L0 tile's developed
    frame and clip to the hexagon — the per-tile analogue of project_fill_to_noct.
l0_tile_subgrid
    The finer H9 hex cells (layers 1,2,3…) within an L0 tile, developed into its
    2D frame — for drawing the progressively-finer subgrid on a printable tile.
"""

import numpy as np
from pathlib import Path
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

def load_shape_polygons(path: str | Path,
                        interior: bool = False) -> list:
    """Load shape polygons from a shapefile and return as lat/lon arrays.

    Reads any Polygon/MultiPolygon shapefile (Natural Earth, OSM land-polygons,
    etc.) and returns a list of (N, 2) arrays in **(lat°, lon°)** order —
    the H9 convention — ready for :func:`project_path`.

    Natural Earth and OSM shapefiles store coordinates as (lon, lat), so this
    function flips the order automatically.

    Parameters
    ----------
    path : str or Path
        Path to the ``.shp`` file.  Prefer :func:`hhg9.geo.naturalearth.ne_layer`,
        which resolves a layer to ``examples/src/landmasses`` and downloads the
        official public-domain archive if it is absent::

            ne_layer('land', '50m')   # 1420 features, quick renders
            ne_layer('land', '10m')   # 11 dissolved landmasses, paper figures
            ne_layer('minor_islands', '10m')   # 2795 islands, use alongside land

        OSM land-polygon extracts (osmdata.openstreetmap.de) also load here, but
        are ODbL — share-alike and attribution-required — so they are not
        bundled or fetched automatically.  See NOTICE.md.
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


def _unwrap_lon(lon: np.ndarray) -> np.ndarray:
    """Make a longitude sequence continuous across the ±180 antimeridian.

    A ring centred on (or crossing) the antimeridian has consecutive vertices
    that jump by ~360° (e.g. +179 → −179).  Left as-is, shapely reads this as a
    polygon spanning the whole globe.  We add multiples of 360 to remove those
    jumps, yielding a continuous (possibly out-of-[−180,180]) longitude run.
    """
    lon = np.asarray(lon, dtype=float)
    d = np.diff(lon)
    adj = np.zeros_like(lon)
    adj[1:] = np.cumsum(np.where(d > 180.0, -360.0,
                                 np.where(d < -180.0, 360.0, 0.0)))
    return lon + adj


def clip_polygon_to_octants(ring_latlon: np.ndarray) -> list:
    """Clip a lat/lon polygon ring to the 8 octahedral octants.

    Each returned sub-ring lies entirely within a single octant, so it
    projects through the H9 pipeline without crossing a net seam and is
    already closed along the octant edges that cut it (meridians ±90/0/180
    and the equator).  This is the robust replacement for the
    seam-split + nearest-vertex-snap heuristic when rendering *filled*
    polygons: clip in geographic space *before* projection.

    Closure direction (interior vs exterior) and concavity are handled by
    shapely's intersection — no manual winding logic required.  Rings that
    cross the ±180 antimeridian (e.g. a Tissot circle centred there) are
    unwrapped first, clipped against the extended octant bands, then shifted
    back into [−180, 180] as whole pieces.

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
    import math
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box

    if len(ring_latlon) < 3:
        return []

    lat = np.asarray(ring_latlon[:, 0], dtype=float)
    ulon = _unwrap_lon(ring_latlon[:, 1])   # continuous across the antimeridian

    # shapely works in (x=lon, y=lat)
    poly = Polygon(np.column_stack([ulon, lat]))
    if not poly.is_valid:
        poly = poly.buffer(0)          # repair self-intersections
    if poly.is_empty:
        return []

    # Octant longitude bands aligned to {…,-180,-90,0,90,180,…}, extended to
    # cover the (possibly unwrapped) longitude span of this ring.
    lo = math.floor((ulon.min() + 180.0) / 90.0) * 90.0 - 180.0
    lon0s = np.arange(lo, ulon.max() + 1e-9, 90.0)

    pieces = []
    for lat0 in (-90.0, 0.0):
        for lon0 in lon0s:
            clipped = poly.intersection(box(lon0, lat0, lon0 + 90.0, lat0 + 90.0))
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
                # shift the whole piece back into [-180, 180] (keeps it
                # contiguous; per-point modulo would tear the seam edge)
                k = np.round(xy[:, 0].mean() / 360.0)
                xy[:, 0] -= 360.0 * k
                pieces.append(xy[:, ::-1])  # back to (lat, lon)
    return pieces


def load_shape_geoms(path: str | Path) -> list:
    """Load polygons (with holes) from a shapefile as shapely Polygons.

    Unlike :func:`load_shape_polygons` (which flattens to exterior/interior
    rings and drops the hole structure), this keeps each polygon intact —
    exterior **and** its interior rings — in native ``(lon, lat)`` order, ready
    for :func:`project_fill_to_noct`.  Use this for hole-bearing fills such as
    the ocean polygon (continents are holes; endorheic seas are separate
    polygons), so continents are cut out rather than painted over.

    Returns
    -------
    list of shapely.geometry.Polygon  (coordinates in (lon°, lat°))
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    gdf = gpd.read_file(path)
    out = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for p in polys:
            if isinstance(p, Polygon) and not p.is_empty:
                out.append(p)
    return out


def _iter_polys(geom):
    """Yield non-empty Polygon parts from any shapely geometry."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        yield geom
    elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for g in geom.geoms:
            yield from _iter_polys(g)


def _octant_sign(lon_c: float, lat_c: float) -> tuple:
    """Octant sign key (sx, sy, sz) matching ``n_oct.face_polys`` keys.

    sx = sign(cos lon), sy = sign(sin lon), sz = sign(lat) — the ECEF-sign
    octant identity.  Evaluated at a piece centroid (interior to its octant).
    """
    lon_r = np.radians(lon_c)
    sx = 1 if np.cos(lon_r) >= 0 else -1
    sy = 1 if np.sin(lon_r) >= 0 else -1
    sz = 1 if lat_c >= 0 else -1
    return (sx, sy, sz)


def _project_ring_to_noct(coords_lonlat, reg, n_oct, max_step_m):
    """Project one ring (lon,lat coords) to n_oct as a single whole ring.

    Seam-splitting is deliberately disabled (``threshold=inf``): a within-octant
    ring belongs to one triangle, but cone-point projection ambiguity can throw
    a spurious far jump that ``split_path_at_seams`` would cut — keeping only a
    fragment.  Instead we keep the whole ring (spurious chord and all) and let
    the caller's triangle confinement clip the out-of-triangle excursion.
    """
    latlon = np.asarray(coords_lonlat)[:, ::-1]
    segs = project_path(latlon, reg, n_oct, max_step_m=max_step_m,
                        closed=True, threshold=np.inf)
    if not segs:
        return None
    return segs[0]


def _project_poly_to_noct(poly, reg, n_oct, max_step_m):
    """Project a shapely Polygon (lon,lat, possibly holed) to an n_oct Polygon."""
    from shapely.geometry import Polygon
    ext = _project_ring_to_noct(poly.exterior.coords, reg, n_oct, max_step_m)
    if ext is None or len(ext) < 3:
        return None
    holes = []
    for ring in poly.interiors:
        h = _project_ring_to_noct(ring.coords, reg, n_oct, max_step_m)
        if h is not None and len(h) >= 3:
            holes.append(h)
    pg = Polygon(ext, holes)
    if not pg.is_valid:
        pg = pg.buffer(0)
    return pg


def project_fill_to_noct(geom,
                         reg,
                         n_oct,
                         max_step_m: float = 50_000.0,
                         threshold: float = None) -> list:
    """Project a lat/lon polygon to n_oct fill polygons, holes preserved.

    Pipeline: **clip → project → confine to octant triangle → union**.

    * Clips ``geom`` to octants (lat/lon rectangles), with antimeridian unwrap.
    * Projects each octant piece's exterior **and interior rings** (so holes —
      e.g. continents inside the ocean polygon — are carried through), then
      **confines each piece to its own octant triangle** (``n_oct.face_polys``).
      The confinement is essential: near an octahedron vertex (a cone point) the
      projection is multivalued, so boundary points otherwise throw spurious
      chords into a neighbouring triangle or the net's angle deficit (impossible
      space).  The projected octant boundary *is* the triangle edge, so no
      legitimate fill is lost.
    * Unions the confined pieces, dissolving octant boundaries that are *joins*
      (so a polygon crossing one stays whole) while keeping genuine *seams*
      split.  A polygon on a cone point yields the best-possible bounded region
      (a full flat circle there is impossible: 4 faces × 60° = 240° ≠ 360°).

    Parameters
    ----------
    geom : (N, 2) ndarray [lat°, lon°]  OR  shapely Polygon/MultiPolygon in
        (lon°, lat°).  Pass a ring for hole-free fills (land, Tissot); pass a
        shapely polygon (e.g. from :func:`load_shape_geoms`) to preserve holes.
    reg : Registrar
    n_oct : OctahedralNet
    max_step_m : float
        Geodesic densification step (metres), passed to :func:`project_path`.
    threshold : float or None
        Accepted for call-site compatibility but unused: per-octant triangle
        confinement replaces seam-splitting (rings are projected whole).

    Returns
    -------
    list of shapely.geometry.Polygon
        n_oct fill polygons (with holes), one per merged component.
    """
    import math
    from shapely.geometry import Polygon, box
    from shapely.affinity import translate
    from shapely.ops import unary_union

    # normalise input to a shapely geometry in (lon, lat)
    if isinstance(geom, np.ndarray):
        if len(geom) < 3:
            return []
        lat = geom[:, 0].astype(float)
        ulon = _unwrap_lon(geom[:, 1])         # continuous across antimeridian
        src = Polygon(np.column_stack([ulon, lat]))
    else:
        src = geom
    if not src.is_valid:
        src = src.buffer(0)
    if src.is_empty:
        return []

    minx, _, maxx, _ = src.bounds
    lo = math.floor((minx + 180.0) / 90.0) * 90.0 - 180.0
    lon0s = np.arange(lo, maxx + 1e-9, 90.0)

    confined = []
    for lat0 in (-90.0, 0.0):
        for lon0 in lon0s:
            clip = src.intersection(box(lon0, lat0, lon0 + 90.0, lat0 + 90.0))
            for part in _iter_polys(clip):
                # shift the whole piece back into [-180, 180]
                k = round(part.centroid.x / 360.0)
                if k:
                    part = translate(part, xoff=-360.0 * k)
                c = part.centroid
                tris = n_oct.face_polys.get(_octant_sign(c.x, c.y))
                if not tris:
                    continue
                region = unary_union([Polygon(t) for t in tris])   # octant cell
                pg_n = _project_poly_to_noct(part, reg, n_oct, max_step_m)
                if pg_n is None:
                    continue
                confined.extend(_iter_polys(pg_n.intersection(region)))
    if not confined:
        return []
    # The two sides of a join trace the same triangle edge with different
    # vertex subdivisions (offsets ~1e-12), and the exact-arithmetic union
    # then keeps micro-sliver components instead of dissolving the shared
    # chord.  Snap to a grid far below max_step_m (1e-9 n_oct units ≈ mm on
    # Earth) so coincident chords are exactly coincident before the union.
    try:
        from shapely import set_precision
        confined = [set_precision(p, 1e-9) for p in confined]
    except ImportError:      # shapely < 2.0: fall back to the exact union
        pass
    return list(_iter_polys(unary_union(confined)))


# ---------------------------------------------------------------------------
# L0 hexagon tiles — per-tile 2-face develop (printable jigsaw / fold-up solid)
# ---------------------------------------------------------------------------

# A tile = an edge-centred L0 hexagon: the two half-hexes meeting across one of
# the octahedron's 12 edges, so it spans exactly 2 octant faces.  We lay the two
# faces flat with the canonical, **net-independent** ``OctahedralNet.local_layout``
# (the fundamental octant in its native b_oct frame, the other hinged onto it by
# a det +1 rotation that closes the seam exactly — so never mirrored, hexes stay
# regular).  Everything — L0 hexagon, fill data, H9 subgrid — is developed in
# b_oct and placed through that one layout, continuous across the shared edge.

_TILE_NET = 'butterfly'        # only a vehicle for local_layout/place; non-c2.
                               # The layout itself does not depend on the net.


def _octant_box(oid):
    """The lat/lon rectangle of octant *oid* (its ECEF-sign quadrant)."""
    from shapely.geometry import box
    from hhg9.h9 import H9O
    sx, sy, sz = (int(v) for v in H9O.oid_cmp[oid])
    lat = (0.0, 90.0) if sz > 0 else (-90.0, 0.0)
    lon = {(1, 1): (0.0, 90.0), (-1, 1): (90.0, 180.0),
           (-1, -1): (-180.0, -90.0), (1, -1): (-90.0, 0.0)}[(sx, sy)]
    return box(lon[0], lat[0], lon[1], lat[1])


def tile_address(addr, level: int) -> str:
    """Hex-digit address tag of a level-*L* H9 cell: the top ``level+1`` nibbles
    of its bin UUID — e.g. ``'4'`` (L0) or ``'42'`` (L1 = L0 hex 4, child 2).
    Ties tile ids to the H9 binning so a tile is named by where it sits."""
    nib = level + 1
    return format(addr.int >> (128 - 4 * nib), f'0{nib}x')


def tile_frame(reg, tile, level: int = 0):
    """Develop H9 hexagon tile *tile* at *level* into a flat 2D frame.

    *tile* is either the mesh index (int) or the hex **address** (str, e.g.
    ``'42'``).  A tile spans 1 or 2 octant faces (no H9 hexagon crosses more than
    one octahedron edge at any depth, so it never wraps a cone vertex).  Returns
    a dict with:

    * ``octants`` — the 1–2 octant faces the tile spans (first = fundamental);
    * ``address`` — the cell's hex address tag (see :func:`tile_address`);
    * ``n_oct``   — the reference net (only a vehicle for ``place``/``local_layout``);
    * ``layout``  — the net-independent ``local_layout`` ``{oid: (matrix, offset)}``;
    * ``hexagon`` — the tile hexagon as a shapely Polygon in that frame;
    * ``geo_bbox``— a generous (lon,lat,lon,lat) box of the cell's geographic
      extent, or ``None`` if it wraps the ±180 seam; pre-clips fill data for speed.
    """
    from shapely.geometry import Polygon
    from hhg9 import Points
    from hhg9.h9.grid import HexMesh

    n_oct = reg.domain(f'n_oct:{_TILE_NET}')
    b_oct, g_gcd = reg.domain('b_oct'), reg.domain('g_gcd')
    mesh = HexMesh.create([level], reg)
    if isinstance(tile, str):
        tile = next((i for i, a in enumerate(mesh.addrs)
                     if tile_address(a, level) == tile),
                    None)
        if tile is None:
            raise ValueError(f'no L{level} tile with address {tile!r}')
    idx = mesh[level][tile]
    hb = np.asarray(mesh.pts.coords)[idx]
    ho = np.asarray(mesh.pts.oid)[idx].astype(int)
    octants = tuple(sorted(set(int(o) for o in ho)))         # 1 or 2

    layout = n_oct.local_layout(octants, fundamental=octants[0]).layout
    h2 = n_oct.place(hb, ho, layout)
    gll = reg.project(Points(hb, b_oct, oid=ho), [b_oct, g_gcd]).coords
    lat, lon = gll[:, 0], gll[:, 1]
    if lon.max() - lon.min() > 180.0:                        # wraps the antimeridian
        geo_bbox = None
    else:                                                    # generous margin (curved-edge bulge + projection slack)
        m = max(3.0, 0.3 * max(lat.max() - lat.min(), lon.max() - lon.min()))
        geo_bbox = (lon.min() - m, lat.min() - m, lon.max() + m, lat.max() + m)
    # North-up rotation: at the cell centre, the angle (deg, CCW) that lifts a
    # short step of geographic North to vertical in this 2D frame.
    cll = gll.mean(0)
    nll = np.array([cll, [min(cll[0] + 0.5, 90.0), cll[1]]])
    npb = reg.project(Points(nll, g_gcd), [g_gcd, b_oct])
    nxy = n_oct.place(*_switch_seam_oids(npb.coords.copy(), np.asarray(npb.oid).copy(),
                                         octants), layout)
    v = nxy[1] - nxy[0]
    north_deg = 0.0 if np.isnan(v).any() else 90.0 - np.degrees(np.arctan2(v[1], v[0]))
    return {'octants': octants, 'address': tile_address(mesh.addrs[tile], level),
            'level': level, 'n_oct': n_oct, 'layout': layout, 'north_deg': north_deg,
            'hexagon': Polygon(h2).buffer(0), 'geo_bbox': geo_bbox}


def _edge_adjacent(oid, octants):
    """The tile octant edge-adjacent to *oid* (ECEF sign differs in exactly one
    axis — equator, or either meridian), or ``None``."""
    from hhg9.h9 import H9O
    s = H9O.oid_cmp[oid]
    for a in octants:
        if int(np.sum(H9O.oid_cmp[a] != s)) == 1:
            return a
    return None


def _switch_seam_oids(coords, oids, octants):
    """Re-home fill points that the projection binned onto a *neighbour* octant.

    A point exactly on an octant seam binds to the mode-0 octant of the pair
    (`reg.project` ignores any supplied oid and re-bins by ECEF sign).  When that
    octant is outside the tile, the point is switchable to the edge-adjacent tile
    octant by inverting its b_oct *y* and swapping the oid — the same flip works
    for every seam type (equator and both meridians, verified).  Otherwise it
    places as NaN, dropping the edge and shaving a sea 'canal' (e.g. the equator
    edge of cell 67, the lon-90 edge of cell 77).  Mutates in place."""
    out = ~np.isin(oids, np.asarray(octants))
    for c in (np.unique(oids[out]) if out.any() else ()):
        a = _edge_adjacent(int(c), octants)
        if a is not None:
            m = out & (oids == c)
            coords[m, 1] *= -1.0                      # invert y: switch to local octant
            oids[m] = a
    return coords, oids


def _place_ring_in_tile(coords_lonlat, reg, n_oct, layout, octants, max_step_m):
    """Densify a (lon,lat) ring, project g_gcd→b_oct, and place via *layout*.

    Projected and placed as one ring (not per octant box), so a ring crossing
    the tile's shared edge stays continuous.  Seam points binned onto a neighbour
    octant are switched back to the local tile octant (see
    :func:`_switch_seam_oids`) so boundary edges are not dropped."""
    from hhg9 import Points
    latlon = np.asarray(coords_lonlat)[:, ::-1]
    dense = densify_poly_geodesic_by_step(latlon, max_step_m=max_step_m, closed=True)
    if len(dense) < 3:
        return None
    pb = reg.project(Points(dense, 'g_gcd'), ['g_gcd', 'b_oct'])
    coords, oids = _switch_seam_oids(pb.coords.copy(), np.asarray(pb.oid).copy(), octants)
    xy = n_oct.place(coords, oids, layout)
    xy = xy[~np.isnan(xy).any(axis=1)]
    return xy if len(xy) >= 3 else None


def _place_poly_in_tile(part, reg, n_oct, layout, octants, max_step_m):
    from shapely.geometry import Polygon
    ext = _place_ring_in_tile(part.exterior.coords, reg, n_oct, layout, octants, max_step_m)
    if ext is None or len(ext) < 3:
        return None
    holes = []
    for ring in part.interiors:
        h = _place_ring_in_tile(ring.coords, reg, n_oct, layout, octants, max_step_m)
        if h is not None and len(h) >= 3:
            holes.append(h)
    pg = Polygon(ext, holes)
    return pg.buffer(0) if not pg.is_valid else pg


def project_fill_to_tile(geom, reg, frame, max_step_m: float = 50_000.0) -> list:
    """Project a lat/lon polygon into an L0 tile's developed frame.

    Per-tile analogue of :func:`project_fill_to_noct`: clips ``geom`` to the
    tile's two octants (a single contiguous lat/lon region), projects
    ``g_gcd → b_oct`` and places through the tile ``layout`` (exterior **and
    holes**), unions, and clips to the hexagon.  Projecting each part whole keeps
    it continuous across the shared octant edge; holes (e.g. continents in the
    ocean polygon) are preserved.

    Parameters
    ----------
    geom : (N, 2) ndarray [lat°, lon°]  OR  shapely Polygon/MultiPolygon (lon, lat)
    reg : Registrar
    frame : dict from :func:`l0_tile_frame`
    max_step_m : float
        Geodesic densification step (metres).

    Returns
    -------
    list of shapely.geometry.Polygon  (in the tile's 2D frame, clipped to the hex)
    """
    from shapely.ops import unary_union
    from shapely.geometry import Polygon

    if isinstance(geom, np.ndarray):
        if len(geom) < 3:
            return []
        src = Polygon(geom[:, ::-1])               # (lat,lon) -> (lon,lat)
    else:
        src = geom
    if not src.is_valid:
        src = src.buffer(0)
    if src.is_empty:
        return []

    from shapely.geometry import box
    n_oct, layout, hexagon = frame['n_oct'], frame['layout'], frame['hexagon']
    octants = frame['octants']
    region = unary_union([_octant_box(o) for o in octants])
    if frame.get('geo_bbox') is not None:                    # cheap pre-clip (finer levels)
        region = region.intersection(box(*frame['geo_bbox']))
    polys = []
    for part in _iter_polys(src.intersection(region)):
        pg = _place_poly_in_tile(part, reg, n_oct, layout, octants, max_step_m)
        if pg is not None and not pg.is_empty:
            polys.append(pg)
    if not polys:
        return []
    return list(_iter_polys(unary_union(polys).intersection(hexagon)))


def tile_subgrid(reg, frame, layers=(1, 2, 3)) -> dict:
    """H9 sub-hex cells developed into a tile's 2D frame, per layer.

    Returns ``{layer: [shapely Polygon, ...]}`` — the finer H9 hex cells placed
    through the tile ``layout`` (same as the data/hexagon), so they are precise
    and properly nested.  A cell is kept if all its vertices lie in the tile's
    octants and it intersects the hexagon; clip to ``frame['hexagon']`` (e.g. an
    SVG clip-path) at draw time so edge-straddling cells render as half-hexagons.

    *layers* are absolute H9 levels (for a level-*L* tile pass ``L+1, L+2, …``).
    """
    from shapely.geometry import Polygon
    from hhg9.h9.grid import HexMesh

    octants = frame['octants']
    n_oct, layout, hexagon = frame['n_oct'], frame['layout'], frame['hexagon']
    mesh = HexMesh.create(list(layers), reg)
    vb = np.asarray(mesh.pts.coords)
    vo = np.asarray(mesh.pts.oid).astype(int)
    v2 = n_oct.place(vb, vo, layout)
    in_tile = np.isin(vo, octants)

    out = {}
    for L in layers:
        cells = mesh[L]                              # (Ncells, 6) vertex indices
        keep = np.where(in_tile[cells].all(axis=1))[0]   # only in-tile cells
        polys = []
        for ci in keep:
            pg = Polygon(v2[cells[ci]])
            if not pg.is_valid:
                pg = pg.buffer(0)
            if not pg.is_empty and pg.intersects(hexagon):
                polys.append(pg)
        out[L] = polys
    return out


def tile_labels(reg, frame, sub=False) -> list:
    """Address labels for a tile, as ``(address, (x, y), depth)`` in its 2D frame.

    Always includes the tile itself at its hexagon centre (``depth`` 0).  If
    *sub*, also each level-``L+1`` child cell whose centre falls inside the tile
    (``depth`` 1) — for annotating the next subdivision.  Centres place through
    the same seam-switch as the fill, so child cells on a seam are not lost."""
    from shapely.geometry import Point
    from hhg9.h9.grid import HexMesh

    hexg = frame['hexagon']
    out = [(frame['address'], np.asarray(hexg.centroid.coords)[0], 0)]
    if not sub:
        return out
    level, octants = frame['level'], frame['octants']
    n_oct, layout, ta = frame['n_oct'], frame['layout'], frame['address']
    mesh = HexMesh.create([level + 1], reg)
    vb = np.asarray(mesh.pts.coords)
    vo = np.asarray(mesh.pts.oid).astype(int)
    for ci, cell in enumerate(mesh[level + 1]):
        a = tile_address(mesh.addrs[ci], level + 1)
        if not a.startswith(ta):                            # a child of this tile
            continue
        xy = n_oct.place(*_switch_seam_oids(vb[cell].copy(), vo[cell].copy(), octants), layout)
        if np.isnan(xy).any():
            continue
        cen = xy.mean(0)
        if hexg.contains(Point(cen)):
            out.append((a, cen, 1))
    return out


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
