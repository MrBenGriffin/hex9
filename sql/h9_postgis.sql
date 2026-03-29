-- H9 PostGIS Functions
-- Part of the Hex9 (H9) Project
-- Copyright ©2025, Ben Griffin
-- Licensed under the Apache License, Version 2.0
--
-- Requires: plpython3u (untrusted Python, needs superuser to install)
-- The hhg9 package must be importable from the PostgreSQL server's Python environment.
--
-- Install:
--   CREATE EXTENSION IF NOT EXISTS plpython3u;
--   \i h9_postgis.sql
--
-- UUID layout (128 bits = 32 nibbles):
--   nibbles  0..30 : hex body L0..L30
--   nibble      31 : key tail = (p_c2 << 1) | r_mo
--
-- Adr byte (smallint, 5 bits used):
--   bit 4    : p_mo
--   bits 3..0: h  (terminal region 0..11)
--
-- The UUID alone supports spatial indexing and hexbin at any layer.
-- The (UUID, adr) pair enables exact round-trip back to lat/lon.

-- 1. NUKE: Clear the board of all previous h9_ functions (dev. only!)
DO $$
DECLARE func_rec record;
BEGIN
    FOR func_rec IN SELECT oid::regprocedure AS func_sig FROM pg_proc WHERE proname LIKE 'h9_%' AND pronamespace = 'public'::regnamespace
    LOOP EXECUTE 'DROP FUNCTION IF EXISTS ' || func_rec.func_sig || ' CASCADE;'; END LOOP;
END; $$;

-- -------------------------------------------------------------------------
-- h9_encode(lon, lat) -> uuid
-- Key-only encode. Sufficient for indexing and binning.
-- NOTE: PostGIS convention is (lon, lat); h9 uses (lat, lon) internally.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_encode(
    lon double precision,
    lat double precision
) RETURNS uuid
LANGUAGE plpython3u
AS $$
    from hhg9.h9.uuid_address import h9_encode as _enc
    uuids, _ = _enc([lat], [lon])
    return str(uuids[0])
$$;


-- -------------------------------------------------------------------------
-- h9_encode(lon, lat) -> (hex9 uuid, adr smallint)
-- Full encode with companion adr byte for round-trip decode.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_encode_adr(
    lon double precision,
    lat double precision,
    OUT hex9 uuid,
    OUT adr  smallint
) RETURNS record
LANGUAGE plpython3u
AS $$
    from hhg9.h9.uuid_address import h9_encode as _enc
    uuids, adrs = _enc([lat], [lon])
    return (str(uuids[0]), int(adrs[0]))
$$;


-- -------------------------------------------------------------------------
-- h9_encode(lons, lats) -> TABLE(hex9 uuid, adr smallint)
-- Bulk encode. Significantly faster than row-by-row for large inputs.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_encode(
    lons double precision[],
    lats double precision[]
) RETURNS TABLE(hex9 uuid, adr smallint)
LANGUAGE plpython3u
AS $$
    from hhg9.h9.uuid_address import h9_encode as _enc
    uuids, adrs = _enc(lats, lons)
    return [{"hex9": str(u), "adr": int(a)} for u, a in zip(uuids, adrs)]
$$;


-- -------------------------------------------------------------------------
-- h9_decode(hex9, adr) -> (lon double precision, lat double precision)
-- Exact round-trip decode. Requires the companion adr byte.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_decode(
    hex9 uuid,
    adr  smallint,
    OUT lon double precision,
    OUT lat double precision
) RETURNS record
LANGUAGE plpython3u
AS $$
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import h9_decode as _dec
    lats, lons = _dec([uuid_mod.UUID(hex9)], [adr])
    return (float(lons[0]), float(lats[0]))
$$;


-- -------------------------------------------------------------------------
-- h9_decode_batch(hex9[], adr[]) -> TABLE(lon, lat)
-- Bulk decode.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_decode(
    hex9s uuid[],
    adrs  smallint[]
) RETURNS TABLE(lon double precision, lat double precision)
LANGUAGE plpython3u
AS $$
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import h9_decode as _dec
    u_list = [uuid_mod.UUID(u) for u in hex9s]
    lats, lons = _dec(u_list, adrs)
    return list(zip(lons.tolist(), lats.tolist()))
$$;


-- -------------------------------------------------------------------------
-- h9_bin(hex9, layer) -> uuid
-- Return the bin-key UUID at the given layer (0..30).
-- The adr byte is NOT required.
-- Bin UUIDs at the same layer compare equal iff the points share a bin.
--
-- IMMUTABLE: deterministic, no side effects — safe for GENERATED STORED
-- columns and functional indexes:
--
--   ALTER TABLE my_points
--       ADD COLUMN bin8  uuid GENERATED ALWAYS AS (h9_bin(hex9, 8))  STORED,
--       ADD COLUMN bin12 uuid GENERATED ALWAYS AS (h9_bin(hex9, 12)) STORED;
--   CREATE INDEX ON my_points (bin8);
--   CREATE INDEX ON my_points (bin12);
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_bin(
    hex9  uuid,
    layer integer
) RETURNS uuid
LANGUAGE plpython3u
IMMUTABLE STRICT
AS $$
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import h9_bin as _bin
    result = _bin([uuid_mod.UUID(hex9)], layer)
    return str(result[0])
$$;


-- -------------------------------------------------------------------------
-- h9_bin(hex9[], layer) -> uuid[]
-- Bulk hexbin at a given layer.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_bin(
    hex9s uuid[],
    layer integer
) RETURNS uuid[]
LANGUAGE plpython3u
IMMUTABLE STRICT
AS $$
    import uuid as uuid_mod
    from hhg9.h9.uuid_address import h9_bin as _bin
    u_list = [uuid_mod.UUID(u) for u in hex9s]
    result = _bin(u_list, layer)
    return [str(u) for u in result]
$$;


-- -------------------------------------------------------------------------
-- h9_in_bin(hex9, bin, layer) -> boolean
-- Test whether a hex9 UUID belongs to the given bin at the given layer.
--
-- Pure SQL — no Python at query time when bin columns are precomputed.
-- With GENERATED STORED columns this reduces to a UUID equality check.
--
--   SELECT h9_in_bin(hex9, '43521606-0155-0000-0000-000000000000', 8)
--   FROM my_points;
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_in_bin(
    hex9  uuid,
    bin   uuid,
    layer integer
) RETURNS boolean
LANGUAGE sql
IMMUTABLE STRICT
AS $$
    SELECT h9_bin(hex9, layer) = bin;
$$;


-- -------------------------------------------------------------------------
-- Example usage
-- -------------------------------------------------------------------------
--
-- Encode a single point (Stonehenge):
--   SELECT * FROM h9_encode(-1.8262, 51.1789);
--   -- returns: (hex9 uuid, adr smallint)
--
-- Bulk encode a table of points:
--   SELECT h9.hex9, h9.adr
--   FROM (SELECT array_agg(lons) AS lon, array_agg(lats) AS lat FROM my_table) b,
--        h9_encode(b.lon, b.lat) AS h9;
--
-- Round-trip test:
--   WITH enc AS (SELECT * FROM h9_encode(-1.8262, 51.1789))
--   SELECT * FROM h9_decode(enc.hex9, enc.adr) FROM enc;
--   -- should return lon ≈ -1.8262, lat ≈ 51.1789
--
-- Hexbin a table of points at layer 15:
--   SELECT h9_bin(hex9, 15) AS bin, COUNT(*) AS n
--   FROM my_points_table
--   GROUP BY bin
--   ORDER BY n DESC;
--
-- Spatial aggregation pipeline:
--   WITH encoded AS (
--       SELECT id, (h9_encode(ST_X(geom), ST_Y(geom))).hex9
--       FROM my_geometry_table
--   )
--   SELECT h9_bin(hex9, 12) AS bin, COUNT(*) AS n
--   FROM encoded
--   GROUP BY bin;
--
-- GENERATED STORED bin columns (zero query-time Python, fully index-scannable):
--   ALTER TABLE my_points
--       ADD COLUMN bin8  uuid GENERATED ALWAYS AS (h9_bin(hex9, 8))  STORED,
--       ADD COLUMN bin12 uuid GENERATED ALWAYS AS (h9_bin(hex9, 12)) STORED;
--   CREATE INDEX ON my_points (bin8);
--   CREATE INDEX ON my_points (bin12);
--
--   -- All points in a bin (pure UUID equality, no Python):
--   SELECT * FROM my_points WHERE bin8 = '43521606-0155-0000-0000-000000000000';
--
--   -- Aggregate at layer 8 (pure UUID equality, no Python):
--   SELECT bin8, COUNT(*) FROM my_points GROUP BY bin8;
--
-- Bin containment test:
--   SELECT h9_in_bin(
--       '43521606-0150-0483-8250-003401502085',
--       h9_bin('43521606-0150-0483-8250-003401502085', 8),
--       8
--   );  -- TRUE
--
--   SELECT h9_in_bin(
--       'ffffffff-ffff-ffff-ffff-ffffffffffff',
--       h9_bin('43521606-0150-0483-8250-003401502085', 8),
--       8
--   );  -- FALSE


-- =========================================================================
-- Polygon fill: generate hexagons covering a geometry (for QGIS display)
-- =========================================================================

-- -------------------------------------------------------------------------
-- h9_bbox_hexagons(min_lon, min_lat, max_lon, max_lat, layer)
--   -> TABLE(bin uuid, hex_geom geometry)
--
-- Low-level: fills a bounding box with H9 hexagons at the given layer.
-- All hexagons whose centre falls within the bbox are returned.
-- Prefer h9_fill_polygon for exact polygon clipping.
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_bbox_hexagons(
    min_lon double precision,
    min_lat double precision,
    max_lon double precision,
    max_lat double precision,
    layer   integer
) RETURNS TABLE(hex9 uuid, bin uuid, hex_geom geometry)
LANGUAGE plpython3u
AS $$
    from hhg9.h9.grid import h9_postgis_hexagons
    rows = h9_postgis_hexagons(min_lon, min_lat, max_lon, max_lat, layer)
    return [
        (hex9_str, bin_str, f'SRID=4326;{wkt}')
        for hex9_str, bin_str, wkt in rows
    ]
$$;

-- -------------------------------------------------------------------------
-- h9_fill_polygon(geom geometry, layer integer)
--   -> TABLE(bin uuid, hex_geom geometry)
--
-- Fills an arbitrary polygon with H9 hexagons at the given layer.
-- Returns only hexagons that intersect the input polygon.
-- Intended for QGIS display: load the result as a geometry layer.
--
-- Example (in DB Manager or QGIS virtual layer):
--   SELECT bin, hex_geom
--   FROM h9_fill_polygon(
--       ST_GeomFromText('POLYGON((-1.87 51.16, -1.78 51.16,
--                                -1.78 51.20, -1.87 51.20,
--                                -1.87 51.16))', 4326),
--       12
--   );
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION h9_fill_polygon(
    geom  geometry,
    layer integer
) RETURNS TABLE(hex9 uuid, bin uuid, hex_geom geometry)
LANGUAGE plpgsql
AS $$
DECLARE
    env geometry;
BEGIN
    env := ST_Envelope(geom);
    RETURN QUERY
        SELECT b.hex9, b.bin, b.hex_geom
        FROM h9_bbox_hexagons(
            ST_XMin(env), ST_YMin(env),
            ST_XMax(env), ST_YMax(env),
            layer
        ) AS b
        WHERE ST_Intersects(b.hex_geom, geom);
END;
$$;

CREATE OR REPLACE FUNCTION h9_populate(area_id integer, level integer)
RETURNS integer
LANGUAGE plpgsql AS $$
DECLARE n integer;
BEGIN
    WITH fill AS (
        SELECT hex9, bin, hex_geom
        FROM h9_areas a, h9_fill_polygon(a.geom, level)
        WHERE a.id = area_id
    ),
    ins_pt AS (
        INSERT INTO h9_pt (id)
        SELECT hex9 FROM fill
        ON CONFLICT DO NOTHING
    ),
    ins_hex AS (
        INSERT INTO h9_hex (pt, level, bin, poly)
        SELECT hex9, level, bin, hex_geom FROM fill
        ON CONFLICT DO NOTHING
    )
    INSERT INTO h9_poly_pts (area_id, level, pt)
    SELECT area_id, level, hex9 FROM fill
    ON CONFLICT DO NOTHING;

    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n;
END;
$$;
