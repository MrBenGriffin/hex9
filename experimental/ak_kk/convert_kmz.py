from pathlib import Path
import numpy as np
from xml.etree import ElementTree as ET

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

def _parse_coord_string(s):
    # "lon,lat,alt lon,lat,alt ..."
    pts = []
    for tok in s.strip().split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0]); lat = float(parts[1])
        # alt may be present but we ignore
        pts.append((lat, lon))
    return np.array(pts, dtype=float)


def load_kml_polygons(path):
    """
    Returns:
      dict[name] -> dict with:
        'verts': (M,2) array in (lat,lon) order of the outer ring (closed)
        'centroid': (2,) lat,lon centroid of the ring
    """
    root = ET.parse(Path(path)).getroot()
    out = {}
    for pm in root.findall(".//kml:Placemark", KML_NS):
        name_el = pm.find("kml:name", KML_NS)
        name = name_el.text.strip() if name_el is not None else None

        # Prefer Polygon outerBoundaryIs/LinearRing/coordinates
        ring = pm.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        if ring is None:
            # Some KMLs hide Polygon inside MultiGeometry
            ring = pm.find(".//kml:MultiGeometry//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        if ring is None:
            continue  # skip non-polygon Placemarks

        verts_ll = _parse_coord_string(ring.text)  # (hex_layer,2) lat,lon
        if len(verts_ll) == 0:
            continue

        # Ensure closed ring
        if not np.allclose(verts_ll[0], verts_ll[-1]):
            verts_ll = np.vstack([verts_ll, verts_ll[0]])

        # Simple planar centroid in lat/lon (good enough for tight rings)
        centroid = verts_ll[:-1].mean(axis=0)

        key = name or f"pm_{len(out)}"
        out[key] = {"verts": verts_ll, "centroid": centroid}
    return out


if __name__ == '__main__':
    kk = load_kml_polygons('ak_3.kml')
    np.save('ak_3.npy', kk)
    kk = load_kml_polygons('../../preparatory/kk_3.kml')
    np.save('kk_3.npy', kk)
