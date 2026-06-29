"""
Geometry engine
───────────────
Parses airport GeoJSON into a taxiway/runway index, pre-computes the
intersections and genuine connections between taxiways, and builds the
directional runway-entry index used to disambiguate landing exits.

All state lives in `state.airport_data`; the functions here read and write it.
"""

from collections import Counter
from typing import Optional

import numpy as np
from shapely.geometry import shape, Point

from state import airport_data

BUFFER_TOLERANCE = 0.00015   # ~15 meters, accounts for OSM gaps
ENDPOINT_THRESHOLD = 0.0003  # ~30 meters, proximity to count as a terminal junction
PEEL_WINDOW = 10             # number of coords to examine from the runway endpoint


def _get_terminal_endpoints(ref: str) -> list:
    """
    Return the true terminal endpoints of a taxiway ref — coords that appear
    exactly once across all its segments (i.e. not internal joints between
    consecutive segments of the same taxiway).
    """
    coord_count = Counter()
    for f in airport_data["taxiway_features"].get(ref, []):
        coords = f["geometry"]["coordinates"]
        coord_count[tuple(coords[0])] += 1
        coord_count[tuple(coords[-1])] += 1
    return [Point(c) for c, cnt in coord_count.items() if cnt == 1]


def _is_runway_phantom(ref_a: str, ref_b: str, ix_point, all_runways) -> bool:
    """
    Return True if the intersection between ref_a and ref_b is a phantom caused
    by both taxiways independently crossing the same runway strip, rather than
    genuinely meeting each other.

    Rule: if the intersection centroid lies on a runway AND is not within
    ENDPOINT_THRESHOLD of a true terminal endpoint of either taxiway, the
    connection is phantom and should be discarded.
    """
    if all_runways is None:
        return False
    if not all_runways.buffer(BUFFER_TOLERANCE).contains(ix_point):
        return False  # intersection is off the runway — always genuine
    # On runway: check whether either taxiway has a terminal endpoint nearby
    terminals = _get_terminal_endpoints(ref_a) + _get_terminal_endpoints(ref_b)
    return not any(ix_point.distance(ep) < ENDPOINT_THRESHOLD for ep in terminals)


def load_geojson(geojson: dict):
    """Parse GeoJSON and build taxiway geometry index."""
    airport_data["geojson"] = geojson
    airport_data["taxiway_geoms"].clear()
    airport_data["taxiway_features"].clear()
    airport_data["runway_geoms"].clear()
    airport_data["valid_refs"].clear()
    airport_data["intersections"].clear()
    airport_data["taxiway_connections"].clear()
    airport_data["runway_entry_taxiways"].clear()

    features = geojson.get("features", [])

    for f in features:
        props = f.get("properties", {})
        aeroway = props.get("aeroway")
        ref = props.get("ref")
        if not ref:
            continue

        try:
            geom = shape(f["geometry"])
        except Exception:
            continue

        if aeroway == "taxiway" or aeroway == "taxilane":
            if ref in airport_data["taxiway_geoms"]:
                airport_data["taxiway_geoms"][ref] = airport_data["taxiway_geoms"][ref].union(geom)
            else:
                airport_data["taxiway_geoms"][ref] = geom

            if ref not in airport_data["taxiway_features"]:
                airport_data["taxiway_features"][ref] = []
            airport_data["taxiway_features"][ref].append(f)

            airport_data["valid_refs"].add(ref)

        elif aeroway == "runway":
            if ref in airport_data["runway_geoms"]:
                airport_data["runway_geoms"][ref] = airport_data["runway_geoms"][ref].union(geom)
            else:
                airport_data["runway_geoms"][ref] = geom

    # Pre-compute intersections between all taxiway pairs
    refs = sorted(airport_data["valid_refs"])
    print(f"Loaded {len(refs)} taxiway refs: {refs}")
    print(f"Loaded {len(airport_data['runway_geoms'])} runway refs: {sorted(airport_data['runway_geoms'].keys())}")

    # Combined runway geometry used for phantom intersection filtering
    all_runways = None
    for rg in airport_data["runway_geoms"].values():
        all_runways = all_runways.union(rg) if all_runways else rg

    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            ga = airport_data["taxiway_geoms"][a].buffer(BUFFER_TOLERANCE)
            gb = airport_data["taxiway_geoms"][b].buffer(BUFFER_TOLERANCE)
            if ga.intersects(gb):
                ix = ga.intersection(gb)
                centroid = ix.centroid
                airport_data["intersections"][(a, b)] = {
                    "point": [centroid.x, centroid.y],
                }
                airport_data["intersections"][(b, a)] = {
                    "point": [centroid.x, centroid.y],
                }

                # Only add to taxiway_connections if intersection is genuine
                # (not a runway phantom - two taxiways crossing the same runway)
                if not _is_runway_phantom(a, b, centroid, all_runways):
                    airport_data["taxiway_connections"].setdefault(a, set()).add(b)
                    airport_data["taxiway_connections"].setdefault(b, set()).add(a)

    print(f"Found {len(airport_data['intersections']) // 2} taxiway intersection pairs")
    print(f"Found {sum(len(v) for v in airport_data['taxiway_connections'].values()) // 2} genuine taxiway connections")

    # Build runway -> touching taxiways index
    airport_data["runway_taxiways"].clear()
    for rwy_ref, rwy_geom in airport_data["runway_geoms"].items():
        touching = set()
        for tw_ref, tw_geom in airport_data["taxiway_geoms"].items():
            if rwy_geom.buffer(BUFFER_TOLERANCE).intersects(tw_geom.buffer(BUFFER_TOLERANCE)):
                touching.add(tw_ref)
        airport_data["runway_taxiways"][rwy_ref] = touching
        print(f"  Runway {rwy_ref} touches taxiways: {sorted(touching)}")

    # Build directional runway entry index
    build_runway_entry_index()


def _taxiway_segment_accessible_from_direction(
    seg_coords: list,
    rwy_geom,
    rwy_vec: np.ndarray,
) -> bool:
    """
    Given a single taxiway segment's coordinates, a runway geometry, and the
    unit vector representing the landing direction, return True if the segment
    peels off in the landing direction (i.e. is accessible from that end).

    Algorithm:
      - Find the entry point where the segment meets the runway.
      - Orient the coord list so it starts from the runway endpoint.
      - Take only the first PEEL_WINDOW coords (avoids hook-shaped taxiways
        where the far end curves back and confuses the direction check).
      - Project each coord onto the runway vector and compute perpendicular
        deviation from the runway line.
      - If the coord furthest in the landing direction has MORE deviation than
        the coord furthest against it, the segment branches off in the landing
        direction → accessible from this runway end.
    """
    ix = rwy_geom.buffer(BUFFER_TOLERANCE).intersection(
        shape({"type": "LineString", "coordinates": seg_coords}).buffer(BUFFER_TOLERANCE)
    )
    entry = np.array([ix.centroid.x, ix.centroid.y])

    # Orient so the runway endpoint is first
    start = np.array(seg_coords[0])
    end   = np.array(seg_coords[-1])
    if np.linalg.norm(end - entry) < np.linalg.norm(start - entry):
        seg_coords = seg_coords[::-1]

    # Only examine the first PEEL_WINDOW coords from the runway endpoint
    window = np.array(seg_coords[:PEEL_WINDOW])

    pairs = []
    for c in window:
        v = c - entry
        proj = float(np.dot(v, rwy_vec))
        perp = float(np.linalg.norm(v - proj * rwy_vec))
        pairs.append((proj, perp))

    pairs.sort()
    dev_at_max_proj = pairs[-1][1]
    dev_at_min_proj = pairs[0][1]
    return dev_at_max_proj > dev_at_min_proj


def build_runway_entry_index():
    """
    For every runway in the GeoJSON, split its ref into the two landing
    directions (e.g. '13L/31R' -> '13L' and '31R') and determine which
    taxiway refs are accessible from each direction using the directional
    peeling algorithm.

    Populates airport_data["runway_entry_taxiways"]:
        {
            "13L": {"D", "ZA", "C1", ...},
            "31R": {"D", "C3", ...},
            "13R": {...},
            ...
        }
    """
    geojson = airport_data["geojson"]
    if not geojson:
        return

    features = geojson.get("features", [])
    airport_data["runway_entry_taxiways"].clear()

    for rwy_combined_ref, rwy_geom in airport_data["runway_geoms"].items():
        # Collect all raw coords for this runway to find its two threshold ends
        rwy_raw_coords = []
        for f in features:
            props = f.get("properties", {})
            if props.get("aeroway") == "runway" and props.get("ref") == rwy_combined_ref:
                rwy_raw_coords.extend(f["geometry"]["coordinates"])

        if not rwy_raw_coords:
            continue

        # NW end = highest latitude, SE end = lowest latitude
        nw_end = np.array(max(rwy_raw_coords, key=lambda c: c[1]))
        se_end = np.array(min(rwy_raw_coords, key=lambda c: c[1]))

        # Two landing direction unit vectors
        vec_nw_to_se = se_end - nw_end
        vec_nw_to_se = vec_nw_to_se / np.linalg.norm(vec_nw_to_se)
        vec_se_to_nw = -vec_nw_to_se

        # Split combined ref into individual runway designators
        # e.g. '13L/31R' -> ['13L', '31R']
        parts = rwy_combined_ref.split("/")
        if len(parts) != 2:
            continue
        # Lower number = NW-to-SE direction (heading ~040-180 range)
        # Higher number = SE-to-NW direction
        # Sort by numeric part: e.g. 13 < 31
        def rwy_num(r):
            return int("".join(filter(str.isdigit, r)))

        parts_sorted = sorted(parts, key=rwy_num)
        lower_rwy = parts_sorted[0]   # e.g. '13L' — NW threshold, lands NW->SE
        upper_rwy = parts_sorted[1]   # e.g. '31R' — SE threshold, lands SE->NW

        entry_sets = {
            lower_rwy: set(),   # NW->SE landing direction
            upper_rwy: set(),   # SE->NW landing direction
        }

        # Check each taxiway that touches this runway
        touching_refs = airport_data["runway_taxiways"].get(rwy_combined_ref, set())
        for tw_ref in touching_refs:
            tw_features = airport_data["taxiway_features"].get(tw_ref, [])

            # Require a true terminal endpoint to lie on the runway —
            # taxiways that merely cross the runway without terminating on it
            # are not valid exits (e.g. C1 crosses 13L but exits toward 31R)
            terminals = _get_terminal_endpoints(tw_ref)
            if not any(rwy_geom.buffer(BUFFER_TOLERANCE).contains(ep) for ep in terminals):
                continue

            for f in tw_features:
                seg_coords = f["geometry"]["coordinates"]
                seg_geom = shape(f["geometry"])
                if not rwy_geom.buffer(BUFFER_TOLERANCE).intersects(seg_geom.buffer(BUFFER_TOLERANCE)):
                    continue  # this segment doesn't touch the runway

                # Test both directions
                if _taxiway_segment_accessible_from_direction(seg_coords, rwy_geom, vec_nw_to_se):
                    entry_sets[lower_rwy].add(tw_ref)
                if _taxiway_segment_accessible_from_direction(seg_coords, rwy_geom, vec_se_to_nw):
                    entry_sets[upper_rwy].add(tw_ref)

        airport_data["runway_entry_taxiways"].update(entry_sets)
        print(f"  {lower_rwy} entry taxiways: {sorted(entry_sets[lower_rwy])}")
        print(f"  {upper_rwy} entry taxiways: {sorted(entry_sets[upper_rwy])}")


def taxiways_intersect(a: str, b: str) -> bool:
    """Check if two taxiways intersect."""
    return (a, b) in airport_data["intersections"]


def get_intersection_point(a: str, b: str) -> Optional[list]:
    """Get the intersection point of two taxiways."""
    ix = airport_data["intersections"].get((a, b))
    return ix["point"] if ix else None


def get_taxiway_segment_near(ref: str, point: list) -> Optional[list]:
    """Get the nearest point on a taxiway to a given point."""
    geom = airport_data["taxiway_geoms"].get(ref)
    if not geom:
        return None
    p = Point(point[0], point[1])
    nearest = geom.interpolate(geom.project(p))
    return [nearest.x, nearest.y]
