"""
Colored segments engine
────────────────────────
Turns a resolved route (plus runway context and aircraft history) into the
colored line segments the frontend draws on the SVG map. Handles partial
taxiway coloring, runway entry/exit splitting, turn-direction endpoints, and
the yellow/purple history-and-bridge rendering.

Color legend:
  red    — runway (landing threshold → exit point)
  orange — current taxi route (ATC-stated)
  purple — BFS-inferred bridge taxiway
  yellow — previously cleared / pending route
  grey   — untraversed portions restored to base appearance
"""

import math as _math
from collections import Counter, deque
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from state import airport_data
from geometry import BUFFER_TOLERANCE, _get_terminal_endpoints
from routing import runway_taxiways_for

HEADING_LOOKBACK = 3  # number of coords before intersection to estimate arrival heading


def _geom_to_linestrings(geom) -> list:
    """Flatten a geometry (LineString or MultiLineString) to a list of LineString objects."""
    if geom is None:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    return []


def _split_line_at_point(line: LineString, split_pt: Point) -> tuple[list, list]:
    """
    Split a LineString at the point on it closest to split_pt.
    Returns (before_coords, after_coords) as lists of [lng, lat] pairs.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return coords, []

    # Find the index of the vertex closest to the split point
    best_i, best_d = 0, float("inf")
    for i, (x, y) in enumerate(coords):
        d = (x - split_pt.x) ** 2 + (y - split_pt.y) ** 2
        if d < best_d:
            best_i, best_d = i, d

    # Insert the exact split point between best_i and best_i+1
    sp = [split_pt.x, split_pt.y]
    before = [[x, y] for x, y in coords[:best_i + 1]] + [sp]
    after  = [sp] + [[x, y] for x, y in coords[best_i + 1:]]
    return before, after


def _runway_true_endpoints(rwy_geom) -> tuple:
    """
    For a MultiLineString (or LineString) runway, find the two true physical
    endpoints — i.e. the coordinate points that are NOT shared between segments
    (each appears exactly once as a segment endpoint).

    Returns (pt_a, pt_b) as (lng, lat) tuples, in arbitrary order.
    Falls back to the first/last coord of the first sub-line if needed.
    """
    segs = _geom_to_linestrings(rwy_geom)
    endpoint_count: Counter = Counter()
    for seg in segs:
        coords = list(seg.coords)
        endpoint_count[coords[0]] += 1
        endpoint_count[coords[-1]] += 1
    true_ends = [pt for pt, cnt in endpoint_count.items() if cnt == 1]
    if len(true_ends) == 2:
        return true_ends[0], true_ends[1]
    # Fallback: just use first and last coord of first segment
    coords = list(segs[0].coords)
    return coords[0], coords[-1]


def _runway_entry_point(rwy_geom, designator: str) -> Point:
    """
    Return the Shapely Point of the runway threshold where the plane ENTERS
    when landing with this designator (e.g. '13L' enters from the 31R end).

    Runway 13L → landing heading 130° → plane approaches from 310°.
    We pick whichever true endpoint is in the 310° direction from the runway centre.
    """
    pt_a, pt_b = _runway_true_endpoints(rwy_geom)

    # Parse designator number
    num_str = "".join(c for c in designator if c.isdigit())
    if not num_str:
        return Point(pt_a)
    landing_heading = int(num_str) * 10          # e.g. 13 → 130°
    approach_heading = (landing_heading + 180) % 360  # plane comes FROM this direction

    # Runway centre
    cx = (pt_a[0] + pt_b[0]) / 2
    cy = (pt_a[1] + pt_b[1]) / 2

    def bearing_to(pt):
        dx = pt[0] - cx
        dy = pt[1] - cy
        return _math.degrees(_math.atan2(dx, dy)) % 360

    def angle_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    # The entry end is the one whose bearing FROM centre ≈ approach_heading
    bear_a = bearing_to(pt_a)
    bear_b = bearing_to(pt_b)
    if angle_diff(bear_a, approach_heading) < angle_diff(bear_b, approach_heading):
        return Point(pt_a)
    else:
        return Point(pt_b)


def _intersection_point(geom_a, geom_b) -> Optional[Point]:
    """Return the centroid of the buffered intersection between two geometries, or None."""
    ix = geom_a.buffer(BUFFER_TOLERANCE).intersection(geom_b.buffer(BUFFER_TOLERANCE))
    if ix.is_empty:
        return None
    c = ix.centroid
    return None if c.is_empty else c


def _arrival_heading_vec(prev_ref: str, intersection_pt: Point) -> Optional[np.ndarray]:
    """
    Estimate the aircraft's arrival heading vector as it reaches intersection_pt
    on prev_ref.

    Strategy:
      1. Find the single GeoJSON feature of prev_ref whose geometry is closest
         to intersection_pt (for compound taxiways like DB with multiple features,
         this picks the right segment rather than flattening all features together).
      2. Orient that feature's coord list so the end closest to intersection_pt
         is LAST — meaning the aircraft was travelling from coords[0] toward
         coords[-1] (i.e. toward the intersection).
      3. Take up to HEADING_LOOKBACK nodes before the intersection end and
         compute the heading vector from that window.

    Returns a unit vector [dx, dy] pointing in the direction of travel INTO the
    intersection, or None if the geometry is insufficient.
    """
    feats = airport_data["taxiway_features"].get(prev_ref, [])
    if not feats:
        return None

    # Step 1: find the feature whose geometry is closest to intersection_pt
    best_feat_coords = None
    best_feat_dist = float("inf")
    for feat in feats:
        geom_json = feat.get("geometry", {})
        if geom_json.get("type") != "LineString":
            continue
        coords = geom_json["coordinates"]
        if len(coords) < 2:
            continue
        line = LineString([(c[0], c[1]) for c in coords])
        d = line.distance(intersection_pt)
        if d < best_feat_dist:
            best_feat_dist = d
            best_feat_coords = coords

    if best_feat_coords is None or len(best_feat_coords) < 2:
        return None

    # Step 2: orient so the end closest to intersection_pt is last
    start = best_feat_coords[0]
    end   = best_feat_coords[-1]
    d_start = (start[0] - intersection_pt.x)**2 + (start[1] - intersection_pt.y)**2
    d_end   = (end[0]   - intersection_pt.x)**2 + (end[1]   - intersection_pt.y)**2
    if d_start < d_end:
        # intersection is closer to start — reverse so it becomes the last coord
        best_feat_coords = best_feat_coords[::-1]

    # Step 3: take HEADING_LOOKBACK nodes before the intersection end
    window = best_feat_coords[max(0, len(best_feat_coords) - HEADING_LOOKBACK - 1):]
    if len(window) < 2:
        return None

    p_start = np.array(window[0][:2])
    p_end   = np.array(window[-1][:2])
    vec = p_end - p_start
    length = np.linalg.norm(vec)
    if length < 1e-12:
        return None
    return vec / length


def _turn_direction_endpoint(
    last_ref: str,
    entry_pt: Point,
    arrival_vec: np.ndarray,
    turn_direction: str,
) -> Optional[Point]:
    """
    Given an aircraft arriving at entry_pt on last_ref with arrival_vec heading,
    and a turn direction ('left' or 'right'), return the endpoint of last_ref
    that the aircraft will head toward after the turn.

    Strategy:
      - Collect the two physical endpoints of last_ref.
      - For each endpoint, compute the vector from entry_pt to that endpoint.
      - Use the cross product of arrival_vec × candidate_vec to determine
        which side (left = positive Z, right = negative Z in 2D cross product).
      - Return the endpoint whose side matches the requested turn_direction.
    """
    feats = airport_data["taxiway_features"].get(last_ref, [])
    if not feats:
        return None

    # Collect all candidate endpoints of the last taxiway
    # (true terminal endpoints — nodes that appear only once)
    endpoints = _get_terminal_endpoints(last_ref)
    if not endpoints:
        # Fallback: use the raw start/end of each feature
        endpoints = []
        for feat in feats:
            geom_json = feat.get("geometry", {})
            if geom_json.get("type") == "LineString":
                coords = geom_json["coordinates"]
                if coords:
                    endpoints.append(Point(coords[0][:2]))
                    endpoints.append(Point(coords[-1][:2]))

    if not endpoints:
        return None

    # Filter out endpoints very close to the entry point (< BUFFER_TOLERANCE * 2)
    # — these are the "start" of the taxiway we're already on, not destinations
    candidates = [ep for ep in endpoints
                  if ep.distance(entry_pt) > BUFFER_TOLERANCE * 2]
    if not candidates:
        candidates = endpoints  # fallback: use all if filtering removed everything

    # For each candidate, compute the 2D cross product of arrival_vec × direction
    # cross > 0 → candidate is to the LEFT of travel direction
    # cross < 0 → candidate is to the RIGHT of travel direction
    scored = []
    for ep in candidates:
        vec_to_ep = np.array([ep.x - entry_pt.x, ep.y - entry_pt.y])
        norm = np.linalg.norm(vec_to_ep)
        if norm < 1e-12:
            continue
        vec_to_ep = vec_to_ep / norm
        # 2D cross product (z-component of 3D cross product)
        cross = float(arrival_vec[0] * vec_to_ep[1] - arrival_vec[1] * vec_to_ep[0])
        scored.append((cross, ep))

    if not scored:
        return None

    if turn_direction == "left":
        # Highest cross product = most to the left
        scored.sort(key=lambda x: -x[0])
    else:  # right
        # Lowest cross product = most to the right
        scored.sort(key=lambda x: x[0])

    return scored[0][1]


def compute_colored_segments(
    origin_runway: Optional[str],
    current_route: list[str],
    prev_routes: list[list[dict]],
    last_taxiway: Optional[str] = None,
    last_confirmed_taxiway: Optional[str] = None,
    last_confirmed_point: Optional[Point] = None,
    confirmed_runway_exit: Optional[str] = None,
    pending_route: Optional[list[str]] = None,
    bfs_bridges: Optional[list[str]] = None,
    turn_direction: Optional[str] = None,
) -> tuple:
    """
    Compute colored line segments for the map, using Shapely for all geometry.

    Returns (segments, route_segs_for_history, new_confirmed_taxiway, new_confirmed_point)

    Option A — confirmed-point tracking
    ────────────────────────────────────
    As we walk the current route, each consecutive pair with a real geometric
    intersection is a *confirmed* transition.  We record the last confirmed
    taxiway and intersection point.  The final taxiway (no confirmed outgoing
    connection) is the *uncertain tail* — colored orange to its end, but
    stored so the next command can trim or anchor from the confirmed point
    rather than from the uncertain tail end.

    Handoff priority for new commands
    ──────────────────────────────────
    1. Runway ∩ first taxiway (landing context)
    2. last_confirmed_taxiway ∩ first taxiway (preferred — geometrically verified)
    3. last_taxiway ∩ first taxiway (uncertain tail fallback)
    4. Nearest-point gap bridge (last resort)
    """
    segments: list[dict] = []
    _nearest_points = nearest_points

    # ── helpers ────────────────────────────────────────────────────────────
    def taxiway_aeroway(ref: str) -> str:
        feats = airport_data["taxiway_features"].get(ref, [])
        return feats[0].get("properties", {}).get("aeroway", "taxiway") if feats else "taxiway"

    def emit_full(ref: str, color: str):
        """Emit one colored segment per original GeoJSON feature for this taxiway ref."""
        feats = airport_data["taxiway_features"].get(ref, [])
        if not feats:
            return
        aeroway = taxiway_aeroway(ref)
        for fi, feat in enumerate(feats):
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            if geom.get("type") == "LineString" and len(coords) >= 2:
                segments.append({"ref": ref, "aeroway": aeroway, "color": color,
                                  "feat_idx": fi,
                                  "coords": [[c[0], c[1]] for c in coords]})

    def _coords_between(seg: LineString, d_start: float, d_end: float) -> list[list[float]]:
        """Extract the coordinate slice of `seg` between two normalized distances."""
        if d_end - d_start < 1e-9:
            return []
        total_len = seg.length
        p_start = seg.interpolate(d_start, normalized=True)
        p_end   = seg.interpolate(d_end,   normalized=True)
        result  = [[p_start.x, p_start.y]]
        lo, hi  = d_start * total_len, d_end * total_len
        for x, y in seg.coords:
            d = seg.project(Point(x, y))
            if lo < d < hi:
                result.append([x, y])
        result.append([p_end.x, p_end.y])
        return result if len(result) >= 2 else []

    def _snap_to_feature(from_pt: Point, feat_coords: list) -> tuple[int, float]:
        """
        Find the nearest vertex index in feat_coords to from_pt, and return
        (vertex_index, normalized_distance_along_line).
        Uses vertex proximity rather than Shapely project() to avoid floating
        point precision issues on near-degenerate segments.
        """
        line = LineString([(c[0], c[1]) for c in feat_coords])
        best_idx, best_dist = 0, float("inf")
        for i, c in enumerate(feat_coords):
            d = Point(c[0], c[1]).distance(from_pt)
            if d < best_dist:
                best_dist, best_idx = d, i
        # Compute normalized distance to the nearest vertex
        d_along = line.project(Point(feat_coords[best_idx][0], feat_coords[best_idx][1]),
                               normalized=True)
        return best_idx, d_along

    def emit_partial(ref: str, color: str, from_pt: Point, to_pt: Optional[Point]):
        """
        Color the portion of taxiway `ref` that the plane traversed.

        Uses original GeoJSON feature coordinates for output (one segment per OSM way).
        Builds an endpoint-adjacency graph of the features, then finds the shortest path
        from the entry feature to the exit feature via BFS.  Features on the path are
        colored fully; the entry feature is partially colored from the entry point onward;
        the exit feature is partially colored up to the exit point.

        If entry and exit fall on topologically disconnected features (e.g. isolated stubs),
        we re-select the entry feature as the closest *connected* one that has a path to
        the exit feature.
        """
        feats = airport_data["taxiway_features"].get(ref, [])
        if not feats:
            return
        aeroway = taxiway_aeroway(ref)

        feat_lines: list[tuple] = []  # (LineString, [[x,y]...])
        for feat in feats:
            geom_json = feat.get("geometry", {})
            if geom_json.get("type") != "LineString":
                continue
            raw_coords = geom_json["coordinates"]
            if len(raw_coords) < 2:
                continue
            feat_lines.append((LineString([(c[0], c[1]) for c in raw_coords]),
                                [[c[0], c[1]] for c in raw_coords]))

        if not feat_lines:
            return

        if to_pt is None:
            for line, raw in feat_lines:
                segments.append({"ref": ref, "aeroway": aeroway, "color": color,
                                  "coords": raw})
            return

        # ── Build endpoint adjacency graph ────────────────────────────────
        # Two features are adjacent when one endpoint of one is close to an endpoint
        # of the other (tolerance ~5m).
        ADJ_TOL = 0.00005
        def ep_close(p1, p2):
            return abs(p1[0]-p2[0]) < ADJ_TOL and abs(p1[1]-p2[1]) < ADJ_TOL

        def endpoints_of(idx):
            r = feat_lines[idx][1]
            return (r[0][0], r[0][1]), (r[-1][0], r[-1][1])

        adj: dict[int, list[int]] = {i: [] for i in range(len(feat_lines))}
        for i in range(len(feat_lines)):
            ai, bi = endpoints_of(i)
            for j in range(len(feat_lines)):
                if i == j: continue
                aj, bj = endpoints_of(j)
                if ep_close(bi, aj) or ep_close(bi, bj) or ep_close(ai, aj) or ep_close(ai, bj):
                    if j not in adj[i]:
                        adj[i].append(j)

        def bfs_path(start: int, goal: int) -> Optional[list[int]]:
            if start == goal:
                return [start]
            q = deque([[start]])
            visited = {start}
            while q:
                path = q.popleft()
                for nb in adj[path[-1]]:
                    if nb == goal:
                        return path + [nb]
                    if nb not in visited:
                        visited.add(nb)
                        q.append(path + [nb])
            return None

        # ── Find best exit feature (closest to to_pt) ────────────────────
        best_to_idx, best_to_dist = 0, float("inf")
        for idx, (line, _) in enumerate(feat_lines):
            d = line.distance(to_pt)
            if d < best_to_dist:
                best_to_dist, best_to_idx = d, idx

        # ── Find best entry feature — must have a path to exit ───────────
        # Try candidates in order of distance from from_pt; pick the closest
        # one that is topologically connected to the exit feature.
        dists_from = [(feat_lines[i][0].distance(from_pt), i) for i in range(len(feat_lines))]
        dists_from.sort()

        best_from_idx = None
        feature_path = None
        for _, idx in dists_from:
            path = bfs_path(idx, best_to_idx)
            if path is not None:
                best_from_idx = idx
                feature_path = path
                break

        # Fallback: no connected path found — just use geometrically closest features
        if best_from_idx is None:
            best_from_idx = dists_from[0][1]
            feature_path = [best_from_idx] if best_from_idx == best_to_idx else [best_from_idx, best_to_idx]

        # ── Emit segments ─────────────────────────────────────────────────
        path_set = set(feature_path)

        for idx, (line, raw) in enumerate(feat_lines):
            if idx not in path_set:
                # Not on the traversed path — grey
                segments.append({"ref": ref, "aeroway": aeroway,
                                  "color": "grey", "feat_idx": idx, "coords": raw})
                continue

            is_entry = (idx == best_from_idx)
            is_exit  = (idx == best_to_idx)
            is_only  = (is_entry and is_exit)

            if is_only:
                # Entry and exit on same feature
                d_from = line.project(from_pt, normalized=True)
                d_to   = line.project(to_pt,   normalized=True)
                if d_from > d_to:
                    d_from, d_to = d_to, d_from
                colored = _coords_between(line, d_from, d_to)
                before  = _coords_between(line, 0.0, d_from)
                after   = _coords_between(line, d_to, 1.0)
                for c, clr in [(colored, color), (before, "grey"), (after, "grey")]:
                    if c:
                        segments.append({"ref": ref, "aeroway": aeroway,
                                          "color": clr, "feat_idx": idx, "coords": c})

            elif is_entry:
                # Partial: from entry point toward exit side.
                # If from_pt is within BUFFER_TOLERANCE of the endpoint that is
                # AWAY from the exit feature, the aircraft traverses the whole
                # feature — color it fully.
                best_v, _ = _snap_to_feature(from_pt, raw)
                c0 = Point(raw[0]); cN = Point(raw[-1])
                exit_line_feat = feat_lines[best_to_idx][0]
                exit_closer_to_c0 = exit_line_feat.distance(c0) <= exit_line_feat.distance(cN)

                # Check if from_pt is near the far end (away from exit) — full traversal
                far_end = cN if exit_closer_to_c0 else c0
                if from_pt.distance(far_end) < BUFFER_TOLERANCE * 3:
                    segments.append({"ref": ref, "aeroway": aeroway,
                                      "color": color, "feat_idx": idx, "coords": raw})
                else:
                    if exit_closer_to_c0:
                        colored = [[c[0], c[1]] for c in line.coords[:best_v + 1]]
                        grey    = [[c[0], c[1]] for c in line.coords[best_v:]]
                    else:
                        colored = [[c[0], c[1]] for c in line.coords[best_v:]]
                        grey    = [[c[0], c[1]] for c in line.coords[:best_v + 1]]
                    for c, clr in [(colored, color), (grey, "grey")]:
                        if c and len(c) >= 2:
                            segments.append({"ref": ref, "aeroway": aeroway,
                                              "color": clr, "feat_idx": idx, "coords": c})

            elif is_exit:
                # Partial: from entry side up to exit point.
                # If the exit point is within BUFFER_TOLERANCE of either endpoint
                # of this feature, the plane traverses the whole feature — color fully.
                c0, cN = Point(raw[0]), Point(raw[-1])
                near_start = to_pt.distance(c0) < BUFFER_TOLERANCE * 3
                near_end   = to_pt.distance(cN) < BUFFER_TOLERANCE * 3
                if near_start or near_end:
                    segments.append({"ref": ref, "aeroway": aeroway,
                                      "color": color, "feat_idx": idx, "coords": raw})
                else:
                    d_to = line.project(to_pt, normalized=True)
                    if to_pt.distance(c0) <= to_pt.distance(cN):
                        colored = _coords_between(line, 0.0, d_to)
                        grey    = _coords_between(line, d_to, 1.0)
                    else:
                        colored = _coords_between(line, d_to, 1.0)
                        grey    = _coords_between(line, 0.0, d_to)
                    for c, clr in [(colored, color), (grey, "grey")]:
                        if c:
                            segments.append({"ref": ref, "aeroway": aeroway,
                                              "color": clr, "feat_idx": idx, "coords": c})

            else:
                # Intermediate on path — fully colored
                segments.append({"ref": ref, "aeroway": aeroway,
                                  "color": color, "feat_idx": idx, "coords": raw})

    # ── 1. Runway coloring ────────────────────────────────────────────────
    if origin_runway:
        rwy_ref = rwy_geom = None
        for ref, geom in airport_data["runway_geoms"].items():
            if origin_runway == ref or origin_runway in ref.split("/"):
                rwy_ref, rwy_geom = ref, geom
                break

        if rwy_geom:
            entry_pt: Point = _runway_entry_point(rwy_geom, origin_runway)

            exit_pt: Optional[Point] = None

            # If a confirmed runway exit taxiway is already locked from a previous
            # command, always use that — never recalculate from the current route.
            if confirmed_runway_exit:
                exit_tw_geom = airport_data["taxiway_geoms"].get(confirmed_runway_exit)
                if exit_tw_geom:
                    exit_pt = _intersection_point(rwy_geom, exit_tw_geom)

            # Otherwise find the exit from the current or previous routes
            if exit_pt is None:
                if current_route:
                    first_tw_geom = airport_data["taxiway_geoms"].get(current_route[0])
                    if first_tw_geom:
                        exit_pt = _intersection_point(rwy_geom, first_tw_geom)
                if exit_pt is None and prev_routes:
                    for route in reversed(prev_routes):
                        if route:
                            first_tw_geom = airport_data["taxiway_geoms"].get(route[0]["ref"])
                            if first_tw_geom:
                                cand = _intersection_point(rwy_geom, first_tw_geom)
                                if cand:
                                    exit_pt = cand
                                    break

            for seg in _geom_to_linestrings(rwy_geom):
                raw = [[x, y] for x, y in seg.coords]
                if exit_pt is None:
                    # Full runway red (landing clearance, no exit yet)
                    segments.append({"ref": rwy_ref, "aeroway": "runway",
                                     "color": "red", "coords": raw})
                else:
                    if seg.buffer(BUFFER_TOLERANCE).contains(exit_pt):
                        before, after = _split_line_at_point(seg, exit_pt)
                        # Decide which half is the "entry side" using entry_pt
                        # Entry side = the half closer to entry_pt
                        if len(before) >= 2 and len(after) >= 2:
                            mid_before = LineString(before).interpolate(0.5, normalized=True)
                            mid_after  = LineString(after).interpolate(0.5, normalized=True)
                            if entry_pt.distance(mid_before) < entry_pt.distance(mid_after):
                                red_part, grey_part = before, after
                            else:
                                red_part, grey_part = after, before
                            segments.append({"ref": rwy_ref, "aeroway": "runway",
                                             "color": "red",  "coords": red_part})
                            segments.append({"ref": rwy_ref, "aeroway": "runway",
                                             "color": "grey", "coords": grey_part})
                        elif len(before) >= 2:
                            segments.append({"ref": rwy_ref, "aeroway": "runway",
                                             "color": "red", "coords": before})
                        elif len(after) >= 2:
                            segments.append({"ref": rwy_ref, "aeroway": "runway",
                                             "color": "red", "coords": after})
                    else:
                        # Color entire segment based on which side of exit it's on
                        mid_seg = seg.interpolate(0.5, normalized=True)
                        dist_entry_to_mid  = entry_pt.distance(mid_seg)
                        dist_entry_to_exit = entry_pt.distance(exit_pt)
                        color = "red" if dist_entry_to_mid < dist_entry_to_exit else "grey"
                        segments.append({"ref": rwy_ref, "aeroway": "runway",
                                         "color": color, "coords": raw})

    # ── 2. Taxiway coloring ───────────────────────────────────────────────
    if not current_route:
        # Just emit history yellow using stored bounds
        for prev_seg_list in prev_routes:
            for seg_info in prev_seg_list:
                ref = seg_info["ref"]
                fp = seg_info.get("from_pt")
                tp = seg_info.get("to_pt")
                if fp is None:
                    emit_full(ref, "yellow")
                else:
                    emit_partial(ref, "yellow", Point(fp), Point(tp) if tp else None)
        return segments, [], last_confirmed_taxiway, last_confirmed_point

    current_refs = set(current_route)

    # ── Find the handoff point onto the first taxiway ─────────────────────
    # Priority:
    #   0. First taxiway already in prev_routes — reuse its stored from_pt
    #      so we don't re-derive a different entry point from ZA geometry.
    #   1. Runway ∩ first taxiway (from landing context)
    #   2. Any taxiway in the previous route that directly intersects the first
    #      taxiway of the new route — searched in reverse order (most recent first).
    #      This correctly finds ZA∩E even when ZA is not the last taxiway.
    #   3. Nearest-point gap bridge from last_confirmed_taxiway (last resort)
    handoff_pt: Optional[Point] = None
    handoff_from_ref: Optional[str] = None  # which prev taxiway provided the handoff
    rwy_geom_for_handoff = None
    if origin_runway:
        for ref, geom in airport_data["runway_geoms"].items():
            if origin_runway == ref or origin_runway in ref.split("/"):
                rwy_geom_for_handoff = geom
                break

    first_tw_geom = airport_data["taxiway_geoms"].get(current_route[0])
    if first_tw_geom:
        # 0. If the first taxiway appeared in a previous route, reuse its
        #    stored from_pt — this gives the correct entry geometry rather
        #    than re-deriving from an adjacent taxiway's intersection.
        for prev_seg_list in reversed(prev_routes):
            for seg_info in prev_seg_list:
                if seg_info["ref"] == current_route[0]:
                    fp = seg_info.get("from_pt")
                    if fp is not None:
                        handoff_pt = Point(fp)
                    break
            if handoff_pt is not None:
                break

        # 1. Runway context — only if this is the very first taxiway command after
        #    landing (prev_routes is empty). Once the plane is in the taxiway system,
        #    the runway handoff is no longer valid — use prev-route scan instead.
        if handoff_pt is None and rwy_geom_for_handoff and not prev_routes:
            cand = _intersection_point(rwy_geom_for_handoff, first_tw_geom)
            if cand:
                _, p_first = _nearest_points(rwy_geom_for_handoff, first_tw_geom)
                handoff_pt = p_first

        # 2. Search ALL taxiways from all previous routes for a real intersection
        #    with the first taxiway of the new command. Search reverse-chronological
        #    so the most recent matching taxiway wins.
        if handoff_pt is None and prev_routes:
            for prev_seg_list in reversed(prev_routes):
                if handoff_pt is not None:
                    break
                for seg_info in reversed(prev_seg_list):
                    prev_ref = seg_info["ref"]
                    if prev_ref == current_route[0]:
                        continue
                    prev_geom = airport_data["taxiway_geoms"].get(prev_ref)
                    if prev_geom:
                        cand = _intersection_point(prev_geom, first_tw_geom)
                        if cand is not None:
                            # Snap to the actual touching point on the target taxiway
                            _, p_first = _nearest_points(prev_geom, first_tw_geom)
                            handoff_pt = p_first
                            handoff_from_ref = prev_ref
                            break

        # 3. Nearest-point gap bridge from best available anchor
        if handoff_pt is None:
            anchor_ref = last_confirmed_taxiway or last_taxiway
            if anchor_ref:
                anchor_geom = airport_data["taxiway_geoms"].get(anchor_ref)
                if anchor_geom:
                    _, p_first = _nearest_points(anchor_geom, first_tw_geom)
                    handoff_pt = p_first

    # ── Color each taxiway in the current route ───────────────────────────
    route_segments_for_history: list[dict] = []
    bfs_bridge_set: set[str] = set(bfs_bridges or [])

    # Confirmed taxiway = first taxiway in route that touches the origin runway.
    # This is the definitive "plane exited runway here" anchor for the next command.
    rwy_adjacent_set = runway_taxiways_for(origin_runway) if origin_runway else set()
    new_confirmed_taxiway: Optional[str] = None
    new_confirmed_point: Optional[Point] = None
    if origin_runway:
        rwy_g_for_conf = None
        for rk, rg in airport_data["runway_geoms"].items():
            if origin_runway == rk or origin_runway in rk.split("/"):
                rwy_g_for_conf = rg
                break
        for ref in current_route:
            if ref in rwy_adjacent_set:
                tw_g = airport_data["taxiway_geoms"].get(ref)
                if rwy_g_for_conf and tw_g:
                    cand = _intersection_point(rwy_g_for_conf, tw_g)
                    if cand:
                        new_confirmed_taxiway = ref
                        new_confirmed_point = cand
                        break

    for i, ref in enumerate(current_route):
        tw_geom = airport_data["taxiway_geoms"].get(ref)
        if tw_geom is None:
            continue

        is_first = (i == 0)
        is_last  = (i == len(current_route) - 1)

        # ── Entry point ───────────────────────────────────────────────────
        # i=0: use handoff_pt (runway exit or prev-route ZA∩E etc.)
        # i>0: real intersection with previous taxiway, else nearest point on
        #      THIS taxiway to the previous taxiway's exit point (gap bridge)
        entry: Optional[Point] = None

        if is_first:
            entry = handoff_pt  # may be None → emit_full below
        else:
            prev_ref  = current_route[i - 1]
            prev_geom = airport_data["taxiway_geoms"].get(prev_ref)
            if prev_geom:
                cand = _intersection_point(prev_geom, tw_geom)
                if cand is not None:
                    entry = cand          # confirmed connection
                else:
                    # Gap — nearest point on THIS taxiway toward previous
                    _, p_tw = _nearest_points(prev_geom, tw_geom)
                    entry = p_tw

        # ── Exit point ────────────────────────────────────────────────────
        # Last taxiway: no exit — color the full taxiway (destination unknown)
        # Otherwise: real intersection with next taxiway, else nearest point
        #            on THIS taxiway toward the next one (gap model)
        exit_: Optional[Point] = None

        if not is_last:
            next_ref  = current_route[i + 1]
            next_geom = airport_data["taxiway_geoms"].get(next_ref)
            if next_geom:
                cand = _intersection_point(tw_geom, next_geom)
                if cand is not None:
                    exit_ = cand          # confirmed connection
                else:
                    # Gap — nearest point on THIS taxiway toward next
                    p_tw, _ = _nearest_points(tw_geom, next_geom)
                    exit_ = p_tw

        # ── Emit ──────────────────────────────────────────────────────────
        # BFS-inferred bridges are purple; ATC-stated taxiways are orange.
        # All taxiways — whether ATC-stated or BFS-inserted — use entry/exit
        # intersection points to determine partial vs full coloring.
        # Only the last taxiway (no exit known) gets emit_full unconditionally.
        seg_color = "purple" if ref in bfs_bridge_set else "orange"

        if is_last:
            # Last taxiway — if turn_direction is known, color only the half
            # the aircraft will traverse based on arrival heading from prev taxiway.
            # Otherwise fall back to full coloring (destination unknown).
            #
            # entry is already the intersection point between prev taxiway and
            # this one — reuse it directly as the boarding point on this taxiway.
            # Arrival heading is derived from the prev taxiway's geometry near entry.
            partial_colored = False
            if turn_direction and i > 0 and entry is not None:
                prev_ref = current_route[i - 1]
                arrival_vec = _arrival_heading_vec(prev_ref, entry)
                if arrival_vec is not None:
                    dest_ep = _turn_direction_endpoint(ref, entry, arrival_vec, turn_direction)
                    if dest_ep is not None:
                        emit_partial(ref, seg_color, entry, dest_ep)
                        route_segments_for_history.append({
                            "ref": ref,
                            "from_pt": [entry.x,   entry.y],
                            "to_pt":   [dest_ep.x, dest_ep.y],
                        })
                        partial_colored = True
                        print(f"[TURN] {turn_direction} onto {ref}: "
                              f"entry={[round(entry.x,5), round(entry.y,5)]} "
                              f"dest_ep={[round(dest_ep.x,5), round(dest_ep.y,5)]} "
                              f"arrival_vec={arrival_vec.tolist()}")

            if not partial_colored:
                emit_full(ref, seg_color)
                route_segments_for_history.append({"ref": ref, "from_pt": None, "to_pt": None})
        elif entry is not None and exit_ is not None:
            emit_partial(ref, seg_color, entry, exit_)
            route_segments_for_history.append({"ref": ref,
                "from_pt": [entry.x, entry.y],
                "to_pt":   [exit_.x, exit_.y]})
        elif exit_ is not None:
            # No confirmed entry (first taxiway with no handoff) — color from
            # the nearest endpoint of this taxiway toward next
            feats_tw = airport_data["taxiway_features"].get(ref, [])
            if feats_tw:
                first_feat_coords = feats_tw[0]["geometry"]["coordinates"]
                start_pt = Point(first_feat_coords[0][0], first_feat_coords[0][1])
            else:
                start_pt = Point(tw_geom.interpolate(0, normalized=True))
            emit_partial(ref, seg_color, start_pt, exit_)
            route_segments_for_history.append({"ref": ref,
                "from_pt": [start_pt.x, start_pt.y],
                "to_pt":   [exit_.x,   exit_.y]})
        else:
            emit_full(ref, seg_color)
            route_segments_for_history.append({"ref": ref, "from_pt": None, "to_pt": None})

    # ── Emit previous routes as yellow ────────────────────────────────────
    current_refs = set(current_route)
    for prev_seg_list in prev_routes:
        for seg_info in prev_seg_list:
            ref = seg_info["ref"]
            if ref in current_refs:
                continue
            fp = seg_info.get("from_pt")
            tp = seg_info.get("to_pt")
            if fp is None:
                emit_full(ref, "yellow")
            else:
                emit_partial(ref, "yellow", Point(fp), Point(tp) if tp else None)

    # ── Emit pending taxiways as yellow ───────────────────────────────────
    # These are taxiways ATC mentioned but whose path has a broken link.
    # They stay yellow until a subsequent command validates the full chain.
    if pending_route:
        pending_refs = set(pending_route) - current_refs
        for ref in pending_route:
            if ref not in pending_refs:
                continue
            emit_full(ref, "yellow")

    return segments, route_segments_for_history, new_confirmed_taxiway, new_confirmed_point
