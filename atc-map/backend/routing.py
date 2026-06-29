"""
Route resolver — disambiguation via geometry
─────────────────────────────────────────────
Turns the raw phonetic letters parsed from an ATC transcript into a concrete
taxiway route. Compound names (e.g. "ZA" vs "Z then A") are disambiguated by
checking which grouping forms a geometrically connected chain, biased toward
the landing runway. Broken chains are repaired with a bounded BFS bridge.
"""

from collections import deque
from typing import Optional

from state import airport_data
from geometry import BUFFER_TOLERANCE, taxiways_intersect, get_intersection_point

# NATO phonetic alphabet mapping
PHONETIC = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H",
    "india": "I", "juliet": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X",
    "yankee": "Y", "zulu": "Z",
}

BFS_MAX_DEPTH = 2          # max intermediate hops for mid-route gaps
BFS_MAX_DEPTH_RUNWAY = 4   # wider search when bridging from the runway exit taxiway


def phonetic_to_letters(names: list[str]) -> list[str]:
    """Convert phonetic names to letters. E.g. ['Echo', 'Foxtrot'] -> ['E', 'F']"""
    result = []
    for name in names:
        letter = PHONETIC.get(name.lower(), name.upper())
        result.append(letter)
    return result


def generate_groupings(letters: list[str], valid_refs: set) -> list[list[str]]:
    """
    Generate all possible groupings of letters into valid taxiway refs.

    E.g. ['E', 'F', 'A'] with valid_refs {'E', 'F', 'A', 'FA', 'EF'}
    -> [['E', 'F', 'A'], ['E', 'FA'], ['EF', 'A']]
    """
    if not letters:
        return [[]]

    results = []
    for end in range(1, len(letters) + 1):
        combined = "".join(letters[:end])
        if combined in valid_refs:
            for rest in generate_groupings(letters[end:], valid_refs):
                results.append([combined] + rest)
    return results


def check_route_connectivity(route: list[str]) -> tuple[bool, list]:
    """
    Check if a route has valid intersections between all consecutive pairs.
    Returns (is_valid, intersection_points).
    """
    if len(route) < 2:
        return True, []

    points = []
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        if not taxiways_intersect(a, b):
            return False, []
        pt = get_intersection_point(a, b)
        if pt:
            points.append({"from": a, "to": b, "point": pt})
    return True, points


def _bfs_bridge(src: str, dst: str, max_depth: int = BFS_MAX_DEPTH) -> Optional[list[str]]:
    """
    Find the shortest path from src to dst through taxiway_connections
    using BFS, up to max_depth intermediate hops.

    Neighbors are sorted alphabetically at each step so the result is
    deterministic and lexicographically minimal when multiple shortest
    paths exist.

    Returns the list of INTERMEDIATE taxiways only (not src or dst),
    or None if no path exists within the depth limit.

    Example: src="ZA", dst="F", path ZA->E->B->F  →  returns ["E", "B"]
    Example: src="E",  dst="F", path E->B->F       →  returns ["B"]
    """
    if src == dst:
        return []

    tc = airport_data["taxiway_connections"]

    queue = deque()
    queue.append((src, []))
    visited = {src}

    while queue:
        node, intermediates = queue.popleft()

        for neighbor in sorted(tc.get(node, set())):
            if neighbor == dst:
                return intermediates
            if neighbor not in visited and len(intermediates) < max_depth:
                visited.add(neighbor)
                queue.append((neighbor, intermediates + [neighbor]))

    return None


def validate_route_partial(
    route: list[str],
    last_confirmed: Optional[str],
    origin_runway: Optional[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Walk the route step by step using taxiway_connections.

    Direct connections are always preferred. BFS bridge is only attempted
    when:
      1. prev→curr is not directly connected, AND
      2. curr onward forms a valid direct chain (i.e. the rest of the route
         is internally consistent — ATC just omitted the connector before curr)

    This prevents BFS from firing prematurely on genuinely broken routes
    (where the rest of the route is also disconnected).

    Returns (confirmed_route, bfs_bridges, pending_route):
      - confirmed_route: all taxiways in the validated path including BFS bridges
      - bfs_bridges:     subset inserted by BFS (not stated by ATC) → purple
      - pending_route:   taxiways ATC mentioned but couldn't be validated yet

    Examples:
      CMD2: route=["ZA","F","A"], last_confirmed=None, origin_runway="13L"
        → ZA valid (13L exit)
        → ZA→F broken; check F→A: direct ✅ so try BFS ZA→F → finds [E,B]
        → confirmed=["ZA","E","B","F","A"], bfs_bridges=["E","B"], pending=[]

      CMD2 (unbridgeable): route=["ZA","X","Y"], ZA→X broken, X→Y broken
        → ZA→X broken; X→Y also broken → rest not a valid chain → pending
        → confirmed=["ZA"], bfs_bridges=[], pending=["X","Y"]
    """
    if not route:
        return [], [], []

    tc = airport_data["taxiway_connections"]
    re = airport_data["runway_entry_taxiways"]

    def chain_is_valid(sub_route: list[str]) -> bool:
        """Check all consecutive pairs in sub_route are directly connected."""
        for i in range(len(sub_route) - 1):
            if sub_route[i + 1] not in tc.get(sub_route[i], set()):
                return False
        return True

    # ── Validate / bridge the first taxiway ──────────────────────────────
    first = route[0]
    first_valid = False
    bridge_prefix: list[str] = []

    if last_confirmed:
        if first in tc.get(last_confirmed, set()):
            first_valid = True
        elif chain_is_valid(route):
            bridge = _bfs_bridge(last_confirmed, first)
            if bridge is not None:
                bridge_prefix = bridge
                first_valid = True
        else:
            remaining_from_first = route
            if chain_is_valid(remaining_from_first[1:]) if len(remaining_from_first) > 1 else True:
                bridge = _bfs_bridge(last_confirmed, first)
                if bridge is not None:
                    bridge_prefix = bridge
                    first_valid = True

    if not first_valid and origin_runway:
        if first in re.get(origin_runway, set()):
            first_valid = True

    if not first_valid:
        return [], [], route

    # ── Walk forward ──────────────────────────────────────────────────────
    confirmed: list[str] = bridge_prefix + [first]
    bfs_bridges: list[str] = list(bridge_prefix)

    # Use a wider BFS depth for the first command after landing (last_confirmed
    # is None) — the aircraft may be several taxiways away from the ATC route.
    walk_bfs_depth = BFS_MAX_DEPTH if last_confirmed else BFS_MAX_DEPTH_RUNWAY

    for i in range(1, len(route)):
        prev = confirmed[-1]
        curr = route[i]

        if curr in tc.get(prev, set()):
            confirmed.append(curr)
        else:
            remaining = route[i:]
            if chain_is_valid(remaining):
                bridge = _bfs_bridge(prev, curr, max_depth=walk_bfs_depth)
                if bridge is not None:
                    confirmed.extend(bridge)
                    confirmed.append(curr)
                    bfs_bridges.extend(bridge)
                    continue
            return confirmed, bfs_bridges, route[i:]

    return confirmed, bfs_bridges, []


def runway_taxiways_for(runway_ref: Optional[str]) -> set:
    """
    Return the set of taxiway refs accessible from the given runway landing
    direction (e.g. '13L'). Uses the directional entry index built at load
    time. Falls back to the undirected runway_taxiways index if the directional
    index has no entry (e.g. runway ref not yet resolved).
    """
    if not runway_ref:
        return set()

    # Exact match in directional index first (e.g. '13L')
    if runway_ref in airport_data["runway_entry_taxiways"]:
        return airport_data["runway_entry_taxiways"][runway_ref]

    # Partial match: '13L' might be stored as part of '13L/31R' key in fallback
    for key, taxiways in airport_data["runway_taxiways"].items():
        if runway_ref in key.split("/"):
            return taxiways

    return set()


def resolve_route(raw_names: list[str], origin_runway: Optional[str] = None) -> dict:
    """
    Given raw taxiway names from ATC transcript, resolve to actual route
    using geometric intersection validation.

    origin_runway: if provided (e.g. '13L'), candidate groupings whose first
    taxiway touches that runway are strongly preferred, resolving ambiguities
    like 'ZA' vs 'Z then A' when the plane just landed on 13L.
    """
    valid_refs = airport_data["valid_refs"]

    # Convert phonetic to letters
    letters = phonetic_to_letters(raw_names)

    # Generate all possible groupings
    groupings = generate_groupings(letters, valid_refs)

    if not groupings:
        return {
            "resolved_route": letters,
            "method": "fallback_no_valid_grouping",
            "intersections": [],
            "path_coordinates": [],
            "all_candidates": [],
            "origin_runway": origin_runway,
        }

    # Check each grouping for connectivity
    candidates = []
    for g in groupings:
        valid, points = check_route_connectivity(g)
        candidates.append({
            "route": g,
            "valid": valid,
            "intersections": points,
        })

    # Determine which taxiways touch the origin runway (if known)
    rwy_adjacent: set = runway_taxiways_for(origin_runway)

    def score(c: dict) -> tuple:
        """Higher is better. Primary: connected to runway. Secondary: path length."""
        first_on_runway = bool(rwy_adjacent and c["route"] and c["route"][0] in rwy_adjacent)
        return (int(c["valid"]), int(first_on_runway), len(c["route"]))

    # Pick the first valid one (prefer longer routes for specificity)
    valid_candidates = [c for c in candidates if c["valid"]]

    if valid_candidates:
        best = max(valid_candidates, key=score)
    else:
        # No valid intersections found — use runway-adjacency to pick the
        # best grouping even without confirmed connectivity.
        # E.g. if 13L is known, prefer ['ZA','F','A'] over ['Z','A','F','A']
        # because ZA touches 13L while Z does not.
        if rwy_adjacent:
            rwy_biased = [c for c in candidates if c["route"] and c["route"][0] in rwy_adjacent]
            pool = rwy_biased if rwy_biased else candidates
        else:
            pool = candidates
        # Among candidates in the pool, prefer longer first segment (more specific)
        best = max(pool, key=lambda c: len(c["route"][0]) if c["route"] else 0) if pool else \
               {"route": letters, "intersections": []}

    # Build the full path with coordinates
    path_coords = build_path_coordinates(best["route"], best.get("intersections", []))

    # Compute the runway exit point: intersection centroid of origin_runway ∩ first taxiway
    runway_exit_point = None
    if origin_runway and best["route"]:
        first_tw = best["route"][0]
        rwy_geom = None
        # Find the runway geometry matching origin_runway (partial match: '13L' in '13L/31R')
        for rwy_ref, rwy_g in airport_data["runway_geoms"].items():
            if origin_runway == rwy_ref or origin_runway in rwy_ref.split("/"):
                rwy_geom = rwy_g
                break
        tw_geom = airport_data["taxiway_geoms"].get(first_tw)
        if rwy_geom and tw_geom:
            ix = rwy_geom.buffer(BUFFER_TOLERANCE).intersection(tw_geom.buffer(BUFFER_TOLERANCE))
            if not ix.is_empty:
                c = ix.centroid
                if not c.is_empty:
                    runway_exit_point = {"lng": c.x, "lat": c.y, "taxiway": first_tw}

    return {
        "resolved_route": best["route"],
        "method": "intersection_validated" if best.get("valid") else "fallback",
        "intersections": best.get("intersections", []),
        "path_coordinates": path_coords,
        "all_candidates": [
            {"route": c["route"], "valid": c["valid"]} for c in candidates
        ],
        "origin_runway": origin_runway,
        "runway_exit_point": runway_exit_point,
    }


def build_path_coordinates(route: list[str], intersections: list[dict]) -> list[dict]:
    """
    Build an ordered list of coordinates along the resolved route.
    Uses intersection points to find the relevant segments of each taxiway.
    """
    if not route or not intersections:
        coords = []
        for ref in route:
            geom = airport_data["taxiway_geoms"].get(ref)
            if geom:
                c = geom.centroid
                coords.append({"ref": ref, "lng": c.x, "lat": c.y, "type": "centroid"})
        return coords

    coords = []
    for i, ix in enumerate(intersections):
        coords.append({
            "ref": f"{ix['from']}∩{ix['to']}",
            "lng": ix["point"][0],
            "lat": ix["point"][1],
            "type": "intersection",
            "from_taxiway": ix["from"],
            "to_taxiway": ix["to"],
        })
    return coords
