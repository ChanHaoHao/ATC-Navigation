"""
Shared in-memory state for the ATC Navigation backend.
───────────────────────────────────────────────────────
These dictionaries are imported (by reference) across the geometry, routing,
visualization, and server modules. Because they are mutable module-level
objects, every importer sees the same instance — mutating them in one module
is visible everywhere.

Restarting the process clears all state.
"""

# ── Airport geometry index (populated by geometry.load_geojson) ───────────────
airport_data = {
    "geojson": None,
    "taxiway_geoms": {},           # ref -> Shapely geometry (merged)
    "taxiway_features": {},        # ref -> list of GeoJSON features
    "runway_geoms": {},            # ref -> Shapely geometry
    "valid_refs": set(),           # all valid taxiway ref names
    "intersections": {},           # (a, b) -> intersection point (all geometric intersections)
    "taxiway_connections": {},     # ref -> set of refs genuinely connected (runway phantom intersections removed)
    "runway_taxiways": {},         # runway_ref -> set of taxiway refs that touch it (undirected)
    "runway_entry_taxiways": {},   # "13L" -> set of taxiway refs accessible from that landing direction
}

# ── Per-callsign aircraft state ───────────────────────────────────────────────
# e.g. aircraft_state["DAL795"] = {"runway": "13L", "prev_routes": [["ZA","F","A"]]}
aircraft_state: dict[str, dict] = {}
