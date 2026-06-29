"""
ATC Navigation Parser Backend
─────────────────────────────
FastAPI application: HTTP endpoints that wire together the geometry engine,
route resolver, visualization layer, and LLM parsing into the pipeline the
frontend consumes.

Module layout:
    state.py          shared in-memory state (airport + aircraft)
    geometry.py       GeoJSON loading, intersections, runway-entry index
    routing.py        route disambiguation + BFS bridging
    visualization.py  colored-segment computation for the map
    llm.py            Llama 3 70B parsing + readback checking
    server.py         this file — HTTP API + app wiring

Usage:
    pip install fastapi uvicorn shapely huggingface_hub numpy python-multipart
    export HF_TOKEN="your-huggingface-token"
    python server.py

    # or
    uvicorn server:app --reload --port 8000
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shapely.geometry import Point

from state import airport_data, aircraft_state
from geometry import load_geojson
from routing import resolve_route, validate_route_partial
from visualization import compute_colored_segments
from llm import parse_atc_with_llm, parse_atc_raw, check_readback


# ══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="ATC Nav Parser", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ParseRequest(BaseModel):
    transcript: str


class ParseResponse(BaseModel):
    parsed: dict
    route: dict
    taxiway_refs: list[str]
    runway_refs: list[str]


class ReadbackRequest(BaseModel):
    atc_parsed: dict        # the parsed ATC result (callsign, instruction_type, route, runway, summary)
    pilot_transcript: str   # the raw pilot readback text


# ══════════════════════════════════════════════════════════════════════════════
#  GEOJSON / HEALTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "ATC Nav Parser",
        "status": "ok",
        "airport_loaded": airport_data["geojson"] is not None,
        "taxiway_count": len(airport_data["valid_refs"]),
    }


@app.post("/load-geojson")
async def upload_geojson(file: UploadFile = File(...)):
    """Upload and load airport GeoJSON data."""
    content = await file.read()
    try:
        geojson = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if geojson.get("type") != "FeatureCollection":
        raise HTTPException(status_code=400, detail="Must be a GeoJSON FeatureCollection")

    load_geojson(geojson)

    return {
        "status": "loaded",
        "filename": file.filename,
        "features": len(geojson.get("features", [])),
        "taxiway_refs": sorted(airport_data["valid_refs"]),
        "runway_refs": sorted(airport_data["runway_geoms"].keys()),
        "intersection_pairs": len(airport_data["intersections"]) // 2,
    }


@app.post("/load-geojson-path")
async def load_geojson_from_path(path: str):
    """Load GeoJSON from a local file path."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    with open(p) as f:
        geojson = json.load(f)

    load_geojson(geojson)

    return {
        "status": "loaded",
        "path": str(p),
        "features": len(geojson.get("features", [])),
        "taxiway_refs": sorted(airport_data["valid_refs"]),
        "runway_refs": sorted(airport_data["runway_geoms"].keys()),
        "intersection_pairs": len(airport_data["intersections"]) // 2,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/parse-raw")
async def parse_raw(req: ParseRequest):
    """
    Debug endpoint: call the LLM and return its raw output + parsed JSON,
    without running route resolution or touching aircraft state.
    Useful for tuning the prompt.
    """
    return parse_atc_raw(req.transcript)


@app.post("/parse")
async def parse_transcript(req: ParseRequest):
    """
    Full pipeline: parse ATC transcript -> resolve route -> return path.
    Requires HF_TOKEN env var and GeoJSON to be loaded.

    Maintains per-callsign aircraft state so that a landing clearance on
    runway 13L is remembered when the follow-up taxi instruction arrives,
    enabling correct disambiguation of compound taxiway names (e.g. 'ZA'
    vs 'Z then A') using geometric intersection with the runway.
    """
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded. POST to /load-geojson first.")

    # Step 1: Parse with Llama 3 70B
    parsed = parse_atc_with_llm(req.transcript)

    callsign: str = parsed.get("callsign", "").upper().replace(" ", "")
    instruction_type: str = parsed.get("instruction_type", "")
    parsed_runway: Optional[str] = parsed.get("runway")

    # Step 2: Update aircraft state
    if callsign:
        state = aircraft_state.setdefault(callsign, {
            "runway": None,
            "prev_routes": [],
            "last_taxiway": None,
            "last_confirmed_taxiway": None,
            "last_confirmed_point": None,
            "confirmed_runway_exit": None,   # taxiway ref — locked once confirmed, never changes
            "pending_route": [],             # taxiways ATC mentioned but path was broken
            "bfs_bridges": [],              # taxiways auto-inserted by BFS, not stated by ATC
        })

        if instruction_type == "landing_clearance" and parsed_runway:
            state["runway"] = parsed_runway
            state["confirmed_runway_exit"] = None   # new landing resets the exit
            state["pending_route"] = []             # new landing clears any pending taxiways
            state["bfs_bridges"] = []               # new landing clears BFS bridges
        elif parsed_runway:
            state["runway"] = parsed_runway
    else:
        state = {
            "runway": None,
            "prev_routes": [],
            "last_taxiway": None,
            "last_confirmed_taxiway": None,
            "last_confirmed_point": None,
        }

    # Step 3: Determine origin runway for route disambiguation.
    origin_runway: Optional[str] = parsed_runway or state.get("runway")

    # Step 4: Resolve route using intersection geometry + runway context
    route_raw = parsed.get("route_raw", [])
    last_confirmed_taxiway: Optional[str] = state.get("last_confirmed_taxiway")
    last_confirmed_point_raw = state.get("last_confirmed_point")
    last_confirmed_point: Optional[Point] = (
        Point(last_confirmed_point_raw) if last_confirmed_point_raw else None
    )
    confirmed_runway_exit: Optional[str] = state.get("confirmed_runway_exit")
    prev_routes: list[list[dict]] = list(state.get("prev_routes", []))
    last_taxiway: Optional[str] = state.get("last_taxiway")
    route_result = resolve_route(route_raw, origin_runway=origin_runway) if route_raw else {
        "resolved_route": [],
        "method": "no_route",
        "intersections": [],
        "path_coordinates": [],
        "all_candidates": [],
        "origin_runway": origin_runway,
        "runway_exit_point": None,
    }

    # Step 4b: Validate route step-by-step using taxiway_connections.
    # Splits the resolved route into confirmed, bfs-bridged, and pending segments.
    # Also tries to bridge from last_confirmed_taxiway if the path has a gap.
    resolved: list[str] = route_result.get("resolved_route", [])
    confirmed_route, bfs_bridges, pending_route = validate_route_partial(
        resolved,
        last_confirmed=last_confirmed_taxiway,
        origin_runway=origin_runway,
    )

    # current_route = confirmed + bridges — used for coloring
    current_route: list[str] = confirmed_route

    # last_atc_taxiway = the last ATC-stated taxiway that was directly
    # connected from the previous ATC-stated taxiway (no bridge crossing).
    # This is the safe handoff anchor: it's where the aircraft provably is
    # in the network without relying on any BFS inference.
    # Example: confirmed=[ZA, E, B, F, A], bridges=[E, B]
    #   ZA is ATC-stated, directly from runway ✓  → last_atc = ZA
    #   E  is bridge — skip
    #   B  is bridge — skip
    #   F  is ATC-stated, but reached via bridge E,B → not direct from ZA
    #   A  is ATC-stated, directly from F ✓         → last_atc = A
    # Result: last_atc = A ... but we want ZA for CMD3.
    # Better rule: last ATC-stated taxiway before the first BFS bridge in the route.
    bfs_bridge_set = set(bfs_bridges)
    last_atc_taxiway: Optional[str] = None
    for ref in confirmed_route:
        if ref in bfs_bridge_set:
            break   # stop at the first bridge — anchor is the last ATC node before it
        last_atc_taxiway = ref

    # Step 5: Compute colored segments (runway + taxiways) for the frontend
    turn_direction: Optional[str] = parsed.get("turn_direction")
    if turn_direction not in ("left", "right"):
        turn_direction = None

    colored_segments, route_segs_for_history, new_confirmed_taxiway, new_confirmed_point = \
        compute_colored_segments(
            origin_runway=origin_runway,
            current_route=current_route,
            prev_routes=prev_routes,
            last_taxiway=last_taxiway,
            last_confirmed_taxiway=last_confirmed_taxiway,
            last_confirmed_point=last_confirmed_point,
            confirmed_runway_exit=confirmed_runway_exit,
            pending_route=pending_route,
            bfs_bridges=bfs_bridges,
            turn_direction=turn_direction,
        )

    # Step 6: Update state for next call
    if callsign:
        if current_route:
            state["prev_routes"] = prev_routes + [route_segs_for_history]
            state["last_taxiway"] = current_route[-1]
            # Use last ATC-stated taxiway (not BFS bridge) as the confirmed anchor
            state["last_confirmed_taxiway"] = last_atc_taxiway or new_confirmed_taxiway
            state["last_confirmed_point"] = (
                [new_confirmed_point.x, new_confirmed_point.y]
                if new_confirmed_point else None
            )
            # Lock confirmed_runway_exit once first set — never overwrite
            if state.get("confirmed_runway_exit") is None and new_confirmed_taxiway:
                state["confirmed_runway_exit"] = new_confirmed_taxiway
        # Always persist pending_route and bfs_bridges for next command
        state["pending_route"] = pending_route
        state["bfs_bridges"] = bfs_bridges

    # Debug: print non-grey colored segments being sent to frontend
    non_grey = [s for s in colored_segments if s.get("color") != "grey"]
    print(f"\n[PARSE] callsign={callsign} route={current_route}")
    print(f"[PARSE] colored_segments (non-grey): {len(non_grey)} of {len(colored_segments)} total")
    for s in non_grey:
        print(f"  aeroway={s['aeroway']} ref={s['ref']} feat_idx={s.get('feat_idx','?')} "
              f"color={s['color']} pts={len(s['coords'])} "
              f"start={s['coords'][0] if s['coords'] else '?'}")
    conf_pt_str = (f"[{round(new_confirmed_point.x,5)}, {round(new_confirmed_point.y,5)}]"
                   if new_confirmed_point else None)
    print(f"[PARSE] first_confirmed_taxiway={new_confirmed_taxiway} at {conf_pt_str}")

    return {
        "parsed": parsed,
        "route": route_result,
        "confirmed_route": confirmed_route,
        "bfs_bridges": bfs_bridges,
        "pending_route": pending_route,
        "colored_segments": colored_segments,
        "aircraft_state": aircraft_state.get(callsign, {}),
        "taxiway_refs": sorted(airport_data["valid_refs"]),
        "runway_refs": sorted(airport_data["runway_geoms"].keys()),
    }


@app.post("/resolve-route")
async def resolve_route_endpoint(names: list[str], runway: Optional[str] = None):
    """
    Resolve a route without the LLM — pass taxiway letters directly.
    Optionally supply ?runway=13L to bias disambiguation toward that runway.
    E.g. POST ["Z", "A"]?runway=13L  ->  resolved as ["ZA"]
    """
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded.")

    result = resolve_route(names, origin_runway=runway)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  AIRCRAFT STATE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/aircraft-state")
async def get_aircraft_state():
    """Return the current state of all tracked aircraft."""
    return aircraft_state


@app.get("/aircraft-state/{callsign}")
async def get_aircraft_state_for(callsign: str):
    """Return the current state for a specific callsign."""
    state = aircraft_state.get(callsign.upper().replace(" ", ""))
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state tracked for callsign '{callsign}'")
    return state


@app.delete("/aircraft-state/{callsign}")
async def clear_aircraft_state(callsign: str):
    """Clear the tracked state for a specific callsign (e.g. after pushback)."""
    key = callsign.upper().replace(" ", "")
    aircraft_state.pop(key, None)
    return {"status": "cleared", "callsign": key}


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY INSPECTION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/intersections")
async def get_intersections():
    """Return all pre-computed taxiway intersections."""
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded.")

    pairs = {}
    for (a, b), data in airport_data["intersections"].items():
        if a < b:  # avoid duplicates
            pairs[f"{a}-{b}"] = data["point"]

    return {
        "count": len(pairs),
        "intersections": pairs,
    }


@app.get("/taxiways")
async def get_taxiways():
    """Return all taxiway refs and which others they intersect."""
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded.")

    result = {}
    for ref in sorted(airport_data["valid_refs"]):
        connects = []
        for (a, b) in airport_data["intersections"]:
            if a == ref:
                connects.append(b)
        result[ref] = sorted(connects)

    return result


@app.get("/runway-entry-taxiways")
async def get_runway_entry_taxiways():
    """
    Return the directional runway entry index: for each landing direction
    (e.g. '13L', '31R'), which taxiway refs can be directly entered from
    that runway end based on geometric direction analysis.

    Example response:
    {
        "13L": ["C1", "D", "ZA", ...],
        "31R": ["C3", "D", "U1", ...],
        ...
    }
    """
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded.")

    return {
        rwy: sorted(taxiways)
        for rwy, taxiways in airport_data["runway_entry_taxiways"].items()
    }


@app.get("/taxiway-connections")
async def get_taxiway_connections():
    """
    Return genuine taxiway-to-taxiway connections, with runway phantom
    intersections removed.

    Two taxiways are considered connected only if their intersection is:
      - off the runway, OR
      - on the runway but near a true terminal endpoint of one of the taxiways
        (meaning one taxiway genuinely terminates at that junction)

    Example response:
    {
        "D":  ["A", "B", "C"],
        "MB": ["B", "M", "P", "Q"],
        ...
    }
    """
    if not airport_data["geojson"]:
        raise HTTPException(status_code=400, detail="No GeoJSON loaded.")

    return {
        ref: sorted(neighbors)
        for ref, neighbors in sorted(airport_data["taxiway_connections"].items())
    }


# ══════════════════════════════════════════════════════════════════════════════
#  READBACK CHECK — verify pilot response against ATC instruction
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/check-readback")
async def check_readback_endpoint(req: ReadbackRequest):
    """
    Use the LLM to check whether a pilot readback correctly acknowledges
    the key elements of the preceding ATC instruction.

    Returns { confirmed: bool, reason: str }
    """
    return check_readback(req.atc_parsed, req.pilot_transcript)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    # Auto-load GeoJSON if path provided
    geojson_path = os.environ.get("GEOJSON_PATH")
    if geojson_path and Path(geojson_path).exists():
        print(f"Auto-loading GeoJSON from {geojson_path}")
        with open(geojson_path) as f:
            load_geojson(json.load(f))

    uvicorn.run(app, host="0.0.0.0", port=8000)
