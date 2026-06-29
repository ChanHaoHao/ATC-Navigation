import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ══════════════════════════════════════════════════════════════════════════════
//  GEO PROJECTION — converts lng/lat to SVG x/y
// ══════════════════════════════════════════════════════════════════════════════
function computeProjection(features, width, height, padding = 80) {
  let minLng = Infinity, maxLng = -Infinity;
  let minLat = Infinity, maxLat = -Infinity;

  const walk = (coords) => {
    if (typeof coords[0] === "number") {
      if (coords[0] < minLng) minLng = coords[0];
      if (coords[0] > maxLng) maxLng = coords[0];
      if (coords[1] < minLat) minLat = coords[1];
      if (coords[1] > maxLat) maxLat = coords[1];
    } else {
      for (const c of coords) walk(c);
    }
  };

  for (const f of features) {
    if (f.geometry?.coordinates) walk(f.geometry.coordinates);
  }

  const lngSpan = maxLng - minLng || 0.01;
  const latSpan = maxLat - minLat || 0.01;
  const drawW = width - padding * 2;
  const drawH = height - padding * 2;
  const scale = Math.min(drawW / lngSpan, drawH / latSpan);
  const cx = (minLng + maxLng) / 2;
  const cy = (minLat + maxLat) / 2;

  return (lng, lat) => [
    width / 2 + (lng - cx) * scale,
    height / 2 - (lat - cy) * scale,
  ];
}

// ══════════════════════════════════════════════════════════════════════════════
//  STYLE CONFIG
// ══════════════════════════════════════════════════════════════════════════════
const LAYER_ORDER = [
  "apron", "parking_position", "taxilane", "taxiway",
  "runway", "terminal", "gate", "holding_position",
];

const LAYER_META = {
  apron:              { label: "Aprons",          stroke: "none",    fill: "#152219",  width: 0,   labelColor: "#3a5c46", labelSize: 8 },
  parking_position:   { label: "Parking",         stroke: "none",    fill: "#2a5540",  width: 0,   labelColor: null },
  taxilane:           { label: "Taxilanes",       stroke: "#265e3f", fill: "none",     width: 1.5, labelColor: "#3a7a55", labelSize: 8 },
  taxiway:            { label: "Taxiways",        stroke: "#3d9e6e", fill: "none",     width: 3,   labelColor: "#5eba8a", labelSize: 11 },
  runway:             { label: "Runways",         stroke: "#b0bec5", fill: "none",     width: 10,  labelColor: "#e0e8ed", labelSize: 14 },
  terminal:           { label: "Terminals",       stroke: "#d4a030", fill: "#1f1c10",  width: 1.5, labelColor: "#e8a735", labelSize: 10 },
  gate:               { label: "Gates",           stroke: "none",    fill: "#4488bb",  width: 0,   labelColor: "#6ab0dd", labelSize: 7 },
  holding_position:   { label: "Hold Positions",  stroke: "#cc3333", fill: "#cc3333",  width: 0,   labelColor: "#ee5555", labelSize: 7 },
};

// ══════════════════════════════════════════════════════════════════════════════
//  A: PER-AIRCRAFT COLOR PALETTE
// ══════════════════════════════════════════════════════════════════════════════
const AIRCRAFT_PALETTE = [
  { current: "#ff9f1c", history: "#ff9f1c55" }, // orange (default)
  { current: "#3b9eff", history: "#3b9eff55" }, // blue
  { current: "#5eba8a", history: "#5eba8a55" }, // green
  { current: "#c97bff", history: "#c97bff55" }, // purple
  { current: "#ff6b6b", history: "#ff6b6b55" }, // coral
  { current: "#00d4d4", history: "#00d4d455" }, // cyan
  { current: "#ffe066", history: "#ffe06655" }, // yellow
  { current: "#ff9de2", history: "#ff9de255" }, // pink
];

function getAircraftPalette(index) {
  return AIRCRAFT_PALETTE[index % AIRCRAFT_PALETTE.length];
}

function formatCallsign(cs) {
  return cs?.replace(/([A-Z]{2,3})(\d+)/, "$1 $2") ?? cs;
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString("en-US", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  SVG PATH BUILDER
// ══════════════════════════════════════════════════════════════════════════════
function toPath(coords, project) {
  // Single ring of coordinates
  if (typeof coords[0][0] === "number") {
    return coords.map((c, i) => {
      const [x, y] = project(c[0], c[1]);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }
  // Nested (polygon ring)
  return toPath(coords[0], project);
}

function midpoint(feature, project) {
  const g = feature.geometry;
  if (g.type === "Point") return project(g.coordinates[0], g.coordinates[1]);
  const flat = g.type === "Polygon" ? g.coordinates[0]
    : g.type === "MultiPolygon" ? g.coordinates[0][0]
    : g.coordinates;
  if (!flat || flat.length === 0) return [0, 0];
  const coords = typeof flat[0][0] === "number" ? flat : flat[0] || flat;
  const mid = Math.floor(coords.length / 2);
  return project(coords[mid][0], coords[mid][1]);
}

// ══════════════════════════════════════════════════════════════════════════════
//  DROP ZONE COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
function DropZone({ onLoad }) {
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        if (json.type === "FeatureCollection" && json.features) {
          onLoad(json, file.name);
        } else {
          alert("Invalid GeoJSON: must be a FeatureCollection");
        }
      } catch (err) {
        alert("Failed to parse JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => fileRef.current?.click()}
      style={{
        width: "100%",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: dragging
          ? "radial-gradient(circle at center, #0f2a1a 0%, #060d09 70%)"
          : "#060d09",
        cursor: "pointer",
        transition: "background 0.3s",
        fontFamily: "'IBM Plex Mono', monospace",
      }}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".geojson,.json"
        style={{ display: "none" }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      <div style={{
        width: 80, height: 80,
        border: `2px dashed ${dragging ? "#5eba8a" : "#1a3528"}`,
        borderRadius: "16px",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: "36px",
        color: dragging ? "#5eba8a" : "#2d5a42",
        marginBottom: "24px",
        transition: "all 0.3s",
      }}>
        ✈
      </div>

      <div style={{
        fontSize: "16px",
        fontWeight: 700,
        color: "#5eba8a",
        letterSpacing: "4px",
        marginBottom: "8px",
      }}>
        ATC AIRPORT MAP
      </div>

      <div style={{
        fontSize: "11px",
        color: "#3d5a48",
        letterSpacing: "1px",
        marginBottom: "32px",
      }}>
        Drop a .geojson file or click to browse
      </div>

      <div style={{
        maxWidth: 460,
        background: "#0a1610",
        border: "1px solid #1a2e22",
        borderRadius: "8px",
        padding: "16px 20px",
        fontSize: "11px",
        color: "#2d5a48",
        lineHeight: "1.8",
      }}>
        <div style={{ color: "#3d5a48", fontWeight: 600, marginBottom: "6px", letterSpacing: "1px" }}>
          EXPORT FROM YOUR NOTEBOOK:
        </div>
        <code style={{ color: "#5eba8a", fontSize: "10px" }}>
          features = ox.features_from_place(<br />
          &nbsp;&nbsp;&nbsp;&nbsp;"John F. Kennedy International Airport",<br />
          &nbsp;&nbsp;&nbsp;&nbsp;tags={`{"aeroway": True}`}<br />
          )<br />
          features.to_file("jfk.geojson", driver="GeoJSON")
        </code>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN MAP COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
function MapView({ data, filename }) {
  const [visible, setVisible] = useState(new Set(LAYER_ORDER));
  const [showLabels, setShowLabels] = useState(true);
  const [hovered, setHovered] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const [panOrigin, setPanOrigin] = useState({ x: 0, y: 0 });
  const [searchTerm, setSearchTerm] = useState("");
  const [highlighted, setHighlighted] = useState(null);
  const svgRef = useRef(null);

  // ── ATC Command state ──────────────────────────────────────────────────
  const [atcInput, setAtcInput] = useState("");
  const [atcResult, setAtcResult] = useState(null);
  const [atcLoading, setAtcLoading] = useState(false);
  const [atcError, setAtcError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("unknown");

  // A: per-callsign segments — { [callsign]: segment[] }
  const [segmentsByCallsign, setSegmentsByCallsign] = useState({});
  // A: per-callsign state from backend — { [callsign]: aircraft_state }
  const [aircraftStates, setAircraftStates] = useState({});
  // A: callsign → palette index
  const [callsignColorIndex, setCallsignColorIndex] = useState({});
  const nextColorRef = useRef(0);

  // Which callsign is currently shown on the map.
  // Auto-set to the most recently addressed callsign (ATC send or pilot send).
  // User can manually pin a callsign by clicking in the AIRCRAFT panel; clicking
  // again unpins and returns to auto-follow.
  const [activeCallsign, setActiveCallsign] = useState(null);
  const [pinnedCallsign, setPinnedCallsign] = useState(null); // null = auto-follow

  // D: transcript log — flat list of {result, transcript, ts, callsign}
  const [parsedLog, setParsedLog] = useState([]);

  // sidebar tab: "aircraft" | "log"
  const [rightTab, setRightTab] = useState("log");

  const [debugMode, setDebugMode] = useState(false);
  const [csvRows, setCsvRows] = useState([]);        // parsed CSV rows
  const [csvPlayIdx, setCsvPlayIdx] = useState(-1);  // which row is "current" for playback
  const csvFileRef = useRef(null);

  // Resizable right panel
  const [rightPanelWidth, setRightPanelWidth] = useState(280);
  const isResizingRight = useRef(false);
  const resizeStartX = useRef(0);
  const resizeStartW = useRef(0);

  useEffect(() => {
    const onMove = (e) => {
      if (!isResizingRight.current) return;
      const dx = resizeStartX.current - e.clientX; // dragging left = wider
      const newW = Math.max(180, Math.min(600, resizeStartW.current + dx));
      setRightPanelWidth(newW);
    };
    const onUp = () => { isResizingRight.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []);

  // Per-pilot-segment confirmation: { [pilotSegment]: { confirmed, reason, loading } }
  const [readbackResults, setReadbackResults] = useState({});
  // Most recent parsed ATC result per callsign: { [callsign]: { segment, parsed } }
  // When a new ATC is sent for a callsign, this updates and clears old readbacks for it
  const [lastAtcByCallsign, setLastAtcByCallsign] = useState({});
  // Loading state per pilot segment (separate from readbackResults to allow optimistic UI)
  const [pilotSending, setPilotSending] = useState({}); // { [segment]: bool }

  // Flat list of all colored segments for debug overlay and override lookup
  const coloredSegments = useMemo(
    () => Object.values(segmentsByCallsign).flat(),
    [segmentsByCallsign]
  );

  // The callsign currently shown on the map: pinned takes priority, then auto-follow
  const displayedCallsign = pinnedCallsign ?? activeCallsign;

  // Build a lookup: "aeroway:ref:feat_idx" → {color, coords, hex}[]
  // Only the displayed callsign is lit — all others produce no overrides (base map shows).
  const segmentOverrides = useMemo(() => {
    const map = {};
    for (const [callsign, segs] of Object.entries(segmentsByCallsign)) {
      if (callsign !== displayedCallsign) continue; // other callsigns invisible

      for (const seg of segs) {
        const fi = seg.feat_idx ?? 0;
        const k = `${seg.aeroway}:${seg.ref}:${fi}`;
        if (!map[k]) map[k] = [];

        let resolvedHex;
        if (seg.color === "red")         resolvedHex = "#e84545";
        else if (seg.color === "orange") resolvedHex = "#ff9f1c";
        else if (seg.color === "yellow") resolvedHex = "#ffe066";
        else if (seg.color === "purple") resolvedHex = "#c97bff";
        else resolvedHex = null; // grey → base map stroke (original green)

        map[k].push({ color: seg.color, hex: resolvedHex, coords: seg.coords });
      }
    }

    const COLOR_ORDER = { grey: 0, yellow: 1, purple: 2, orange: 3, red: 4 };
    for (const k of Object.keys(map)) {
      map[k].sort((a, b) => (COLOR_ORDER[a.color] ?? 0) - (COLOR_ORDER[b.color] ?? 0));
    }
    return map;
  }, [segmentsByCallsign, displayedCallsign]);

  const BACKEND_URL = "http://localhost:8000";

  const W = 1200;
  const H = 900;

  const features = data.features || [];

  // Per-ref feature index: maps each feature object → its index within its ref group.
  // Must be defined after `features`.
  const featureIndexMap = useMemo(() => {
    const map = new Map();
    const refCounters = {};
    features.forEach((f) => {
      const ref = f.properties?.ref;
      const aeroway = f.properties?.aeroway;
      if (!ref || !aeroway) return;
      const k = `${aeroway}:${ref}`;
      if (refCounters[k] === undefined) refCounters[k] = 0;
      map.set(f, refCounters[k]++);
    });
    return map;
  }, [features]);

  const project = useMemo(
    () => computeProjection(features, W, H),
    [features]
  );

  // Group by aeroway type
  const grouped = useMemo(() => {
    const g = {};
    LAYER_ORDER.forEach((t) => (g[t] = []));
    features.forEach((f) => {
      const t = f.properties?.aeroway;
      if (t && g[t] !== undefined) g[t].push(f);
    });
    return g;
  }, [features]);

  // Counts
  const counts = useMemo(() => {
    const c = {};
    Object.entries(grouped).forEach(([k, v]) => (c[k] = v.length));
    return c;
  }, [grouped]);

  // Search index — exact ref matches rank above partial matches.
  // Shows all individual GeoJSON features (one result per feature) so clicking
  // focuses the exact segment, and debug mode shows each one labeled.
  const searchResults = useMemo(() => {
    if (!searchTerm.trim()) return [];
    const term = searchTerm.toLowerCase();
    const exact = [];
    const partial = [];
    const refCounters = {};
    features.forEach((f) => {
      const ref  = (f.properties?.ref  || "").toLowerCase();
      const name = (f.properties?.name || "").toLowerCase();
      if (!ref.includes(term) && !name.includes(term)) return;
      const refKey = `${f.properties?.aeroway}:${f.properties?.ref}`;
      if (refCounters[refKey] === undefined) refCounters[refKey] = 0;
      const fi = refCounters[refKey]++;
      const entry = {
        feature: f,
        ref: f.properties?.ref,
        name: f.properties?.name,
        type: f.properties?.aeroway,
        featIdx: fi,
      };
      if (ref === term) exact.push(entry);
      else partial.push(entry);
    });
    return [...exact, ...partial].slice(0, 12);
  }, [features, searchTerm]);

  // ── Backend connection ─────────────────────────────────────────────────
  // Check backend status and upload GeoJSON on mount
  useEffect(() => {
    const init = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/`);
        if (res.ok) {
          setBackendStatus("connected");
          // Upload the GeoJSON to backend
          const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
          const formData = new FormData();
          formData.append("file", blob, filename || "airport.geojson");
          const uploadRes = await fetch(`${BACKEND_URL}/load-geojson`, {
            method: "POST",
            body: formData,
          });
          if (uploadRes.ok) {
            const info = await uploadRes.json();
            console.log("Backend loaded:", info);
          }
        }
      } catch {
        setBackendStatus("error");
      }
    };
    init();
  }, [data, filename]);

  // ── CSV transcript loader ──────────────────────────────────────────────
  const parseCsv = (text) => {
    const lines = text.trim().split("\n");
    if (lines.length < 2) return [];
    const header = lines[0].split(",").map(h => h.trim());
    const idx = (name) => header.indexOf(name);
    return lines.slice(1).map((line) => {
      const cols = [];
      let cur = "", inQ = false;
      for (const ch of line) {
        if (ch === '"') { inQ = !inQ; }
        else if (ch === "," && !inQ) { cols.push(cur); cur = ""; }
        else { cur += ch; }
      }
      cols.push(cur);
      return {
        segment:    parseInt(cols[idx("segment")] || "0"),
        start_s:    parseFloat(cols[idx("start_s")] || "0"),
        speaker:    (cols[idx("speaker")] || "").trim(),
        similarity: parseFloat(cols[idx("similarity")] || "0"),
        transcript: (cols[idx("transcript")] || "").trim().replace(/^"|"$/g, ""),
      };
    }).filter(r => r.transcript);
  };

  const loadCsvFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const rows = parseCsv(e.target.result);
      setCsvRows(rows);
      setCsvPlayIdx(-1);
      setReadbackResults({});
      setRightTab("transcript");
    };
    reader.readAsText(file);
  };

  // Parse one ATC row, update map, then check all following pilot readbacks
  // Send an ATC row: parse it, update the map, store as latest ATC for that callsign,
  // and clear any existing readback confirmations for that callsign (new command = new slate).
  const sendCsvRow = async (row) => {
    if (row.speaker !== "ATC") return;
    setAtcInput(row.transcript);
    setCsvPlayIdx(row.segment);
    setAtcLoading(true);
    setAtcError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: row.transcript }),
      });
      const result = await res.json();
      setAtcResult(result);
      const callsign = result?.parsed?.callsign?.toUpperCase().replace(/\s/g, "") || "";
      const newSegs = result.colored_segments || [];
      if (newSegs.length > 0) {
        const newKeys = new Set(newSegs.map(s => `${s.aeroway}:${s.ref}`));
        setSegmentsByCallsign(prev => {
          const prevCallsign = callsign ? (prev[callsign] || []) : [];
          const filtered = prevCallsign.filter(s => !newKeys.has(`${s.aeroway}:${s.ref}`));
          return { ...prev, [callsign]: [...filtered, ...newSegs] };
        });
      }
      if (callsign) {
        setAircraftStates(prev => ({ ...prev, [callsign]: result.aircraft_state || {} }));
        setCallsignColorIndex(prev => {
          if (prev[callsign] !== undefined) return prev;
          const idx = nextColorRef.current++;
          return { ...prev, [callsign]: idx };
        });
      }
      setParsedLog(prev => [...prev, { result, transcript: row.transcript, ts: Date.now(), callsign }]);

      // Auto-follow this callsign on the map (unless user has pinned another)
      if (callsign) setActiveCallsign(callsign);

      // Store as latest ATC for this callsign, clear old readbacks for it
      if (callsign) {
        setLastAtcByCallsign(prev => ({ ...prev, [callsign]: { segment: row.segment, parsed: result.parsed } }));
        // Clear readback results for pilot rows that follow previous ATC commands for this callsign
        setReadbackResults(prev => {
          const next = { ...prev };
          // Remove any pilot-segment readbacks that belonged to this callsign's old ATC
          Object.keys(next).forEach(k => {
            if (next[k]?.callsign === callsign) delete next[k];
          });
          return next;
        });
      }
    } catch (err) {
      setAtcError(err.message);
    } finally {
      setAtcLoading(false);
    }
  };

  // Send a pilot row: check against the most recent ATC command for any known callsign.
  // We don't know which callsign the pilot belongs to from the CSV, so we use the most
  // recently sent ATC result overall (last entry in lastAtcByCallsign by segment number).
  const sendPilotRow = async (row) => {
    if (row.speaker !== "Pilot") return;
    setPilotSending(prev => ({ ...prev, [row.segment]: true }));
    try {
      // Find the most recent ATC result (highest segment number that's still < this row)
      const candidates = Object.values(lastAtcByCallsign)
        .filter(a => a.segment < row.segment)
        .sort((a, b) => b.segment - a.segment);
      const atcEntry = candidates[0];
      if (!atcEntry) {
        setReadbackResults(prev => ({ ...prev, [row.segment]: { confirmed: false, reason: "No ATC command sent yet", loading: false } }));
        return;
      }

      setReadbackResults(prev => ({ ...prev, [row.segment]: { loading: true, callsign: atcEntry.parsed?.callsign } }));

      // Auto-follow the callsign this pilot is responding to
      const respondingTo = atcEntry.parsed?.callsign?.toUpperCase().replace(/\s/g, "");
      if (respondingTo) setActiveCallsign(respondingTo);
      const res = await fetch(`${BACKEND_URL}/check-readback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ atc_parsed: atcEntry.parsed, pilot_transcript: row.transcript }),
      });
      const result = await res.json();
      setReadbackResults(prev => ({
        ...prev,
        [row.segment]: {
          confirmed: result.confirmed,
          reason: result.reason,
          loading: false,
          callsign: atcEntry.parsed?.callsign,
        },
      }));
    } catch {
      setReadbackResults(prev => ({ ...prev, [row.segment]: { confirmed: false, loading: false } }));
    } finally {
      setPilotSending(prev => ({ ...prev, [row.segment]: false }));
    }
  };

  // ── Parse ATC command ──────────────────────────────────────────────────
  const parseAtcCommand = async () => {
    if (!atcInput.trim()) return;
    setAtcLoading(true);
    setAtcError(null);
    setAtcResult(null);

    try {
      const res = await fetch(`${BACKEND_URL}/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: atcInput }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Parse failed");
      }

      const result = await res.json();
      setAtcResult(result);

      const callsign = result?.parsed?.callsign;

      // D: Add to transcript log
      setParsedLog((prev) => [...prev, { result, transcript: atcInput, ts: Date.now(), callsign }]);

      const newSegs = result.colored_segments || [];
      if (callsign && newSegs.length > 0) {
        // A: Assign a palette color index if this is a new callsign
        setCallsignColorIndex((prev) => {
          if (prev[callsign] !== undefined) return prev;
          const idx = nextColorRef.current++;
          return { ...prev, [callsign]: idx };
        });

        // A: Update aircraft state
        if (result.aircraft_state) {
          setAircraftStates((prev) => ({ ...prev, [callsign]: result.aircraft_state }));
        }

        // A: Merge segments for this callsign (replace refs that changed)
        setSegmentsByCallsign((prev) => {
          const existing = prev[callsign] || [];
          const newRefs = new Set(newSegs.map((s) => `${s.aeroway}:${s.ref}`));
          const kept = existing.filter((s) => !newRefs.has(`${s.aeroway}:${s.ref}`));
          return { ...prev, [callsign]: [...kept, ...newSegs] };
        });

        // Auto-follow this callsign
        setActiveCallsign(callsign);
      }
    } catch (err) {
      setAtcError(err.message);
    } finally {
      setAtcLoading(false);
      setAtcInput("");
    }
  };

  // Wheel zoom — zooms toward mouse pointer
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const rect = svgEl.getBoundingClientRect();
    // Mouse position in screen pixels relative to SVG element
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    // Convert screen pixels to SVG viewBox coordinates
    // The viewBox is W x H, but the actual element size is rect.width x rect.height
    // preserveAspectRatio="xMidYMid meet" means uniform scaling with centering
    const svgAspect = W / H;
    const elAspect = rect.width / rect.height;
    let svgScale, offsetX, offsetY;
    if (elAspect > svgAspect) {
      // Element is wider than viewBox — letterboxed horizontally
      svgScale = rect.height / H;
      offsetX = (rect.width - W * svgScale) / 2;
      offsetY = 0;
    } else {
      // Element is taller than viewBox — letterboxed vertically
      svgScale = rect.width / W;
      offsetX = 0;
      offsetY = (rect.height - H * svgScale) / 2;
    }

    // Mouse position in viewBox units
    const vbX = (screenX - offsetX) / svgScale;
    const vbY = (screenY - offsetY) / svgScale;

    const factor = e.deltaY < 0 ? 1.15 : 0.87;

    setZoom((prevZoom) => {
      const newZoom = Math.max(0.2, Math.min(15, prevZoom * factor));
      const ratio = newZoom / prevZoom;

      // Adjust pan so the point under the mouse stays fixed
      setPan((prevPan) => ({
        x: vbX - ratio * (vbX - prevPan.x),
        y: vbY - ratio * (vbY - prevPan.y),
      }));

      return newZoom;
    });
  }, []);

  useEffect(() => {
    const el = svgRef.current;
    if (el) el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el?.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Pan — convert screen deltas to viewBox units
  const getSvgScale = useCallback(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return 1;
    const rect = svgEl.getBoundingClientRect();
    const svgAspect = W / H;
    const elAspect = rect.width / rect.height;
    return elAspect > svgAspect ? rect.height / H : rect.width / W;
  }, []);

  const onDown = (e) => {
    setPanning(true);
    setPanOrigin({ x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y });
  };
  const onMove = (e) => {
    if (!panning) return;
    const s = getSvgScale();
    setPan({
      x: panOrigin.panX + (e.clientX - panOrigin.x) / s,
      y: panOrigin.panY + (e.clientY - panOrigin.y) / s,
    });
  };
  const onUp = () => setPanning(false);

  // Toggle layer
  const toggle = (layer) => {
    setVisible((prev) => {
      const next = new Set(prev);
      next.has(layer) ? next.delete(layer) : next.add(layer);
      return next;
    });
  };

  // Focus on a feature
  const focusFeature = (feature) => {
    const [x, y] = midpoint(feature, project);
    setZoom(4);
    // Center on the feature
    const svgEl = svgRef.current;
    if (svgEl) {
      const rect = svgEl.getBoundingClientRect();
      setPan({
        x: rect.width / 2 - x * 4,
        y: rect.height / 2 - y * 4,
      });
    }
    setHighlighted(feature);
    setTimeout(() => setHighlighted(null), 3000);
  };

  // ── Render feature ─────────────────────────────────────────────────────
  const renderFeature = (f, idx, type) => {
    const meta = LAYER_META[type];
    if (!meta) return null;
    const { geometry: g, properties: p } = f;
    const key = `${type}-${idx}`;
    const isHov = hovered === key;
    const isHl = highlighted === f;

    const handlers = {
      onMouseEnter: () => setHovered(key),
      onMouseLeave: () => setHovered(null),
    };

    // Look up backend color overrides for this specific GeoJSON feature
    const featIdx = featureIndexMap.get(f) ?? 0;
    const overrideKey = `${type}:${p.ref}:${featIdx}`;
    const overrides = (p.ref && segmentOverrides[overrideKey]) || null;
    if (overrides && overrides.some(o => o.color !== "grey")) {
      console.log(`[RENDER] applying override for ${overrideKey}:`, overrides.map(o => o.color));
    }

    // ── Point ──────────────────────────────────────────────────────────
    if (g.type === "Point") {
      const [x, y] = project(g.coordinates[0], g.coordinates[1]);
      const baseR = type === "gate" ? 2 : type === "parking_position" ? 1.5 : 4;
      const r = baseR / Math.max(zoom, 0.5);
      return (
        <g key={key} {...handlers}>
          {isHl && (
            <circle cx={x} cy={y} r={r + 8 / zoom} fill="none" stroke="#ffe066" strokeWidth={2 / zoom} opacity={0.8}>
              <animate attributeName="r" from={r + 4 / zoom} to={r + 14 / zoom} dur="1s" repeatCount="indefinite" />
              <animate attributeName="opacity" from="0.8" to="0" dur="1s" repeatCount="indefinite" />
            </circle>
          )}
          <circle cx={x} cy={y} r={isHov ? r * 1.4 : r}
            fill={meta.fill} opacity={isHov ? 1 : 0.8}
            stroke={isHov ? "#fff" : isHl ? "#ffe066" : "none"} strokeWidth={1 / zoom} />
        </g>
      );
    }

    // ── LineString / MultiLineString ───────────────────────────────────
    if (g.type === "LineString" || g.type === "MultiLineString") {
      const segments = g.type === "MultiLineString" ? g.coordinates : [g.coordinates];
      const meta_w = meta.width;

      // If the backend has sent color overrides for this ref, draw them directly
      // and skip the default rendering entirely.
      if (overrides) {
        return (
          <g key={key} {...handlers}>
            {/* Base shadow for runways */}
            {type === "runway" && segments.map((coords, si) => {
              const d = toPath(coords, project);
              return <path key={`sh-${si}`} d={d} fill="none" stroke="#040a07"
                strokeWidth={meta_w + 6} strokeLinecap="round" opacity={0.7} />;
            })}
            {/* Colored override segments from backend */}
            {overrides.map((ov, oi) => {
              if (ov.coords.length < 2) return null;
              const d = toPath(ov.coords, project);
              const hex = ov.hex || meta.stroke;
              const isGrey = ov.color === "grey";
              const glowOpacity = isGrey ? 0 : ov.color === "red" ? 0.25 : ov.color === "orange" ? 0.2 : ov.color === "purple" ? 0.2 : 0.15;
              return (
                <g key={`ov-${oi}`}>
                  {glowOpacity > 0 && (
                    <path d={d} fill="none" stroke={hex}
                      strokeWidth={meta_w + 10} strokeLinecap="round" opacity={glowOpacity} />
                  )}
                  <path d={d} fill="none"
                    stroke={isGrey ? meta.stroke : hex}
                    strokeWidth={isGrey ? meta_w : meta_w + 2}
                    strokeLinecap="round" strokeLinejoin="round"
                    opacity={isGrey ? 0.85 : 1} />
                </g>
              );
            })}
          </g>
        );
      }

      // Default rendering (no backend override)
      return (
        <g key={key} {...handlers}>
          {segments.map((coords, si) => {
            const d = toPath(coords, project);
            return (
              <g key={`seg-${si}`}>
                {type === "runway" && (
                  <path d={d} fill="none" stroke="#040a07" strokeWidth={meta_w + 6}
                    strokeLinecap="round" opacity={0.7} />
                )}
                {isHl && (
                  <path d={d} fill="none" stroke="#ffe066" strokeWidth={meta_w + 6}
                    strokeLinecap="round" opacity={0.4}>
                    <animate attributeName="opacity" values="0.4;0.1;0.4" dur="1.5s" repeatCount="indefinite" />
                  </path>
                )}
                <path d={d} fill="none"
                  stroke={isHl ? "#ffe066" : isHov ? "#fff" : meta.stroke}
                  strokeWidth={isHov ? meta_w + 1.5 : meta_w}
                  strokeLinecap="round" strokeLinejoin="round"
                  opacity={meta.stroke === "none" ? 0 : 0.85} />
                {type === "runway" && (
                  <path d={d} fill="none" stroke="#6a7880" strokeWidth={1}
                    strokeDasharray="10 7" opacity={0.45} />
                )}
              </g>
            );
          })}
        </g>
      );
    }

    // ── Polygon / MultiPolygon ─────────────────────────────────────────
    if (g.type === "Polygon" || g.type === "MultiPolygon") {
      const rings = g.type === "MultiPolygon"
        ? g.coordinates.flatMap((p) => p)
        : g.coordinates;
      return (
        <g key={key} {...handlers}>
          {rings.map((ring, ri) => {
            const d = toPath([ring], project) + " Z";
            return (
              <path key={ri} d={d}
                fill={meta.fill !== "none" ? meta.fill : "none"}
                stroke={meta.stroke !== "none" ? meta.stroke : "none"}
                strokeWidth={isHov ? meta.width + 1 : meta.width}
                opacity={isHov ? 1 : 0.7} />
            );
          })}
        </g>
      );
    }

    return null;
  };

  // ── Render label ───────────────────────────────────────────────────────
  const renderLabel = (f, idx, type) => {
    const meta = LAYER_META[type];
    if (!meta?.labelColor) return null;
    const label = f.properties?.ref || f.properties?.name;
    if (!label) return null;

    const [x, y] = midpoint(f, project);
    const baseSz = meta.labelSize || 9;
    const sz = baseSz / Math.max(zoom, 0.5);
    const sw = 3 / Math.max(zoom, 0.5);

    const key = `${type}-${idx}`;
    const isHov = hovered === key;

    // Determine dominant color from backend overrides for this specific feature
    const featIdx = featureIndexMap.get(f) ?? 0;
    const overrideKey = `${type}:${label}:${featIdx}`;
    const overrides = segmentOverrides[overrideKey];
    const PRIORITY = { red: 3, orange: 2, yellow: 1, grey: 0 };
    let dominantHex = null;
    if (overrides) {
      let best = -1;
      for (const ov of overrides) {
        const p = PRIORITY[ov.color] ?? 0;
        if (p > best) { best = p; dominantHex = ov.hex || null; }
      }
    }

    const color = isHov ? "#ffffff" : dominantHex || meta.labelColor;

    return (
      <g key={`lbl-${type}-${idx}`}>
        <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
          fill="#060d09" fontSize={sz} fontWeight={type === "runway" ? 800 : 600}
          fontFamily="'IBM Plex Mono', monospace"
          stroke="#060d09" strokeWidth={sw} paintOrder="stroke"
          letterSpacing={type === "runway" ? "2px" : "0.5px"}>
          {label}
        </text>
        <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
          fill={color} fontSize={sz} fontWeight={type === "runway" ? 800 : 600}
          fontFamily="'IBM Plex Mono', monospace"
          letterSpacing={type === "runway" ? "2px" : "0.5px"}>
          {label}
        </text>
      </g>
    );
  };

  const transform = `translate(${pan.x}, ${pan.y}) scale(${zoom})`;

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', monospace",
      background: "#060d09",
      color: "#a0b8a8",
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      userSelect: "none",
    }}>
      {/* ── HEADER ─────────────────────────────────────────────────────── */}
      <div style={{
        padding: "8px 20px",
        borderBottom: "1px solid #1a2e22",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        background: "#080f0b",
        flexShrink: 0,
      }}>
        <div style={{
          width: 28, height: 28,
          background: "#122a1e",
          border: "1px solid #1a3528",
          borderRadius: "5px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "13px", color: "#5eba8a",
        }}>✈</div>
        <div>
          <span style={{ fontSize: "12px", fontWeight: 700, color: "#5eba8a", letterSpacing: "3px" }}>
            ATC SURFACE MAP
          </span>
          <span style={{ fontSize: "9px", color: "#2d5a48", marginLeft: 12 }}>
            {filename} — {features.length} features
          </span>
        </div>

        {/* Search */}
        <div style={{ marginLeft: "auto", position: "relative" }}>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search taxiway, runway..."
            style={{
              width: 220,
              padding: "5px 10px 5px 28px",
              background: "#0a1610",
              border: "1px solid #1a3528",
              borderRadius: "4px",
              color: "#a0b8a8",
              fontSize: "10px",
              fontFamily: "inherit",
              outline: "none",
            }}
          />
          <span style={{
            position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)",
            fontSize: "11px", color: "#2d5a48",
          }}>⌕</span>

          {/* Search results dropdown */}
          {searchResults.length > 0 && searchTerm.trim() && (
            <div style={{
              position: "absolute", top: "100%", left: 0, right: 0,
              marginTop: 4,
              background: "#0a1610",
              border: "1px solid #1a3528",
              borderRadius: "6px",
              maxHeight: 240,
              overflow: "auto",
              zIndex: 100,
            }}>
              {searchResults.map((r, i) => (
                <div key={i}
                  onClick={() => { focusFeature(r.feature); setSearchTerm(""); }}
                  style={{
                    padding: "6px 10px",
                    fontSize: "10px",
                    cursor: "pointer",
                    borderBottom: "1px solid #0e1a13",
                    display: "flex", gap: 8, alignItems: "center",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = "#122a1e"}
                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                >
                  <span style={{
                    color: LAYER_META[r.type]?.labelColor || "#5eba8a",
                    fontWeight: 700, minWidth: 50,
                  }}>{r.ref || "—"}{debugMode ? <span style={{ color: "#ff9f1c", fontWeight: 400 }}>[{r.featIdx}]</span> : null}</span>
                  <span style={{ color: "#3d5a48" }}>{r.type}</span>
                  {r.name && <span style={{ color: "#546e56", marginLeft: "auto" }}>{r.name}</span>}
                </div>
              ))}
            </div>
          )}
        </div>

        <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
          style={{
            padding: "5px 10px", background: "#0a1610",
            border: "1px solid #1a3528", borderRadius: "4px",
            fontSize: "9px", color: "#5eba8a", cursor: "pointer",
            fontFamily: "inherit", letterSpacing: "1px",
          }}>RESET</button>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* ── SIDEBAR ────────────────────────────────────────────────── */}
        <div style={{
          width: 190,
          borderRight: "1px solid #1a2e22",
          padding: "12px",
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          flexShrink: 0,
          overflowY: "auto",
          background: "#070e0a",
        }}>
          <div style={{
            fontSize: "8px", fontWeight: 700, color: "#2d5a48",
            letterSpacing: "2px", marginBottom: "4px",
          }}>LAYERS</div>

          {LAYER_ORDER.map((layer) => {
            const meta = LAYER_META[layer];
            const count = counts[layer] || 0;
            if (count === 0) return null;
            const active = visible.has(layer);
            return (
              <button key={layer} onClick={() => toggle(layer)} style={{
                display: "flex", alignItems: "center", gap: "7px",
                padding: "5px 7px",
                background: active ? "rgba(94,186,138,0.05)" : "transparent",
                border: `1px solid ${active ? "#142e20" : "transparent"}`,
                borderRadius: "3px",
                cursor: "pointer",
                fontFamily: "inherit",
                width: "100%",
              }}>
                <div style={{
                  width: 10, height: 10, borderRadius: "2px",
                  background: meta.fill !== "none" ? meta.fill : meta.stroke,
                  opacity: active ? 1 : 0.25,
                  border: layer === "runway" ? "1px solid #b0bec5" : "none",
                }} />
                <span style={{ fontSize: "9px", color: active ? "#8ab89a" : "#2d4a38", flex: 1, textAlign: "left" }}>
                  {meta.label}
                </span>
                <span style={{ fontSize: "8px", color: "#1e3a2a" }}>{count}</span>
              </button>
            );
          })}

          <div style={{ borderTop: "1px solid #1a2e22", margin: "6px 0" }} />

          <button onClick={() => setShowLabels(!showLabels)} style={{
            display: "flex", alignItems: "center", gap: "7px",
            padding: "5px 7px",
            background: showLabels ? "rgba(94,186,138,0.05)" : "transparent",
            border: `1px solid ${showLabels ? "#142e20" : "transparent"}`,
            borderRadius: "3px",
            cursor: "pointer",
            fontFamily: "inherit",
            width: "100%",
            fontSize: "9px",
            color: showLabels ? "#8ab89a" : "#2d4a38",
          }}>
            <div style={{
              width: 10, height: 10, borderRadius: "2px",
              border: `1px solid ${showLabels ? "#5eba8a" : "#1e3a2a"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "7px", color: "#5eba8a",
            }}>{showLabels ? "✓" : ""}</div>
            Labels
          </button>

          <button onClick={() => setDebugMode(!debugMode)} style={{
            display: "flex", alignItems: "center", gap: "7px",
            padding: "5px 7px",
            background: debugMode ? "rgba(255,159,28,0.08)" : "transparent",
            border: `1px solid ${debugMode ? "#3a2a10" : "transparent"}`,
            borderRadius: "3px",
            cursor: "pointer",
            fontFamily: "inherit",
            width: "100%",
            fontSize: "9px",
            color: debugMode ? "#ff9f1c" : "#2d4a38",
          }}>
            <div style={{
              width: 10, height: 10, borderRadius: "2px",
              border: `1px solid ${debugMode ? "#ff9f1c" : "#1e3a2a"}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "7px", color: "#ff9f1c",
            }}>{debugMode ? "✓" : ""}</div>
            Debug Segments
          </button>

          {/* CSV upload — only visible when debug mode is on */}
          {debugMode && (
            <div style={{ marginTop: 4 }}>
              <input
                ref={csvFileRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={(e) => { if (e.target.files[0]) loadCsvFile(e.target.files[0]); }}
              />
              <button
                onClick={() => csvFileRef.current?.click()}
                style={{
                  display: "flex", alignItems: "center", gap: "7px",
                  padding: "5px 7px",
                  background: csvRows.length > 0 ? "rgba(102,204,255,0.07)" : "transparent",
                  border: `1px solid ${csvRows.length > 0 ? "#1a3a4a" : "transparent"}`,
                  borderRadius: "3px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  width: "100%",
                  fontSize: "9px",
                  color: csvRows.length > 0 ? "#66ccff" : "#2d4a38",
                  textAlign: "left",
                }}>
                <span style={{ fontSize: "11px" }}>⬆</span>
                {csvRows.length > 0 ? `CSV (${csvRows.length} rows)` : "Load CSV Transcript"}
              </button>
            </div>
          )}

          {/* Stats */}
          <div style={{ marginTop: "auto", paddingTop: 10, borderTop: "1px solid #1a2e22" }}>
            <div style={{ fontSize: "8px", color: "#1e3a2a", lineHeight: 1.8, letterSpacing: "0.5px" }}>
              Scroll → zoom<br />
              Drag → pan<br />
              Search → find & focus<br />
              Zoom: {(zoom * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* ── MAP ────────────────────────────────────────────────────── */}
        <div style={{
          flex: 1, overflow: "hidden", position: "relative",
          cursor: panning ? "grabbing" : "grab",
        }}>
          <svg
            ref={svgRef}
            width="100%" height="100%"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="xMidYMid meet"
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerLeave={onUp}
            style={{ display: "block" }}
          >
            <defs>
              <pattern id="g" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#0c1a12" strokeWidth="0.4" />
              </pattern>
            </defs>

            <rect width={W} height={H} fill="#060d09" />
            <rect width={W} height={H} fill="url(#g)" opacity="0.5" />

            <g transform={transform}>
              {/* Features */}
              {LAYER_ORDER.map((type) => {
                if (!visible.has(type)) return null;
                return (
                  <g key={`f-${type}`}>
                    {(grouped[type] || []).map((f, i) => renderFeature(f, i, type))}
                  </g>
                );
              })}

              {/* Labels — pointer-events none so hover passes through to features */}
              {showLabels && LAYER_ORDER.map((type) => {
                if (!visible.has(type)) return null;
                return (
                  <g key={`l-${type}`} style={{ pointerEvents: "none" }}>
                    {(grouped[type] || []).map((f, i) => renderLabel(f, i, type))}
                  </g>
                );
              })}

              {/* Route path overlay removed — taxiways are colored directly */}

              {/* ── DEBUG OVERLAY — geometry of all features ──────────── */}
              {debugMode && (
                <g style={{ pointerEvents: "none" }}>
                  {(() => {
                    const GEOM_COLORS = {
                      runway:   "#ffffff",
                      taxiway:  "#22aadd",
                      taxilane: "#117755",
                    };
                    const GEOM_TYPES = ["runway", "taxiway", "taxilane"];
                    const sw      = 0.8 / Math.max(zoom, 0.3);
                    const dotR    = 2   / Math.max(zoom, 0.3);
                    const fontSize = 5.5 / Math.max(zoom, 0.3);
                    const dash = `${3/zoom} ${2/zoom}`;
                    const refCounters = {};
                    const items = [];

                    for (const type of GEOM_TYPES) {
                      const dc = GEOM_COLORS[type];
                      for (const f of (grouped[type] || [])) {
                        const ref = f.properties?.ref;
                        if (!ref) continue;
                        const g = f.geometry;
                        const segsRaw = g.type === "MultiLineString" ? g.coordinates
                          : g.type === "LineString" ? [g.coordinates] : null;
                        if (!segsRaw) continue;

                        const k = `${type}:${ref}`;
                        if (refCounters[k] === undefined) refCounters[k] = 0;
                        const fi = refCounters[k]++;

                        for (let si = 0; si < segsRaw.length; si++) {
                          const coords = segsRaw[si];
                          if (coords.length < 2) continue;
                          const d = coords.map(([lng, lat], ci) => {
                            const [x, y] = project(lng, lat);
                            return `${ci === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
                          }).join(" ");

                          const mid = coords[Math.floor(coords.length / 2)];
                          const [mx, my] = project(mid[0], mid[1]);
                          const [sx, sy] = project(coords[0][0], coords[0][1]);
                          const [ex, ey] = project(coords[coords.length-1][0], coords[coords.length-1][1]);
                          const label = `${ref}[${fi}]`;

                          items.push(
                            <g key={`gdbg-${type}-${ref}-${fi}-${si}`} opacity={0.75}>
                              <path d={d} fill="none" stroke="#000" strokeWidth={sw * 2.5}
                                strokeLinecap="round" strokeLinejoin="round" />
                              <path d={d} fill="none" stroke={dc} strokeWidth={sw}
                                strokeDasharray={dash}
                                strokeLinecap="round" strokeLinejoin="round" />
                              <circle cx={sx} cy={sy} r={dotR} fill={dc} stroke="#000" strokeWidth={0.3/zoom} />
                              <circle cx={ex} cy={ey} r={dotR} fill="#000" stroke={dc} strokeWidth={sw} />
                              <text x={mx} y={my - fontSize * 1.2} textAnchor="middle"
                                fill="#000" fontSize={fontSize} fontWeight={600}
                                fontFamily="monospace"
                                stroke="#000" strokeWidth={sw * 2} paintOrder="stroke">{label}</text>
                              <text x={mx} y={my - fontSize * 1.2} textAnchor="middle"
                                fill={dc} fontSize={fontSize} fontWeight={600}
                                fontFamily="monospace">{label}</text>
                            </g>
                          );
                        }
                      }
                    }
                    return items;
                  })()}
                </g>
              )}

              {/* ── DEBUG OVERLAY — colored segments ──────────────────── */}
              {debugMode && coloredSegments.length > 0 && (
                <g style={{ pointerEvents: "none" }}>
                  {(() => {
                    // Group segments by ref so we can show per-feature index within each ref
                    const refCounters = {};
                    return coloredSegments.map((seg, si) => {
                      if (!seg.coords || seg.coords.length < 2) return null;
                      const DEBUG_COLORS = {
                        red: "#ff4444", orange: "#ff9f1c",
                        yellow: "#ffe066", purple: "#c97bff", grey: "#556655",
                      };
                      const dc = DEBUG_COLORS[seg.color] || "#ffffff";
                      const fontSize  = 7  / Math.max(zoom, 0.3);
                      const dotR      = 3  / Math.max(zoom, 0.3);
                      const sw        = 1.5 / Math.max(zoom, 0.3);
                      const swOuter   = 3  / Math.max(zoom, 0.3);

                      // Per-ref segment index
                      refCounters[seg.ref] = (refCounters[seg.ref] ?? -1) + 1;
                      const segIdx = refCounters[seg.ref];

                      // SVG path
                      const d = seg.coords.map(([lng, lat], i) => {
                        const [x, y] = project(lng, lat);
                        return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
                      }).join(" ");

                      // Mid-point for label
                      const mid = seg.coords[Math.floor(seg.coords.length / 2)];
                      const [mx, my] = project(mid[0], mid[1]);

                      // Start and end dots
                      const [sx, sy] = project(seg.coords[0][0], seg.coords[0][1]);
                      const [ex, ey] = project(seg.coords[seg.coords.length - 1][0], seg.coords[seg.coords.length - 1][1]);

                      const label = `${seg.ref}[${segIdx}] ${seg.color}`;

                      return (
                        <g key={`dbg-${si}`}>
                          {/* Dashed outline in debug colour */}
                          <path d={d} fill="none" stroke="#000" strokeWidth={swOuter}
                            strokeLinecap="round" strokeLinejoin="round" opacity={0.7} />
                          <path d={d} fill="none" stroke={dc} strokeWidth={sw}
                            strokeLinecap="round" strokeLinejoin="round"
                            strokeDasharray={`${4/zoom} ${3/zoom}`} opacity={0.95} />
                          {/* Start dot (filled) */}
                          <circle cx={sx} cy={sy} r={dotR}
                            fill={dc} stroke="#000" strokeWidth={0.5/zoom} />
                          {/* End dot (hollow) */}
                          <circle cx={ex} cy={ey} r={dotR}
                            fill="#000" stroke={dc} strokeWidth={sw} />
                          {/* Label */}
                          <text x={mx} y={my - fontSize * 1.4} textAnchor="middle"
                            fill="#000" fontSize={fontSize} fontWeight={700}
                            fontFamily="'IBM Plex Mono', monospace"
                            stroke="#000" strokeWidth={swOuter * 0.7} paintOrder="stroke">
                            {label}
                          </text>
                          <text x={mx} y={my - fontSize * 1.4} textAnchor="middle"
                            fill={dc} fontSize={fontSize} fontWeight={700}
                            fontFamily="'IBM Plex Mono', monospace">
                            {label}
                          </text>
                        </g>
                      );
                    });
                  })()}
                </g>
              )}
            </g>
          </svg>

          {/* Tooltip */}
          {hovered && (() => {
            let feat = null;
            for (const t of LAYER_ORDER) {
              const idx = (grouped[t] || []).findIndex((_, i) => `${t}-${i}` === hovered);
              if (idx >= 0) { feat = grouped[t][idx]; break; }
            }
            if (!feat) return null;
            const p = feat.properties;
            return (
              <div style={{
                position: "absolute", bottom: 14, left: "50%",
                transform: "translateX(-50%)",
                background: "rgba(6,13,9,0.93)",
                border: "1px solid #1a3528",
                borderRadius: "5px",
                padding: "6px 14px",
                fontSize: "10px",
                display: "flex", gap: 14,
                pointerEvents: "none",
              }}>
                <span><span style={{ color: "#2d5a48" }}>TYPE </span><span style={{ color: "#5eba8a" }}>{p.aeroway}</span></span>
                {p.ref && <span><span style={{ color: "#2d5a48" }}>REF </span><span style={{ color: "#e8a735", fontWeight: 700 }}>{p.ref}</span></span>}
                {p.name && <span><span style={{ color: "#2d5a48" }}>NAME </span><span style={{ color: "#c4ccd4" }}>{p.name}</span></span>}
                {p.surface && <span><span style={{ color: "#2d5a48" }}>SFC </span><span style={{ color: "#78909c" }}>{p.surface}</span></span>}
              </div>
            );
          })()}
        </div>

        {/* ── RIGHT PANEL — aircraft + log ───────────────────────────── */}
        <div style={{
          width: rightPanelWidth,
          borderLeft: "1px solid #1a2e22",
          background: "#070e0a",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
          overflow: "hidden",
          position: "relative",
        }}>
          {/* Drag handle */}
          <div
            onMouseDown={(e) => {
              isResizingRight.current = true;
              resizeStartX.current = e.clientX;
              resizeStartW.current = rightPanelWidth;
              e.preventDefault();
            }}
            style={{
              position: "absolute", left: 0, top: 0, bottom: 0, width: 4,
              cursor: "ew-resize", zIndex: 10,
              background: "transparent",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "#1a3528"}
            onMouseLeave={e => e.currentTarget.style.background = "transparent"}
          />
          {/* B+D: Tab bar */}
          <div style={{
            display: "flex",
            borderBottom: "1px solid #1a2e22",
            flexShrink: 0,
          }}>
            {[
              { id: "aircraft", label: "AIRCRAFT", count: Object.keys(segmentsByCallsign).length },
              { id: "log",      label: "LOG",      count: parsedLog.length },
              { id: "transcript", label: "SCRIPT", count: csvRows.length },
            ].map(({ id, label, count }) => (
              <button key={id} onClick={() => setRightTab(id)} style={{
                flex: 1,
                padding: "7px 4px",
                background: "none",
                border: "none",
                borderBottom: `2px solid ${rightTab === id ? "#5eba8a" : "transparent"}`,
                color: rightTab === id ? "#5eba8a" : "#2d5a48",
                fontSize: "8px", fontWeight: 700, letterSpacing: "1.5px",
                fontFamily: "inherit",
                cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 5,
              }}>
                {label}
                {count > 0 && (
                  <span style={{
                    background: rightTab === id ? "#1a3528" : "#111e16",
                    color: rightTab === id ? "#5eba8a" : "#2d5a48",
                    borderRadius: 8, padding: "0 5px", fontSize: "8px",
                  }}>{count}</span>
                )}
              </button>
            ))}
          </div>

          {/* B: Aircraft panel */}
          {rightTab === "aircraft" && (
            <div style={{ flex: 1, overflowY: "auto", padding: "6px" }}>
              {Object.keys(segmentsByCallsign).length === 0 ? (
                <div style={{
                  padding: "20px 12px", fontSize: "9px",
                  color: "#1e3a2a", textAlign: "center", lineHeight: 1.8,
                }}>
                  No active aircraft.<br />Parse an ATC command<br />to track aircraft here.
                </div>
              ) : (
                Object.entries(segmentsByCallsign).map(([cs, segs]) => {
                  const state = aircraftStates[cs];
                  const isDisplayed = displayedCallsign === cs;
                  const isPinned = pinnedCallsign === cs;
                  const isActive = activeCallsign === cs && !pinnedCallsign;
                  const currentSegs = segs.filter(s => s.color === "orange");
                  const historySegs = segs.filter(s => s.color === "yellow");

                  // Accent: orange for active/displayed, dim for others
                  const accent = isDisplayed ? "#ff9f1c" : "#2d4a38";

                  return (
                    <div key={cs}
                      onClick={() => setPinnedCallsign(isPinned ? null : cs)}
                      style={{
                        margin: "0 0 6px",
                        padding: "8px 10px",
                        background: isDisplayed ? "rgba(255,159,28,0.06)" : "rgba(255,255,255,0.01)",
                        border: `1px solid ${isDisplayed ? "#3a2a10" : "#111e16"}`,
                        borderLeft: `3px solid ${accent}`,
                        borderRadius: "5px",
                        fontSize: "9px",
                        cursor: "pointer",
                      }}>
                      {/* Header row */}
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          {/* Pin indicator */}
                          <span style={{
                            width: 8, height: 8, borderRadius: isPinned ? "2px" : "50%",
                            background: accent, display: "inline-block", flexShrink: 0,
                          }} title={isPinned ? "Pinned" : isActive ? "Auto-following" : ""} />
                          <span style={{ color: accent, fontWeight: 700, fontSize: "11px" }}>
                            {formatCallsign(cs)}
                          </span>
                          {isPinned && (
                            <span style={{ fontSize: "7px", color: "#ff9f1c88", letterSpacing: "0.5px" }}>PINNED</span>
                          )}
                          {isActive && !isPinned && (
                            <span style={{ fontSize: "7px", color: "#ff9f1c66", letterSpacing: "0.5px" }}>LIVE</span>
                          )}
                        </div>
                        <button
                          onClick={async (e) => {
                            e.stopPropagation();
                            try { await fetch(`${BACKEND_URL}/aircraft-state/${cs}`, { method: "DELETE" }); } catch (_) {}
                            setSegmentsByCallsign(prev => { const n = { ...prev }; delete n[cs]; return n; });
                            setAircraftStates(prev => { const n = { ...prev }; delete n[cs]; return n; });
                            if (pinnedCallsign === cs) setPinnedCallsign(null);
                            if (activeCallsign === cs) setActiveCallsign(null);
                          }}
                          style={{
                            background: "none", border: "none", color: "#2d4a38",
                            cursor: "pointer", fontSize: "14px", lineHeight: 1, padding: "0 2px",
                          }}
                          title="Remove aircraft"
                        >×</button>
                      </div>

                      {/* Badges */}
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 5 }}>
                        {state?.runway && (
                          <span style={{
                            fontSize: "8px", padding: "1px 5px",
                            background: "#200f0f", border: "1px solid #3a1a1a",
                            borderRadius: 3, color: "#e84545",
                          }}>RWY {state.runway}</span>
                        )}
                        {state?.last_taxiway && (
                          <span style={{
                            fontSize: "8px", padding: "1px 5px",
                            background: `${accent}15`, border: `1px solid ${accent}30`,
                            borderRadius: 3, color: accent,
                          }}>at {state.last_taxiway}</span>
                        )}
                      </div>

                      {/* Route chips */}
                      {segs.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 3, alignItems: "center" }}>
                          {segs.map((seg, i) => {
                            const isCurrent = seg.color === "orange";
                            const isBridge  = seg.color === "purple";
                            const isPending = seg.color === "yellow";
                            const chipColor = isCurrent ? "#ff9f1c"
                                            : isBridge  ? "#c97bff"
                                            : isPending ? "#ffe066"
                                            : "#2d5a48";
                            const chipBg    = isCurrent ? "rgba(255,159,28,0.15)"
                                            : isBridge  ? "rgba(201,123,255,0.15)"
                                            : isPending ? "rgba(255,224,102,0.10)"
                                            : "#0e1a13";
                            const chipBorder = isCurrent ? "#ff9f1c44"
                                             : isBridge  ? "#c97bff44"
                                             : isPending ? "#ffe06644"
                                             : "#1a2e22";
                            return (
                              <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 2 }}>
                                <span style={{
                                  padding: "1px 5px", borderRadius: 3,
                                  fontSize: "9px", fontFamily: "inherit",
                                  background: chipBg,
                                  color: chipColor,
                                  border: `1px solid ${chipBorder}`,
                                  fontWeight: isCurrent || isBridge ? 700 : 400,
                                }}>{seg.ref}{isBridge ? " *" : ""}</span>
                                {i < segs.length - 1 && (
                                  <span style={{ color: "#1e3a2a", fontSize: "8px" }}>›</span>
                                )}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}

          {/* D: Log panel */}
          {rightTab === "log" && (
            <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
              {parsedLog.length === 0 ? (
                <div style={{
                  padding: "20px 12px", fontSize: "9px",
                  color: "#1e3a2a", textAlign: "center", lineHeight: 1.8,
                }}>
                  No messages yet.<br />Parse an ATC command<br />to see results here.
                </div>
              ) : (
                [...parsedLog].reverse().map((entry, i) => {
                  const { result, transcript, ts, callsign } = entry;
                  const p = result?.parsed || {};
                  const route = result?.route;
                  const isLatest = i === 0;

                  return (
                    <div key={ts} style={{
                      margin: "0 8px 6px",
                      padding: "8px 10px",
                      background: isLatest ? "rgba(94,186,138,0.04)" : "rgba(255,255,255,0.01)",
                      border: `1px solid ${isLatest ? "#1a3528" : "#111e16"}`,
                      borderLeft: callsign ? "3px solid #ff9f1c44" : "1px solid #111e16",
                      borderRadius: "5px",
                      fontSize: "9px",
                    }}>
                      {/* Timestamp + callsign */}
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span style={{ color: "#ff9f1c", fontWeight: 700, fontSize: "10px" }}>
                          {p.callsign ? formatCallsign(p.callsign) : "—"}
                        </span>
                        <span style={{ color: "#1e3a2a", fontSize: "8px" }}>
                          {formatTime(ts)}
                        </span>
                      </div>

                      {/* Instruction type badge */}
                      {p.instruction_type && (
                        <div style={{ marginBottom: 4 }}>
                          <span style={{
                            fontSize: "7px", fontWeight: 700, letterSpacing: "1px",
                            color: isLatest ? "#5eba8a" : "#2d5a48",
                            background: isLatest ? "rgba(94,186,138,0.1)" : "transparent",
                            padding: "1px 5px", borderRadius: "3px",
                          }}>
                            {p.instruction_type.replace(/_/g, " ").toUpperCase()}
                          </span>
                        </div>
                      )}

                      {/* Runway */}
                      {p.runway && (
                        <div style={{ marginBottom: 3, display: "flex", gap: 6 }}>
                          <span style={{ color: "#2d5a48", minWidth: 42 }}>RWY</span>
                          <span style={{ color: "#e84545", fontWeight: 700 }}>{p.runway}</span>
                        </div>
                      )}

                      {/* Route */}
                      {route?.resolved_route?.length > 0 && (
                        <div style={{ marginBottom: 3, display: "flex", gap: 6, alignItems: "flex-start" }}>
                          <span style={{ color: "#2d5a48", minWidth: 42, flexShrink: 0 }}>ROUTE</span>
                          <span style={{ color: "#ff9f1c", fontWeight: 700, wordBreak: "break-all" }}>
                            {route.resolved_route.join(" → ")}
                          </span>
                        </div>
                      )}

                      {/* Frequency */}
                      {p.frequency_change && (
                        <div style={{ marginBottom: 3, display: "flex", gap: 6 }}>
                          <span style={{ color: "#2d5a48", minWidth: 42 }}>FREQ</span>
                          <span style={{ color: "#66ccff", fontWeight: 700 }}>{p.frequency_change} MHz</span>
                        </div>
                      )}

                      {/* Hold short */}
                      {p.hold_short && (
                        <div style={{ marginBottom: 3, display: "flex", gap: 6 }}>
                          <span style={{ color: "#2d5a48", minWidth: 42 }}>HOLD</span>
                          <span style={{ color: "#cc3333", fontWeight: 700 }}>{p.hold_short}</span>
                        </div>
                      )}

                      {/* Wind */}
                      {p.wind && (
                        <div style={{ marginBottom: 3, display: "flex", gap: 6 }}>
                          <span style={{ color: "#2d5a48", minWidth: 42 }}>WIND</span>
                          <span style={{ color: "#78909c" }}>
                            {p.wind.direction}° {p.wind.speed}kt
                            {p.wind.gust ? ` G${p.wind.gust}` : ""}
                          </span>
                        </div>
                      )}

                      {/* Transcript snippet */}
                      <div style={{
                        marginTop: 5, paddingTop: 5,
                        borderTop: "1px solid #0e1a13",
                        color: "#1e3a2a", fontSize: "8px", lineHeight: 1.5,
                        overflow: "hidden",
                        display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
                      }}>
                        {transcript}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          )}

          {/* TRANSCRIPT panel */}
          {rightTab === "transcript" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Controls bar */}
              {csvRows.length > 0 && (
                <div style={{
                  padding: "5px 10px", borderBottom: "1px solid #111e16",
                  display: "flex", justifyContent: "flex-end", flexShrink: 0,
                }}>
                  <button
                    onClick={() => {
                      setCsvRows([]); setCsvPlayIdx(-1);
                      setReadbackResults({}); setLastAtcByCallsign({});
                    }}
                    style={{
                      padding: "3px 8px", background: "transparent",
                      border: "1px solid #2a1818", borderRadius: "3px",
                      color: "#663333", fontSize: "8px", cursor: "pointer", fontFamily: "inherit",
                    }}>✕ Clear</button>
                </div>
              )}
              <div style={{ flex: 1, overflowY: "auto" }}>
                {csvRows.length === 0 ? (
                  <div style={{
                    padding: "20px 12px", fontSize: "9px",
                    color: "#1e3a2a", textAlign: "center", lineHeight: 1.8,
                  }}>
                    No transcript loaded.<br />Enable Debug Segments<br />then upload a CSV.
                  </div>
                ) : csvRows.map((row) => {
                  const isATC   = row.speaker === "ATC";
                  const isPilot = row.speaker === "Pilot";
                  const isCurrent = row.segment === csvPlayIdx;

                  // Pilot readback: keyed by pilot segment number
                  const rb = isPilot ? readbackResults[row.segment] : null;
                  const confirmed = rb?.confirmed;
                  const rbLoading = rb?.loading;
                  const isPilotSending = pilotSending[row.segment];

                  return (
                    <div key={row.segment} style={{
                      margin: "3px 8px", padding: "6px 8px", borderRadius: "4px",
                      border: `1px solid ${isCurrent ? "#2d5a42" : "#0e1812"}`,
                      background: isCurrent ? "rgba(94,186,138,0.05)" : "transparent",
                      borderLeft: `3px solid ${isATC ? "#ff9f1c55" : isPilot ? "#5eba8a33" : "#111e16"}`,
                    }}>
                      {/* Header */}
                      <div style={{
                        display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 4,
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{
                            fontSize: "7px", fontWeight: 700,
                            padding: "1px 4px", borderRadius: "2px",
                            background: isATC ? "#241a06" : "#0a1e10",
                            color: isATC ? "#ff9f1c" : "#5eba8a",
                            border: `1px solid ${isATC ? "#3a2a10" : "#142e1c"}`,
                          }}>{row.speaker}</span>
                          <span style={{ fontSize: "7px", color: "#1a3020" }}>
                            {Math.floor(row.start_s / 60)}:{String(Math.floor(row.start_s % 60)).padStart(2, "0")}
                          </span>
                        </div>

                        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                          {/* Confirmation checkbox — pilot rows only */}
                          {isPilot && (
                            <div title={rb?.reason || "Send this row to check readback"} style={{
                              width: 14, height: 14, borderRadius: "3px",
                              border: `1px solid ${confirmed ? "#5eba8a" : (rbLoading || isPilotSending) ? "#3d5a48" : "#1e3a2a"}`,
                              background: confirmed ? "#0e2a1a" : "transparent",
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: "9px", color: confirmed ? "#5eba8a" : "#1e3a2a",
                              flexShrink: 0,
                            }}>
                              {(rbLoading || isPilotSending) ? "·" : confirmed ? "✓" : ""}
                            </div>
                          )}

                          {/* SEND button — both ATC and Pilot */}
                          <button
                            onClick={() => isATC ? sendCsvRow(row) : sendPilotRow(row)}
                            disabled={
                              (isATC && (atcLoading || backendStatus !== "connected")) ||
                              (isPilot && (isPilotSending || backendStatus !== "connected"))
                            }
                            style={{
                              padding: "2px 7px",
                              background: isCurrent ? "#142e20" : "#0a1812",
                              border: `1px solid ${isCurrent ? "#2d5a42" : "#1a2e22"}`,
                              borderRadius: "2px",
                              color: isCurrent ? "#5eba8a" : "#2d5a48",
                              fontSize: "7px", fontWeight: 700,
                              cursor: "pointer", fontFamily: "inherit",
                              opacity: backendStatus !== "connected" ? 0.4 : 1,
                            }}>
                            {(isATC && isCurrent && atcLoading) || isPilotSending ? "..." : "SEND"}
                          </button>
                        </div>
                      </div>

                      {/* Transcript text */}
                      <div style={{ fontSize: "8px", lineHeight: 1.5, color: isATC ? "#8a7a5a" : "#4a7a5a" }}>
                        {row.transcript}
                      </div>

                      {/* LLM reason inline below pilot text */}
                      {isPilot && rb?.reason && !rbLoading && !isPilotSending && (
                        <div style={{
                          marginTop: 3, fontSize: "7px", lineHeight: 1.4,
                          color: confirmed ? "#3a6a4a" : "#6a3a3a", fontStyle: "italic",
                        }}>
                          {rb.reason}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── ATC COMMAND PANEL (bottom) ─────────────────────────────────── */}
      <div style={{
        borderTop: "1px solid #1a2e22",
        background: "#070e0a",
        padding: "10px 20px",
        display: "flex",
        gap: "12px",
        alignItems: "flex-start",
        flexShrink: 0,
      }}>
        {/* Input area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "6px" }}>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <span style={{ fontSize: "8px", fontWeight: 700, color: "#2d5a48", letterSpacing: "2px" }}>
              ATC COMMAND
            </span>
            <span style={{
              fontSize: "8px",
              color: backendStatus === "connected" ? "#5eba8a" : "#cc3333",
              display: "flex", alignItems: "center", gap: "4px",
            }}>
              <span style={{
                width: 5, height: 5, borderRadius: "50%",
                background: backendStatus === "connected" ? "#5eba8a" : backendStatus === "error" ? "#cc3333" : "#555",
                display: "inline-block",
              }} />
              {backendStatus === "connected" ? "BACKEND OK" : backendStatus === "error" ? "BACKEND OFFLINE" : "CHECKING..."}
            </span>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              type="text"
              value={atcInput}
              onChange={(e) => setAtcInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && parseAtcCommand()}
              placeholder="e.g. Delta 795 continues, Echo Foxtrot Alpha to the ramp."
              style={{
                flex: 1,
                padding: "7px 12px",
                background: "#0a1610",
                border: "1px solid #1a3528",
                borderRadius: "4px",
                color: "#a0b8a8",
                fontSize: "11px",
                fontFamily: "inherit",
                outline: "none",
              }}
            />
            <button
              onClick={parseAtcCommand}
              disabled={atcLoading || !atcInput.trim() || backendStatus !== "connected"}
              style={{
                padding: "7px 16px",
                background: atcLoading ? "#1a2e22" : "#1a3528",
                border: "1px solid #2d5a42",
                borderRadius: "4px",
                color: atcLoading ? "#3d5a48" : "#5eba8a",
                fontSize: "10px",
                fontFamily: "inherit",
                fontWeight: 700,
                letterSpacing: "1px",
                cursor: atcLoading ? "wait" : "pointer",
                whiteSpace: "nowrap",
                opacity: (!atcInput.trim() || backendStatus !== "connected") ? 0.4 : 1,
              }}
            >
              {atcLoading ? "PARSING..." : "PARSE ▶"}
            </button>
            {(Object.keys(segmentsByCallsign).length > 0) && (
              <button
                onClick={() => {
                  setSegmentsByCallsign({});
                  setAircraftStates({});
                  setCallsignColorIndex({});
                  nextColorRef.current = 0;
                  setParsedLog([]);
                  setAtcResult(null);
                  setSelectedCallsign(null);
                }}
                style={{
                  padding: "7px 12px",
                  background: "transparent",
                  border: "1px solid #3a2020",
                  borderRadius: "4px",
                  color: "#cc5555",
                  fontSize: "10px",
                  fontFamily: "inherit",
                  cursor: "pointer",
                }}
              >
                CLEAR
              </button>
            )}
          </div>
          {atcError && (
            <div style={{ fontSize: "9px", color: "#cc3333" }}>
              Error: {atcError}
            </div>
          )}
        </div>

        {/* Result area */}
        {atcResult && (
          <div style={{
            minWidth: 300, maxWidth: 450,
            background: "#0a1610",
            border: "1px solid #1a3528",
            borderRadius: "6px",
            padding: "8px 12px",
            fontSize: "10px",
            maxHeight: 120,
            overflowY: "auto",
          }}>
            <div style={{ marginBottom: "4px" }}>
              <span style={{ color: "#2d5a48" }}>CALLSIGN </span>
              <span style={{ color: "#ffe066", fontWeight: 700 }}>{atcResult.parsed?.callsign}</span>
            </div>
            {atcResult.parsed?.runway && (
              <div style={{ marginBottom: "4px" }}>
                <span style={{ color: "#2d5a48" }}>RUNWAY </span>
                <span style={{ color: "#e84545", fontWeight: 700 }}>{atcResult.parsed.runway}</span>
              </div>
            )}
            {atcResult.route?.resolved_route?.length > 0 && (
              <div style={{ marginBottom: "4px" }}>
                <span style={{ color: "#2d5a48" }}>ROUTE </span>
                <span style={{ color: "#ff9f1c", fontWeight: 700 }}>
                  {atcResult.route.resolved_route.join(" → ")}
                </span>
                <span style={{ color: "#2d5a48", marginLeft: 8, fontSize: "8px" }}>
                  ({atcResult.route?.method})
                </span>
              </div>
            )}
            {atcResult.parsed?.frequency_change && (
              <div style={{ marginBottom: "4px" }}>
                <span style={{ color: "#2d5a48" }}>FREQ </span>
                <span style={{ color: "#66ccff", fontWeight: 700 }}>{atcResult.parsed.frequency_change} MHz</span>
              </div>
            )}
            {atcResult.parsed?.destination && (
              <div style={{ marginBottom: "4px" }}>
                <span style={{ color: "#2d5a48" }}>DEST </span>
                <span style={{ color: "#5eba8a" }}>{atcResult.parsed.destination}</span>
              </div>
            )}
            {atcResult.parsed?.hold_short && (
              <div style={{ marginBottom: "4px" }}>
                <span style={{ color: "#2d5a48" }}>HOLD SHORT </span>
                <span style={{ color: "#cc3333", fontWeight: 700 }}>{atcResult.parsed.hold_short}</span>
              </div>
            )}
            {atcResult.route?.all_candidates?.length > 1 && (
              <div style={{ marginTop: "4px", paddingTop: "4px", borderTop: "1px solid #142e20" }}>
                <span style={{ color: "#2d5a48", fontSize: "8px" }}>CANDIDATES: </span>
                {atcResult.route.all_candidates.map((c, i) => (
                  <span key={i} style={{
                    color: c.valid ? "#3d8c6e" : "#553333",
                    fontSize: "8px",
                    marginRight: 6,
                  }}>
                    {c.route.join("-")}{c.valid ? "✓" : "✗"}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
//  APP ROOT
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [data, setData] = useState(null);
  const [filename, setFilename] = useState("");

  const handleLoad = (geojson, name) => {
    setData(geojson);
    setFilename(name);
  };

  if (!data) {
    return <DropZone onLoad={handleLoad} />;
  }

  return <MapView data={data} filename={filename} />;
}