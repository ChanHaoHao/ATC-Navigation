# ATC Airport Surface Map + Navigation Parser

Interactive airport map with ATC command parsing. Parses ATC transcripts,
resolves taxiway routes using geometric intersection validation, and
highlights the navigation path on the map.

## Quick Start

### 1. Backend (Python)

```bash
cd atc-map/backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"

# Option A: auto-load GeoJSON on startup
GEOJSON_PATH=path/to/jfk.geojson python server.py

# Option B: start empty, upload via frontend
python server.py
```

Backend runs at `http://localhost:8000`

### 2. Frontend (React)

```bash
cd atc-map
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

### 3. Use It

1. Drop your `.geojson` file into the frontend (it auto-uploads to backend)
2. Type an ATC command in the bottom panel, e.g.:
   - `Delta 795 continues, Echo Foxtrot Alpha to the ramp.`
   - `United 479, taxi to runway one three left via Echo, Kilo`
3. Hit PARSE — the backend parses it with Claude, resolves the route using
   taxiway intersection geometry, and highlights the path on the map

## Prepare GeoJSON

```python
import osmnx as ox

features = ox.features_from_place(
    "John F. Kennedy International Airport",
    tags={"aeroway": True}
)
features.to_file("jfk_aeroway.geojson", driver="GeoJSON")
```

## How Route Resolution Works

When ATC says "Echo Foxtrot Alpha", the system:

1. Claude parses the phonetic names into letters: `["E", "F", "A"]`
2. The backend generates all possible groupings:
   - `E, F, A` (three separate taxiways)
   - `E, FA` (E + compound taxiway FA)
   - `EF, A` (compound EF + A — if EF exists)
3. For each grouping, checks if consecutive taxiways actually intersect
   in the GeoJSON geometry
4. Returns the grouping where all pairs have valid intersections

## API Endpoints

- `GET /` — health check
- `POST /load-geojson` — upload GeoJSON file
- `POST /parse` — parse ATC transcript (needs ANTHROPIC_API_KEY)
- `POST /resolve-route` — resolve route without Claude (pass letter array)
- `GET /intersections` — view all taxiway intersection pairs
- `GET /taxiways` — view all taxiway refs and their connections

## Project Structure

```
atc-map/
├── backend/
│   ├── server.py           # FastAPI backend
│   └── requirements.txt
├── src/
│   ├── main.jsx
│   └── App.jsx             # React frontend
├── index.html
├── package.json
├── vite.config.js
└── README.md
```
