"""
LLM parsing (Llama 3 70B on HuggingFace)
─────────────────────────────────────────
Builds the airport-specific system prompt, parses ATC transcripts into
structured JSON, and checks pilot readbacks against the issued instruction.

Requires the HF_TOKEN environment variable.
"""

import json
import os

from huggingface_hub import InferenceClient
from fastapi import HTTPException

from state import airport_data

LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _client() -> InferenceClient:
    """Build a HuggingFace inference client, raising 500 if the token is missing."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set. Export your HuggingFace token.")
    return InferenceClient(model=LLM_MODEL, token=hf_token)


def build_system_prompt(valid_refs: list, runway_refs: list) -> str:
    return f"""You are an ATC communication parser for airport ground operations.
Parse the ATC instruction into structured JSON. Return ONLY valid JSON with no markdown, no explanation, no backticks.

Valid taxiway names at this airport: {', '.join(valid_refs)}
Valid runway designators: {', '.join(runway_refs)}

CRITICAL RULES:
1. When phonetic letters are spoken in sequence (e.g. "Zulu Alpha"), output each
   letter separately: ["Z", "A"]. Do NOT try to combine them yourself.
   The backend will check geometric intersections to determine if "ZA" is a
   compound taxiway name or two separate taxiways.
2. "Monitor 1219" means frequency 121.9 MHz
3. "Company" means another aircraft of the same airline
4. Extract the route as individual phonetic-to-letter conversions
5. For "runway", extract just the designator as spoken, e.g. "13L", "04R", "22R".
   Do NOT output the full paired designator — just what was spoken.
   Examples: "runway 13 left" -> "13L", "runway 4 right" -> "04R"
6. For landing clearances, set instruction_type to "landing_clearance" and populate "runway".
7. For taxi instructions after landing, set instruction_type to "taxi_route".

Return this exact JSON structure:
{{
  "callsign": "airline and flight number, e.g. DAL795",
  "controller": "tower|ground|clearance|approach",
  "instruction_type": "landing_clearance|taxi_route|taxi_continue|hold|frequency_change",
  "route_raw": ["each phonetic name as a separate letter, e.g. E, F, A — empty list for landing clearances"],
  "runway": "runway designator as spoken, e.g. 13L or 04R — or null if not mentioned",
  "hold_short": "what to hold short of, or null",
  "cross_runway": "runway to cross, or null",
  "frequency_change": "frequency if mentioned, or null",
  "destination": "ramp|gate number|terminal|null",
  "turn_direction": "left|right|null — ONLY set this when a turn word (left/right) is spoken as a turning instruction that directly precedes the LAST taxiway name in the route (e.g. 'turn left onto Alpha', 'then turn right on Bravo'). Do NOT set this for exit directions like 'exit to the right at Delta' — those describe how to exit the runway, not a turn onto the last taxiway. null if not an explicit turning instruction onto the last taxiway.",
  "wind": {{"direction": number, "speed": number, "gust": number_or_null}} or null,
  "sequence": "sequence number if mentioned, or null",
  "summary": "plain English summary of the full instruction"
}}"""


def parse_atc_with_llm(transcript: str) -> dict:
    """Send transcript to Llama 3 70B on HuggingFace for structured parsing."""
    client = _client()

    valid_refs = sorted(airport_data["valid_refs"])
    runway_refs = sorted(airport_data["runway_geoms"].keys())
    system_prompt = build_system_prompt(valid_refs, runway_refs)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this ATC instruction:\n\n{transcript}"},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    text = response.choices[0].message.content
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def parse_atc_raw(transcript: str) -> dict:
    """
    Debug helper: call the LLM and return its raw output + parsed JSON,
    without running route resolution or touching aircraft state.
    Useful for tuning the prompt.
    """
    client = _client()

    valid_refs = sorted(airport_data["valid_refs"])
    runway_refs = sorted(airport_data["runway_geoms"].keys())
    system_prompt = build_system_prompt(valid_refs, runway_refs)

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this ATC instruction:\n\n{transcript}"},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content
    cleaned = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

    parsed = None
    parse_error = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        parse_error = str(e)

    return {
        "raw_llm_output": raw_text,
        "cleaned_text": cleaned,
        "parsed_json": parsed,
        "parse_error": parse_error,
        "valid_refs_sent": valid_refs,
        "runway_refs_sent": runway_refs,
    }


def check_readback(atc_parsed: dict, pilot_transcript: str) -> dict:
    """
    Use the LLM to check whether a pilot readback correctly acknowledges
    the key elements of the preceding ATC instruction.

    Returns { confirmed: bool, reason: str }
    """
    callsign = atc_parsed.get("callsign", "")
    route = atc_parsed.get("route_raw") or []
    runway = atc_parsed.get("runway") or ""
    summary = atc_parsed.get("summary", "")

    # Build a concise description of what the pilot needed to confirm
    atc_key_elements = []
    if callsign:
        atc_key_elements.append(f"callsign: {callsign}")
    if runway:
        atc_key_elements.append(f"runway: {runway}")
    if route:
        atc_key_elements.append(f"route: {' '.join(route)}")
    if summary:
        atc_key_elements.append(f"instruction summary: {summary}")

    elements_str = " | ".join(atc_key_elements) if atc_key_elements else "(no specific elements)"

    client = _client()

    prompt = f"""You are an ATC readback checker. Determine if a pilot's readback correctly confirms the key elements of the ATC instruction.

ATC instruction key elements:
{elements_str}

Pilot readback:
"{pilot_transcript}"

Rules:
- The pilot must acknowledge the callsign (or close phonetic approximation)
- If a runway was given, the pilot must repeat it
- If a route was given, the pilot must repeat at least the main taxiways
- Minor phonetic errors or abbreviations are acceptable
- If the readback is unrelated noise or a different aircraft, it is NOT confirmed

Respond with ONLY a JSON object, no markdown, no explanation:
{{"confirmed": true/false, "reason": "one short sentence"}}"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    # Strip any markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
        return {
            "confirmed": bool(result.get("confirmed", False)),
            "reason": result.get("reason", ""),
        }
    except Exception:
        # Fallback: check for "true" in raw
        return {
            "confirmed": "true" in raw.lower(),
            "reason": raw[:120],
        }
