"""
overview_agent.py — Project Overview Agent for Emergency Assist AI.

Answers any question about the project using a built-in knowledge base.
When IBM watsonx.ai (Granite) is configured it uses the LLM for richer
free-form answers. When watsonx is unavailable it uses local keyword
matching against the knowledge base to answer any question accurately.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Tuple

import watsonx_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comprehensive project knowledge base — used for both LLM prompting
# and local keyword-based Q&A
# ---------------------------------------------------------------------------

PROJECT_KNOWLEDGE: dict[str, str] = {
    "purpose": """
**Purpose**
Emergency Assist AI is a free, AI-powered public emergency assistance platform built for India.
It helps citizens during emergencies by:
- Analysing the emergency situation using AI (text + optional photo)
- Assessing severity level (Critical / High / Moderate / Low / Unknown)
- Generating step-by-step safety guidance and immediate actions
- Finding the nearest hospitals, police stations, fire stations, shelters and relief centres
- Providing emergency contact numbers (112 — India's unified emergency number)
- Generating a downloadable emergency assistance report
""",

    "architecture": """
**Architecture — Two-Agent Pipeline**

The core workflow is a sequential two-agent pipeline:

1. **Emergency Assessment Agent** (`agents.py -> EmergencyAssessmentAgent`)
   - Receives: emergency type, description, location, optional image
   - Calls: `gemini_client.analyze_emergency()` -> Google Gemini 2.0 Flash Lite
   - Returns: structured JSON -> severity, risks, recommended services, image quality, info gaps
   - Validated by Pydantic schema before passing to Agent 2

2. **Emergency Response Agent** (`agents.py -> EmergencyResponseAgent`)
   - Receives: validated assessment + nearby help centers from OpenStreetMap
   - Calls: `gemini_client.generate_emergency_guidance()`
   - Returns: structured JSON -> immediate actions, safety warnings, recommended services
   - Validated by Pydantic schema and safety guardrails before display

3. **Project Overview Agent** (`overview_agent.py -> ProjectOverviewAgent`)
   - Answers questions about the project itself
   - Powered by IBM watsonx.ai Granite 3 8B Instruct (or local knowledge base fallback)
   - Triggered by the 📋 Project Overview button on the main page
""",

    "agents": """
**All Agents in This Project**

| Agent | File | Model | Role |
|---|---|---|---|
| Emergency Assessment Agent | `agents.py` | Gemini 2.0 Flash Lite | Analyses emergency, returns severity + risks |
| Emergency Response Agent | `agents.py` | Gemini 2.0 Flash Lite | Generates safety guidance + immediate actions |
| Project Overview Agent | `overview_agent.py` | IBM Granite 3 8B (watsonx.ai) | Answers questions about this project |

**Agent State Machine** (`schemas.py -> AgentState`):
IDLE -> PENDING -> RUNNING -> COMPLETED (or FAILED / BLOCKED / RETRYING)

Each agent run produces an `AgentExecutionRecord` logged with:
- state, status, execution time, validation result, token counts, notes
""",

    "technology": """
**Technology Stack**

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35+ (Python) |
| Primary AI | Google Gemini 2.0 Flash Lite (free, multimodal — text + image) |
| Alternative AI | Groq (`openai/gpt-oss-120b`) · Replicate (Llama models) |
| Overview Agent AI | IBM watsonx.ai — Granite 3 8B Instruct |
| Geocoding | OpenStreetMap Nominatim (free, no key) |
| Nearby search | Overpass API — OSM (free, no key) |
| Map rendering | Folium + streamlit-folium |
| Live location | streamlit-js-eval -> browser `navigator.geolocation` |
| Data validation | Pydantic v2 |
| Image processing | Pillow (resize, EXIF strip) |
| Session memory | SQLite via `memory.py` |
| Data display | Pandas |
| HTTP | requests |
""",

    "data flow": """
**Data Flow — Step by Step**

```
1. User clicks "📍 Use My Live Location"
        -> Browser GPS -> coordinates stored in session state
        -> Live location mini-map appears immediately

2. User fills form: category, description, location, optional image
        -> PII redaction (Aadhaar, PAN, card numbers, email stripped)
        -> Guardrail validation (length, injection check)

3. Emergency Assessment Agent (Gemini 2.0 Flash Lite)
        -> Analyses text + image
        -> Returns JSON: severity, risks, recommended services, image quality
        -> Pydantic schema validation

4. Google Maps / OpenStreetMap nearby search
        -> Nominatim geocodes the location
        -> Overpass API finds hospitals, police, fire stations within 5 km
        -> Results validated and sorted by distance

5. Emergency Response Agent (Gemini 2.0 Flash Lite)
        -> Receives assessment + help centers
        -> Returns JSON: immediate actions, safety warnings, contact
        -> Safety guardrails + Pydantic validation
        -> Severity consistency check between agents

6. Results displayed:
        -> Assessment card (severity badge, risks, indicators)
        -> Response card (immediate actions, safety warnings)
        -> Interactive Folium map (markers, routing lines, layer control)
        -> Help centers table with distances and OSM links
        -> Downloadable .txt emergency report

7. Session metrics updated (tokens, cost, latency, guardrail hits)
        -> Persisted to SQLite for lifetime stats
```
""",

    "guardrails": """
**Guardrails Pipeline** (`guardrails.py`)

Input guardrails (before AI):
- Description length check (max chars enforced)
- Empty field detection
- Prompt injection detection (suspicious override patterns)
- PII redaction — Aadhaar numbers, PAN cards, credit card numbers, emails
- Coordinate range validation (lat −90/+90, lon −180/+180)
- Image metadata stripping (EXIF data removed via Pillow)

Output guardrails (after AI, before display):
- Pydantic schema validation on assessment JSON
- Pydantic schema validation on response JSON
- Cross-agent severity consistency check (response must match assessment)
- Safety content check — blocks dangerous instructions
- Help center data validation (rejects malformed entries)

All guardrail interventions are counted in session metrics.
""",

    "safety": """
**Safety Design**

- Emergency number **112** (India) always visible at top of page
- Banner reminds users to call 112 for immediate danger — never wait for AI
- AI never fabricates emergency contact numbers
- AI never diagnoses medical conditions or prescribes medication
- AI never guarantees emergency service arrival times
- AI never recommends entering burning buildings or confronting criminals
- All AI outputs pass Pydantic schema validation + safety content check
- PII (personal info) is auto-redacted before sending to any AI
- Session rate limiting: 10 requests/minute per session
- Token budget control (configurable in sidebar)
- System prompt injection attacks detected and blocked
""",

    "maps": """
**Map & Location Features**

All map features are completely free — no API key required.

Live location:
- Browser GPS via `navigator.geolocation` (streamlit-js-eval)
- Captured coordinates auto-fill the form fields
- Live mini-map appears immediately after GPS capture
  -> Folium interactive map (Street / Satellite / Dark tiles)
  -> Red ★ marker + accuracy ring at user position
  -> LocateControl button to re-centre on GPS
  -> Fullscreen button
  -> OSM Street View iframe tab

Nearby search (after workflow runs):
- Nominatim geocodes the entered location
- Overpass API searches hospitals, police, fire, shelters within 5 km
- Results shown on interactive Folium map with:
  -> Per-category feature groups (toggleable)
  -> Routing polylines (straight-line, colour-coded by distance)
  -> Distance rings (1 km, 3 km, 5 km)
  -> Popup with name, address, distance, phone, OSM link
  -> Layer control, fullscreen, LocateControl
- Second tab: embedded OSM iframe

Data sources: OpenStreetMap · Nominatim · Overpass API · Esri tiles
""",

    "emergency categories": """
**Supported Emergency Categories**

1. Medical Emergency -> finds Hospitals, Clinics, Doctors
2. Road Accident -> finds Hospitals, Clinics, Police Stations
3. Fire Emergency -> finds Fire Stations, Hospitals
4. Crime or Security Threat -> finds Police Stations
5. Natural Disaster -> finds Relief Centers, Emergency Shelters, Hospitals
6. Flood or Water Emergency -> finds Emergency Shelters, Relief Centers, Hospitals
7. Building Collapse -> finds Fire & Rescue, Hospitals, Police Stations
8. Missing Person -> finds Police Stations
9. Other Emergency -> finds Hospitals, Police Stations, Fire Stations
""",

    "files": """
**Project File Structure**

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — main entry point |
| `agents.py` | Two-agent orchestration (Assessment + Response) |
| `gemini_client.py` | Google Gemini API client (primary AI) |
| `groq_client.py` | Groq API client (alternative AI) |
| `replicate_client.py` | Replicate API client (legacy AI) |
| `watsonx_client.py` | IBM watsonx.ai client (Overview Agent) |
| `overview_agent.py` | Project Overview Agent — this file |
| `maps_service.py` | Free OpenStreetMap geocoding + nearby search |
| `guardrails.py` | Input/output validation + safety checks |
| `schemas.py` | Pydantic models for all data structures |
| `utils.py` | Image processing, PII redaction, metrics, reports |
| `memory.py` | SQLite session persistence |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (API keys, model names) |
""",

    "api keys": """
**API Keys & Configuration**

Required (in `.env` or sidebar):
- `GEMINI_API_KEY` — Google AI Studio key (free: https://aistudio.google.com/app/apikey)

Optional:
- `GROQ_API_KEY` — Groq free key (https://console.groq.com) — alternative AI backend
- `REPLICATE_API_TOKEN` — Replicate token — legacy AI backend
- `WATSONX_API_KEY` — IBM Cloud key for Project Overview Agent
- `WATSONX_PROJECT_ID` — watsonx.ai project ID

Not required (everything uses free OSM services):
- No Google Maps API key needed
- Map data from OpenStreetMap (Nominatim + Overpass API)

Models used:
- Assessment Agent: `gemini-2.0-flash-lite`
- Response Agent: `gemini-2.0-flash-lite`
- Overview Agent: `ibm/granite-3-8b-instruct`
""",

    "memory": """
**Session Memory & Persistence** (`memory.py`)

In-session:
- `SessionMetrics` — token counts, latency, cost estimates, guardrail hits
- `LatencyPoint` history — per-request breakdown for trend charts
- `AgentExecutionRecord` log — per-agent timing, token usage, validation results

Cross-session (SQLite — `.data/emergency_assist_memory.db`):
- Lifetime request count, total tokens, total cost
- Per-request history: type, severity, status, time, tokens, cost, guardrail hits
- Viewable in the 📊 Agent Performance tab -> History sub-tab

Controls:
- "🗑️ Clear session metrics" resets in-session counters
- "🗑️ Clear saved history" wipes the SQLite database
""",

    "performance": """
**Performance Dashboard** (📊 Agent Performance tab)

Sub-tabs:
1. **Overview** — live metrics: requests, success rate, avg response time, tokens, cost
2. **Agent Timeline** — per-request agent execution records with state badges
3. **Latency Trends** — line chart of assessment vs response vs maps latency over time
4. **Token Usage** — bar chart of input vs output tokens per request
5. **History** — table of all past requests from SQLite memory
6. **System State** — raw session state for debugging

Metrics tracked:
- Total / successful / failed workflow count
- Assessment, response, maps latency (seconds)
- Gemini API call count, Maps API call count
- Input tokens, output tokens, estimated cost (USD)
- Guardrail interventions
""",

    "how to run": """
**How to Run the Project**

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Gemini API key (free at https://aistudio.google.com/app/apikey):
   - Edit `.env` -> set `GEMINI_API_KEY=your-key-here`
   - Or paste it directly in the sidebar when the app starts

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Open http://localhost:8501 in your browser

5. In the sidebar: paste your Gemini API key -> "✅ Gemini AI" will appear

6. On the main page: select emergency category, enter location, describe the situation -> click "🔍 Find Emergency Assistance"
""",
}

# ---------------------------------------------------------------------------
# Keyword -> knowledge section mapping
# ---------------------------------------------------------------------------

_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["purpose", "what is", "about", "overview", "summary", "describe project", "explain project"], "purpose"),
    (["architecture", "pipeline", "how does it work", "workflow", "structure"], "architecture"),
    (["agent", "agents", "assessment agent", "response agent", "overview agent"], "agents"),
    (["technology", "stack", "tech", "built with", "framework", "library", "libraries", "tools"], "technology"),
    (["data flow", "flow", "steps", "process", "how it processes", "sequence", "pipeline steps"], "data flow"),
    (["guardrail", "safety check", "validation", "pii", "redaction", "injection", "security"], "guardrails"),
    (["safety", "safe", "emergency number", "112", "disclaimer", "risk", "rate limit"], "safety"),
    (["map", "maps", "location", "gps", "geocod", "nominatim", "overpass", "osm", "openstreetmap", "live location", "nearby"], "maps"),
    (["categor", "emergency type", "medical", "fire", "flood", "accident", "crime", "disaster", "missing", "collapse"], "emergency categories"),
    (["file", "files", "folder", "structure", "module", "modules", "codebase"], "files"),
    (["api key", "key", "credential", "token", "gemini key", "configure", "setup", "env", ".env"], "api keys"),
    (["memory", "sqlite", "database", "persist", "history", "session", "stored", "storage"], "memory"),
    (["performance", "dashboard", "metrics", "latency", "token", "cost", "chart", "stats"], "performance"),
    (["run", "start", "install", "how to", "launch", "execute", "setup", "get started"], "how to run"),
]


def _find_relevant_sections(question: str) -> list[str]:
    """Return knowledge section keys that match the question keywords."""
    q = question.lower()
    matched: list[str] = []
    seen: set[str] = set()
    for keywords, section_key in _KEYWORD_MAP:
        for kw in keywords:
            if kw in q and section_key not in seen:
                matched.append(section_key)
                seen.add(section_key)
                break
    return matched


def _answer_from_knowledge(question: str) -> str:
    """Build a direct answer from the knowledge base matching the question."""
    sections = _find_relevant_sections(question)

    if not sections:
        # No specific match — return full overview
        return _full_overview()

    if len(sections) == 1:
        header = f"### {sections[0].replace('_', ' ').title()}\n\n"
        return header + PROJECT_KNOWLEDGE[sections[0]].strip()

    # Multiple relevant sections
    parts = [f"Here's what's relevant to your question:\n"]
    for key in sections:
        parts.append(PROJECT_KNOWLEDGE[key].strip())
        parts.append("")
    return "\n".join(parts)


def _full_overview() -> str:
    """Return a complete project overview from all knowledge sections."""
    return "\n\n---\n\n".join(
        section.strip()
        for section in [
            PROJECT_KNOWLEDGE["purpose"],
            PROJECT_KNOWLEDGE["architecture"],
            PROJECT_KNOWLEDGE["agents"],
            PROJECT_KNOWLEDGE["technology"],
            PROJECT_KNOWLEDGE["data flow"],
            PROJECT_KNOWLEDGE["guardrails"],
            PROJECT_KNOWLEDGE["safety"],
            PROJECT_KNOWLEDGE["maps"],
        ]
    )


# ---------------------------------------------------------------------------
# Overview Agent
# ---------------------------------------------------------------------------

class ProjectOverviewAgent:
    """
    Answers any question about the Emergency Assist AI project.

    Primary path : IBM watsonx.ai (Granite 3 8B) when configured.
    Fallback path: Local knowledge-base Q&A — always answers questions
                   even when watsonx credentials are not set.
    """

    AGENT_NAME = "Project Overview Agent"

    def generate_overview(self, user_question: str = "") -> Tuple[str, float, bool]:
        """
        Answer a question about this project, or return a full overview.

        Returns (answer_text, latency_seconds, used_watsonx)
        """
        start = time.perf_counter()
        question = user_question.strip()

        # ── Try watsonx (Granite) first ───────────────────────────────────────
        if watsonx_client.is_configured():
            default_q = (
                "Give a comprehensive overview of this project covering: purpose, "
                "agent architecture, technology stack, data flow, safety design, "
                "key files, and how to run it. Use clear markdown sections."
            )
            prompt = (
                "You are the Project Overview Agent for Emergency Assist AI.\n"
                "Answer questions accurately using ONLY the project knowledge below.\n"
                "Be direct, structured, and use markdown formatting.\n\n"
                "=== PROJECT KNOWLEDGE ===\n"
                + "\n\n".join(PROJECT_KNOWLEDGE.values())
                + "\n=== END KNOWLEDGE ===\n\n"
                f"Question: {question or default_q}\n\nAnswer:"
            )
            try:
                text = watsonx_client.generate_text(prompt, max_new_tokens=1024)
                latency = round(time.perf_counter() - start, 2)
                if text.strip():
                    logger.info("ProjectOverviewAgent (watsonx) answered in %.2fs", latency)
                    return text.strip(), latency, True
            except Exception as exc:
                logger.warning("watsonx failed, falling back to local KB: %s", exc)

        # ── Local knowledge-base Q&A (always works) ───────────────────────────
        if question:
            answer = _answer_from_knowledge(question)
        else:
            answer = _full_overview()

        latency = round(time.perf_counter() - start, 3)
        logger.info("ProjectOverviewAgent (local KB) answered in %.3fs", latency)
        return answer, latency, False
