# Emergency Assist AI

> AI-Powered Public Emergency Assistance and Nearest Help Center Finder

Emergency Assist AI is a two-agent Streamlit application that uses **Google Gemini** and **Google Maps Platform** to:

1. Analyze emergency situations (text description + optional image)
2. Generate immediate safety guidance
3. Find real nearby emergency help centers (hospitals, police, fire, ambulance, relief centers)
4. Display them on an interactive map
5. Produce a downloadable emergency assistance report

**🚨 If you are in immediate danger, call 112 in India right now. Do not wait for AI analysis.**

---

## Architecture

```
Emergency Input
       │
       ▼
  Input Guardrails (guardrails.py)
       │
       ▼
Emergency Assessment Agent ─────────► Gemini API (gemini_client.py)
       │
       ▼ (validated JSON)
Google Maps Places Search ──────────► Google Maps API (maps_service.py)
       │
       ▼ (real nearby centers)
Emergency Response Agent ───────────► Gemini API
       │
       ▼
  Output Guardrails
       │
       ▼
Results + Map + Downloadable Report
```

---

## Project Structure

```
emergency-assist-ai/
├── app.py              # Streamlit UI and workflow orchestration
├── agents.py           # Two-agent collaboration logic
├── gemini_client.py    # Google Gemini API integration
├── maps_service.py     # Google Maps Platform integration
├── guardrails.py       # Safety guardrails (input, output, data, security)
├── schemas.py          # Pydantic data models
├── utils.py            # Image processing, report generation, metrics
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml.example   # Credential template
└── tests/
    ├── test_guardrails.py
    ├── test_maps_service.py
    └── test_workflow.py
```

---

## Prerequisites

- Python 3.10 or later
- A **Google Gemini API key** — [Get one at Google AI Studio](https://aistudio.google.com/app/apikey)
- A **Google Maps Platform API key** — [Create one in Cloud Console](https://console.cloud.google.com/google/maps-apis)
  - Enable these APIs on your key:
    - **Places API** (Nearby Search)
    - **Geocoding API**
    - **Place Details API** (for phone numbers)

---

## Installation

```bash
# 1. Clone or download the project
cd emergency-assist-ai

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

### Option A — Streamlit secrets (recommended for local use)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIza..."
MAPS_API_KEY   = "AIza..."
```

**Never commit `secrets.toml` to version control.**

### Option B — Environment variables

```bash
export GEMINI_API_KEY="AIza..."
export MAPS_API_KEY="AIza..."
```

### Option C — Sidebar input

Enter both API keys directly in the application sidebar at runtime.

---

## Running the Application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Safety Guardrails

The application implements the following protection layers:

| Layer | Description |
|---|---|
| Input validation | Text length, prompt injection detection, PII warning |
| Image validation | Size limit (5 MB), magic bytes, decompression bomb protection, EXIF stripping |
| Location validation | Coordinate range, address length |
| Output validation | Pydantic schema enforcement, unsafe content scan |
| Agent communication | Assessment must precede response; schema mismatch is blocked |
| Data protection | Images kept in memory; credentials never logged |
| Ethical | AI assessments labelled as preliminary; 112 always visible |

---

## Limitations

- AI severity classifications are **preliminary** and not official emergency dispatch priorities.
- Help center distances are **straight-line** (haversine), not driving distance.
- The application does **not** dispatch emergency services or contact authorities automatically.
- Opening hours and real-time availability are **not** confirmed unless provided by the Places API response.
- Image analysis can identify visible indicators only — it cannot diagnose medical conditions.

---

## Disclaimer

This application provides AI-generated preliminary emergency assistance information.
It is **not** a replacement for official emergency services, professional medical advice, police, fire, or disaster management authorities.
For immediate danger in India, **call 112**.

---

## License

For evaluation and demonstration purposes. Not for production emergency dispatch use.
