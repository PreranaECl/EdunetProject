"""
maps_service.py — Free map services for Emergency Assist AI.

Uses only FREE, key-free public APIs:
  - Nominatim (OpenStreetMap) for geocoding
  - Overpass API for nearby emergency place search
  - OpenStreetMap links for directions

No Google Maps API key required.

Usage policy:
  - Nominatim: max 1 req/second, must send a descriptive User-Agent.
  - Overpass API: free, public, no authentication.
  - Please respect rate limits to avoid IP bans.
"""

from __future__ import annotations

import logging
import math
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free API endpoints (no keys needed)
# ---------------------------------------------------------------------------

NOMINATIM_URL       = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE   = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL        = "https://overpass-api.de/api/interpreter"

# Respectful User-Agent as required by Nominatim usage policy
_USER_AGENT = "EmergencyAssistAI/1.0 (public emergency assistance tool)"
_HEADERS    = {"User-Agent": _USER_AGENT, "Accept-Language": "en"}

DEFAULT_SEARCH_RADIUS_M = 5000   # 5 km
MAX_RESULTS_PER_CATEGORY = 5

# ---------------------------------------------------------------------------
# Module flag — always "configured" since no key is needed
# ---------------------------------------------------------------------------

_enabled: bool = True


def configure_maps(api_key: str = "") -> None:
    """
    No-op kept for API compatibility.
    Free OSM services need no key — this function always succeeds.
    """
    global _enabled
    _enabled = True
    logger.info("Map service: using free OpenStreetMap / Overpass (no API key required).")


def is_configured() -> bool:
    return _enabled


# ---------------------------------------------------------------------------
# Emergency category → Overpass OSM tag queries
# ---------------------------------------------------------------------------
# Each entry is a list of (amenity_tag, label) pairs searched in sequence.

EMERGENCY_OSM_MAP: Dict[str, List[Dict]] = {
    "Medical Emergency": [
        {"amenity": "hospital",           "label": "Hospital"},
        {"amenity": "clinic",             "label": "Clinic"},
        {"amenity": "doctors",            "label": "Doctor"},
    ],
    "Road Accident": [
        {"amenity": "hospital",           "label": "Hospital"},
        {"amenity": "clinic",             "label": "Clinic"},
        {"amenity": "police",             "label": "Police Station"},
    ],
    "Fire Emergency": [
        {"amenity": "fire_station",       "label": "Fire Station"},
        {"amenity": "hospital",           "label": "Hospital"},
    ],
    "Crime or Security Threat": [
        {"amenity": "police",             "label": "Police Station"},
    ],
    "Natural Disaster": [
        {"amenity": "social_facility",    "label": "Relief Center"},
        {"amenity": "shelter",            "label": "Emergency Shelter"},
        {"amenity": "hospital",           "label": "Hospital"},
    ],
    "Flood or Water Emergency": [
        {"amenity": "shelter",            "label": "Emergency Shelter"},
        {"amenity": "social_facility",    "label": "Relief Center"},
        {"amenity": "hospital",           "label": "Hospital"},
    ],
    "Building Collapse": [
        {"amenity": "fire_station",       "label": "Fire & Rescue"},
        {"amenity": "hospital",           "label": "Hospital"},
        {"amenity": "police",             "label": "Police Station"},
    ],
    "Missing Person": [
        {"amenity": "police",             "label": "Police Station"},
    ],
    "Other Emergency": [
        {"amenity": "hospital",           "label": "Hospital"},
        {"amenity": "police",             "label": "Police Station"},
        {"amenity": "fire_station",       "label": "Fire Station"},
    ],
}

DEFAULT_OSM_SEARCH = [
    {"amenity": "hospital",           "label": "Hospital"},
    {"amenity": "police",             "label": "Police Station"},
    {"amenity": "fire_station",       "label": "Fire Station"},
]


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line (haversine) distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Geocoding — Nominatim (OpenStreetMap)
# ---------------------------------------------------------------------------

def reverse_geocode(lat: float, lon: float) -> str:
    """
    Convert coordinates to a human-readable address using Nominatim reverse geocoding.
    Returns display_name string, or a coordinate string on failure.
    """
    params = {
        "lat":    lat,
        "lon":    lon,
        "format": "json",
        "zoom":   16,
    }
    try:
        resp = requests.get(
            NOMINATIM_REVERSE, params=params,
            headers=_HEADERS, timeout=8,
        )
        data = resp.json()
        return data.get("display_name") or f"{lat:.5f}, {lon:.5f}"
    except Exception:
        return f"{lat:.5f}, {lon:.5f}"


def geocode_location(location: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Convert a place name / address string into (latitude, longitude, display_name).

    Uses Nominatim — free, no key required.
    Returns (None, None, error_message) on failure.
    """
    params = {
        "q":      location,
        "format": "json",
        "limit":  1,
    }
    try:
        start = time.perf_counter()
        resp = requests.get(
            NOMINATIM_URL, params=params,
            headers=_HEADERS, timeout=10,
        )
        elapsed = time.perf_counter() - start
        logger.info("Nominatim geocoding completed in %.2fs", elapsed)

        data = resp.json()
        if not data:
            return None, None, f"Location '{location}' not found. Please enter a more specific address."

        result = data[0]
        lat = float(result["lat"])
        lon = float(result["lon"])
        display = result.get("display_name", location)
        return lat, lon, display

    except requests.exceptions.Timeout:
        return None, None, "Geocoding request timed out. Please check your internet connection."
    except requests.exceptions.RequestException as exc:
        return None, None, f"Geocoding network error: {exc}"
    except (KeyError, ValueError, IndexError) as exc:
        return None, None, f"Unexpected geocoding response: {exc}"


# ---------------------------------------------------------------------------
# Overpass API — nearby place search
# ---------------------------------------------------------------------------

def _overpass_query(lat: float, lon: float, amenity: str, radius_m: int) -> List[dict]:
    """
    Run one Overpass QL query for a given amenity tag within radius_m metres.
    Returns a list of element dicts (may be empty).
    """
    query = f"""
[out:json][timeout:20];
(
  node["amenity"="{amenity}"](around:{radius_m},{lat},{lon});
  way["amenity"="{amenity}"](around:{radius_m},{lat},{lon});
  relation["amenity"="{amenity}"](around:{radius_m},{lat},{lon});
);
out center tags {MAX_RESULTS_PER_CATEGORY * 2};
"""
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=_HEADERS,
            timeout=25,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        return elements
    except requests.exceptions.Timeout:
        logger.warning("Overpass query timed out for amenity=%s", amenity)
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("Overpass query error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Unexpected Overpass error: %s", exc)
        return []


def _element_coords(element: dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract lat/lon from a node or way/relation (which has a 'center' key)."""
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def _element_name(element: dict) -> str:
    tags = element.get("tags", {})
    return (
        tags.get("name:en")
        or tags.get("name")
        or tags.get("official_name")
        or "Unnamed facility"
    )


def _element_address(element: dict) -> str:
    tags = element.get("tags", {})
    parts = []
    for key in ("addr:housenumber", "addr:street", "addr:suburb",
                "addr:city", "addr:state"):
        val = tags.get(key)
        if val:
            parts.append(val)
    return ", ".join(parts) if parts else tags.get("addr:full", "")


def _element_phone(element: dict) -> Optional[str]:
    tags = element.get("tags", {})
    return tags.get("phone") or tags.get("contact:phone") or tags.get("telephone")


def generate_maps_link(
    osm_id: Optional[str] = None,
    osm_type: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    name: Optional[str] = None,
    # kept for backwards-compatibility with old callers
    place_id: Optional[str] = None,
) -> str:
    """
    Generate a free OpenStreetMap URL for a place or coordinate.
    No API key required.
    """
    if osm_id and osm_type:
        # Direct OSM object link
        type_map = {"node": "node", "way": "way", "relation": "relation"}
        t = type_map.get(osm_type, "node")
        return f"https://www.openstreetmap.org/{t}/{osm_id}"
    if latitude is not None and longitude is not None:
        # Zoom to coordinates with a marker
        label = urllib.parse.quote(name or "Emergency Location")
        return (
            f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}"
            f"&zoom=16#map=16/{latitude}/{longitude}"
        )
    return "https://www.openstreetmap.org"


def find_nearest_help_centers(
    latitude: float,
    longitude: float,
    emergency_category: str,
    radius_m: int = DEFAULT_SEARCH_RADIUS_M,
) -> dict:
    """
    Search for real nearby emergency facilities using the free Overpass API.

    Returns a dict matching the HelpCenterSearchResult schema.
    No API key required.
    """
    searches = EMERGENCY_OSM_MAP.get(emergency_category, DEFAULT_OSM_SEARCH)

    seen_ids: set = set()
    centers: list = []
    total_api_calls = 0
    start = time.perf_counter()

    for search_cfg in searches:
        amenity = search_cfg["amenity"]
        label   = search_cfg["label"]

        elements = _overpass_query(latitude, longitude, amenity, radius_m)
        total_api_calls += 1

        for elem in elements[:MAX_RESULTS_PER_CATEGORY]:
            uid = f"{elem.get('type','?')}/{elem.get('id','?')}"
            if uid in seen_ids:
                continue
            seen_ids.add(uid)

            elat, elon = _element_coords(elem)
            distance = None
            if elat is not None and elon is not None:
                distance = round(calculate_distance(latitude, longitude, elat, elon), 2)

            osm_url = generate_maps_link(
                osm_id=str(elem.get("id", "")),
                osm_type=elem.get("type", "node"),
                latitude=elat,
                longitude=elon,
                name=_element_name(elem),
            )

            centers.append({
                "name":        _element_name(elem),
                "category":    label,
                "address":     _element_address(elem),
                "distance_km": distance,
                "phone":       _element_phone(elem),
                "place_id":    uid,
                "maps_url":    osm_url,
                "latitude":    elat,
                "longitude":   elon,
            })

        # Nominatim usage policy: max 1 req/sec
        time.sleep(0.5)

    # Sort by distance (None goes to end)
    centers.sort(key=lambda c: (c["distance_km"] is None, c["distance_km"] or 9999))

    elapsed = time.perf_counter() - start
    logger.info(
        "Free map search: %d results in %.2fs (%d API calls)",
        len(centers), elapsed, total_api_calls,
    )

    return {
        "search_location": {"latitude": latitude, "longitude": longitude},
        "help_centers":    centers,
        "data_source":     "OpenStreetMap / Overpass API (free)",
        "status":          "success" if centers else "no_results",
        "error_message":   None if centers else (
            "No verified nearby help centers found within the search radius. "
            "Try a larger area or call 112 directly."
        ),
        "api_calls":  total_api_calls,
        "latency_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Connection validation — no key, just a lightweight Nominatim ping
# ---------------------------------------------------------------------------

def validate_maps_connection(api_key: str = "") -> Tuple[bool, str]:
    """
    Verify that the free Nominatim service is reachable.
    The api_key parameter is ignored — kept for API compatibility.
    """
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": "Bengaluru India", "format": "json", "limit": 1},
            headers=_HEADERS,
            timeout=8,
        )
        data = resp.json()
        if data:
            return True, "Connected (OpenStreetMap / Nominatim)"
        return False, "Nominatim returned empty results — service may be unavailable."
    except requests.exceptions.RequestException as exc:
        return False, f"Cannot reach OpenStreetMap services: {exc}"
