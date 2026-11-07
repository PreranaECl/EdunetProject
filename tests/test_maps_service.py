"""
tests/test_maps_service.py — Unit tests for maps_service.py (free OSM/Overpass backend).

Uses mocked HTTP responses — no real API calls made.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import maps_service


# ── Mock response data ────────────────────────────────────────────────────────

NOMINATIM_OK = [
    {
        "lat": "12.9716",
        "lon": "77.5946",
        "display_name": "MG Road, Bengaluru, Karnataka, India",
    }
]

NOMINATIM_EMPTY = []

OVERPASS_OK = {
    "elements": [
        {
            "type": "node",
            "id": 111111,
            "lat": 12.9599,
            "lon": 77.5727,
            "tags": {
                "amenity": "hospital",
                "name": "Victoria Hospital",
                "addr:street": "Fort Road",
                "addr:city": "Bengaluru",
                "phone": "+91 80 2670 1150",
            },
        },
        {
            "type": "way",
            "id": 222222,
            "center": {"lat": 12.9784, "lon": 77.6013},
            "tags": {
                "amenity": "hospital",
                "name": "Bowring Hospital",
                "addr:suburb": "Shivajinagar",
            },
        },
    ]
}

OVERPASS_EMPTY = {"elements": []}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def configure_maps_service():
    """Configure (always-on) free map service before each test."""
    maps_service.configure_maps()
    yield


# ── is_configured ─────────────────────────────────────────────────────────────

def test_is_configured():
    assert maps_service.is_configured() is True


# ── Geocoding tests ───────────────────────────────────────────────────────────

class TestGeocoding:
    @patch("maps_service.requests.get")
    def test_geocode_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = NOMINATIM_OK
        mock_get.return_value = mock_resp

        lat, lon, addr = maps_service.geocode_location("MG Road Bengaluru")
        assert abs(lat - 12.9716) < 0.001
        assert abs(lon - 77.5946) < 0.001
        assert "Bengaluru" in addr

    @patch("maps_service.requests.get")
    def test_geocode_empty_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = NOMINATIM_EMPTY
        mock_get.return_value = mock_resp

        lat, lon, msg = maps_service.geocode_location("zzznowhere xyz")
        assert lat is None
        assert lon is None
        assert "not found" in msg.lower()

    @patch("maps_service.requests.get")
    def test_geocode_timeout(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.Timeout()

        lat, lon, msg = maps_service.geocode_location("somewhere")
        assert lat is None
        assert "timed out" in msg.lower()


# ── Distance calculation ──────────────────────────────────────────────────────

class TestDistanceCalculation:
    def test_same_location_zero(self):
        dist = maps_service.calculate_distance(12.9716, 77.5946, 12.9716, 77.5946)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        dist = maps_service.calculate_distance(12.9716, 77.5946, 12.9599, 77.5727)
        assert 2.0 < dist < 4.0

    def test_large_distance(self):
        dist = maps_service.calculate_distance(0.0, 0.0, 1.0, 1.0)
        assert 100.0 < dist < 200.0


# ── Overpass nearby search ────────────────────────────────────────────────────

class TestNearbySearch:
    @patch("maps_service.requests.post")
    def test_search_returns_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = OVERPASS_OK
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = maps_service.find_nearest_help_centers(12.9716, 77.5946, "Medical Emergency")
        assert result["data_source"] == "OpenStreetMap / Overpass API (free)"
        assert result["status"] == "success"
        assert len(result["help_centers"]) > 0

        first = result["help_centers"][0]
        assert "name" in first
        assert first["distance_km"] is not None

    @patch("maps_service.requests.post")
    def test_search_empty_returns_no_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = OVERPASS_EMPTY
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = maps_service.find_nearest_help_centers(12.9716, 77.5946, "Medical Emergency")
        assert result["status"] == "no_results"
        assert result["help_centers"] == []

    @patch("maps_service.requests.post")
    def test_search_timeout_returns_empty(self, mock_post):
        import requests as req_lib
        mock_post.side_effect = req_lib.exceptions.Timeout()

        # Should not raise — just return empty list
        result = maps_service.find_nearest_help_centers(12.9716, 77.5946, "Fire Emergency")
        assert result["help_centers"] == []

    @patch("maps_service.requests.post")
    def test_results_sorted_by_distance(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = OVERPASS_OK
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = maps_service.find_nearest_help_centers(12.9716, 77.5946, "Medical Emergency")
        centers = [c for c in result["help_centers"] if c["distance_km"] is not None]
        for i in range(len(centers) - 1):
            assert centers[i]["distance_km"] <= centers[i + 1]["distance_km"]

    @patch("maps_service.requests.post")
    def test_way_element_uses_center_coords(self, mock_post):
        """Overpass way elements have coords under 'center', not top-level."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = OVERPASS_OK
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = maps_service.find_nearest_help_centers(12.9716, 77.5946, "Medical Emergency")
        names = [c["name"] for c in result["help_centers"]]
        assert "Bowring Hospital" in names   # way element with center coords


# ── OSM link generation ───────────────────────────────────────────────────────

class TestOSMLinks:
    def test_node_link(self):
        url = maps_service.generate_maps_link(osm_id="111111", osm_type="node")
        assert "openstreetmap.org/node/111111" in url

    def test_way_link(self):
        url = maps_service.generate_maps_link(osm_id="222222", osm_type="way")
        assert "openstreetmap.org/way/222222" in url

    def test_coordinate_link(self):
        url = maps_service.generate_maps_link(latitude=12.97, longitude=77.59)
        assert "openstreetmap.org" in url
        assert "12.97" in url

    def test_fallback_link(self):
        url = maps_service.generate_maps_link()
        assert "openstreetmap.org" in url


# ── Connection validation ─────────────────────────────────────────────────────

class TestConnectionValidation:
    @patch("maps_service.requests.get")
    def test_valid_connection(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = NOMINATIM_OK
        mock_get.return_value = mock_resp

        ok, msg = maps_service.validate_maps_connection()
        assert ok
        assert "OpenStreetMap" in msg or "Nominatim" in msg or "Connected" in msg

    @patch("maps_service.requests.get")
    def test_empty_response_fails(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        ok, msg = maps_service.validate_maps_connection()
        assert not ok

    @patch("maps_service.requests.get")
    def test_network_error_fails(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("no network")

        ok, msg = maps_service.validate_maps_connection()
        assert not ok
        assert "OpenStreetMap" in msg or "network" in msg.lower() or "reach" in msg.lower()


# ── Element helpers ───────────────────────────────────────────────────────────

class TestElementHelpers:
    def test_node_coords(self):
        elem = {"type": "node", "id": 1, "lat": 12.5, "lon": 77.5, "tags": {}}
        lat, lon = maps_service._element_coords(elem)
        assert lat == 12.5
        assert lon == 77.5

    def test_way_coords_from_center(self):
        elem = {"type": "way", "id": 2, "center": {"lat": 13.0, "lon": 78.0}, "tags": {}}
        lat, lon = maps_service._element_coords(elem)
        assert lat == 13.0
        assert lon == 78.0

    def test_element_name_english_preferred(self):
        elem = {"type": "node", "id": 1, "tags": {"name": "ಆಸ್ಪತ್ರೆ", "name:en": "Hospital"}}
        assert maps_service._element_name(elem) == "Hospital"

    def test_element_phone_extracted(self):
        elem = {"type": "node", "id": 1, "tags": {"phone": "+91 80 1234 5678"}}
        assert maps_service._element_phone(elem) == "+91 80 1234 5678"

    def test_element_address_built(self):
        elem = {"type": "node", "id": 1, "tags": {
            "addr:housenumber": "10",
            "addr:street": "Hospital Road",
            "addr:city": "Bengaluru",
        }}
        addr = maps_service._element_address(elem)
        assert "Hospital Road" in addr
        assert "Bengaluru" in addr
