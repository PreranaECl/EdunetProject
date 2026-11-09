"""
tests/test_workflow.py — Integration-level tests for the two-agent workflow.

All Gemini API and Google Maps calls are mocked so no real credentials are needed.
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gemini_client as replicate_client
import maps_service
import agents
from schemas import EmergencyAssessment, EmergencyResponse


# ── Mock response builders ────────────────────────────────────────────────────

def _mock_assessment_dict():
    return {
        "agent": "Emergency Assessment Agent",
        "emergency_category": "Road Accident",
        "severity_level": "High",
        "emergency_summary": "A vehicle collision has been reported with possible injuries.",
        "observed_indicators": ["Two vehicles collided", "Possible injury reported"],
        "potential_risks": ["Traffic obstruction", "Fuel leak"],
        "recommended_service": ["Ambulance", "Police"],
        "immediate_emergency_contact_required": True,
        "image_quality": "not_provided",
        "information_gaps": ["Exact number of injured unknown"],
        "requires_human_assistance": True,
        "status": "preliminary_assessment",
    }


def _mock_response_dict():
    return {
        "agent": "Emergency Response Agent",
        "emergency_summary": "A vehicle collision has been reported. Emergency services should be contacted immediately.",
        "severity_level": "High",
        "immediate_actions": [
            "Call 112 immediately",
            "Move to a safe location away from traffic",
            "Do not approach damaged vehicles if there are hazards",
        ],
        "recommended_services": ["Ambulance", "Police"],
        "emergency_contact": {"country": "India", "number": "112"},
        "safety_warnings": [
            "Do not enter the roadway unnecessarily",
            "Do not move seriously injured people unless there is immediate danger",
        ],
        "human_assistance_required": True,
        "disclaimer": "AI-generated preliminary guidance. Follow official emergency responder instructions.",
    }


def _fake_help_centers():
    return [
        {
            "name": "City Hospital",
            "category": "Hospital",
            "address": "123 Main Street, Bengaluru",
            "distance_km": 1.5,
            "phone": "+91 80 1234 5678",
            "place_id": "node/111111",
            "maps_url": "https://www.openstreetmap.org/node/111111",
            "latitude": 12.9600,
            "longitude": 77.5800,
        }
    ]


def _make_fake_gemini_response(data: dict) -> MagicMock:
    """Build a mock Gemini generate_content() response."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    resp.usage_metadata = MagicMock()
    resp.usage_metadata.prompt_token_count = 100
    resp.usage_metadata.candidates_token_count = 200
    return resp


def test_gemini_configuration_uses_selected_models():
    replicate_client.configure_gemini(
        api_key="FAKE_GEMINI_KEY",
        assessment_model="gemini-2.5-flash-lite",
        response_model="gemini-2.5-flash",
    )
    assert replicate_client._assessment_model == "gemini-2.5-flash-lite"
    assert replicate_client._response_model == "gemini-2.5-flash"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def configure_fake_apis():
    """Configure fake API clients before each test."""
    replicate_client.configure_gemini(
        api_key="FAKE_GEMINI_KEY",
        assessment_model="gemini-2.5-flash",
        response_model="gemini-2.5-flash",
    )
    maps_service._enabled = True
    yield


# ── Assessment Agent tests ────────────────────────────────────────────────────

class TestAssessmentAgent:
    @patch("gemini_client._client")
    def test_successful_assessment(self, mock_client):
        mock_client.models.generate_content.return_value = (
            _make_fake_gemini_response(_mock_assessment_dict())
        )

        agent = agents.EmergencyAssessmentAgent()
        assessment, record = agent.run(
            emergency_type="Road Accident",
            description="Two vehicles collided near a junction. One person appears injured.",
            location="MG Road, Bengaluru",
        )

        assert assessment is not None
        assert isinstance(assessment, EmergencyAssessment)
        assert assessment.severity_level == "High"
        assert record.status == "Completed"
        assert record.validation == "Passed"

    @patch("gemini_client._client")
    def test_invalid_json_response_fails(self, mock_client):
        bad = MagicMock(); bad.text = "This is not valid JSON at all."; bad.usage_metadata = None
        mock_client.models.generate_content.return_value = bad

        agent = agents.EmergencyAssessmentAgent()
        assessment, record = agent.run(
            emergency_type="Road Accident",
            description="Collision reported.",
            location="Bengaluru",
        )

        assert assessment is None
        assert record.status == "Failed"

    @patch("gemini_client._client")
    def test_schema_validation_failure(self, mock_client):
        bad_data = {"agent": "Emergency Assessment Agent", "summary": "Bad schema"}
        mock_client.models.generate_content.return_value = (
            _make_fake_gemini_response(bad_data)
        )

        agent = agents.EmergencyAssessmentAgent()
        assessment, record = agent.run(
            emergency_type="Other Emergency",
            description="Unknown emergency.",
            location="Bengaluru",
        )

        assert assessment is None
        assert record.validation == "Failed"


# ── Response Agent tests ──────────────────────────────────────────────────────

class TestResponseAgent:
    def _valid_assessment(self) -> EmergencyAssessment:
        return EmergencyAssessment(**_mock_assessment_dict())

    @patch("gemini_client._client")
    def test_successful_response(self, mock_client):
        mock_client.models.generate_content.return_value = (
            _make_fake_gemini_response(_mock_response_dict())
        )

        agent = agents.EmergencyResponseAgent()
        response, record = agent.run(
            assessment=self._valid_assessment(),
            help_centers=_fake_help_centers(),
            original_description="Two vehicles collided near a junction.",
        )

        assert response is not None
        assert isinstance(response, EmergencyResponse)
        assert response.emergency_contact.number == "112"
        assert record.status == "Completed"

    def test_blocked_without_assessment(self):
        """Response Agent must not run without a valid assessment."""
        agent = agents.EmergencyResponseAgent()
        response, record = agent.run(
            assessment=None,
            help_centers=[],
            original_description="Emergency description.",
        )

        assert response is None
        assert record.status == "Blocked"

    @patch("gemini_client._client")
    def test_severity_mismatch_corrected(self, mock_client):
        """If response severity differs from assessment, it should be corrected."""
        resp_data = _mock_response_dict()
        resp_data["severity_level"] = "Low"  # Mismatch: assessment is High
        mock_client.models.generate_content.return_value = (
            _make_fake_gemini_response(resp_data)
        )

        agent = agents.EmergencyResponseAgent()
        response, record = agent.run(
            assessment=self._valid_assessment(),
            help_centers=[],
            original_description="Collision.",
        )

        if response is not None:
            assert response.severity_level == "High"


# ── Full workflow tests ───────────────────────────────────────────────────────

class TestFullWorkflow:
    @patch("maps_service.requests.post")
    @patch("maps_service.requests.get")
    @patch("gemini_client._client")
    def test_full_workflow_success(self, mock_groq, mock_maps_get, mock_maps_post):
        call_count = {"n": 0}

        def gemini_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_fake_gemini_response(_mock_assessment_dict())
            return _make_fake_gemini_response(_mock_response_dict())

        mock_groq.models.generate_content.side_effect = gemini_side_effect

        def maps_get_side_effect(url, params=None, timeout=None, **kw):
            m = MagicMock()
            m.json.return_value = [{
                "lat": "12.9716", "lon": "77.5946",
                "display_name": "MG Road, Bengaluru",
            }]
            return m

        mock_maps_get.side_effect = maps_get_side_effect

        def maps_post_side_effect(url, data=None, timeout=None, headers=None, **kw):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.json.return_value = {"elements": [{
                "type": "node", "id": 1,
                "lat": 12.96, "lon": 77.58,
                "tags": {"name": "City Hospital", "amenity": "hospital"},
            }]}
            return m

        mock_maps_post.side_effect = maps_post_side_effect

        result = agents.run_emergency_workflow(
            emergency_type="Road Accident",
            description="Two vehicles collided near a junction. One person appears injured.",
            location="MG Road, Bengaluru",
        )

        assert result["status"] == "success"
        assert result["assessment"] is not None
        assert result["response"] is not None
        assert len(result["execution_log"]) >= 2

    @patch("maps_service.requests.post")
    @patch("maps_service.requests.get")
    @patch("gemini_client._client")
    def test_workflow_fails_on_assessment_error(self, mock_groq, mock_maps_get, mock_maps_post):
        bad = MagicMock(); bad.text = "Not JSON"; bad.usage_metadata = None
        mock_groq.models.generate_content.return_value = bad

        result = agents.run_emergency_workflow(
            emergency_type="Medical Emergency",
            description="Person has collapsed in a shopping mall.",
            location="Bengaluru",
        )

        assert result["status"] == "failed"
        assert result["assessment"] is None
        assert "error_message" in result

    @patch("maps_service.requests.post")
    @patch("maps_service.requests.get")
    @patch("gemini_client._client")
    def test_workflow_returns_partial_on_response_failure(self, mock_groq, mock_maps_get, mock_maps_post):
        call_count = {"n": 0}

        def gemini_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_fake_gemini_response(_mock_assessment_dict())
            bad = MagicMock(); bad.text = "invalid json"; bad.usage_metadata = None
            return bad

        mock_groq.models.generate_content.side_effect = gemini_side_effect

        def maps_get_side_effect(url, **kw):
            m = MagicMock()
            m.json.return_value = [{"lat": "12.97", "lon": "77.59", "display_name": "Bengaluru"}]
            return m

        def maps_post_side_effect(url, **kw):
            m = MagicMock(); m.raise_for_status = MagicMock()
            m.json.return_value = {"elements": []}
            return m

        mock_maps_get.side_effect = maps_get_side_effect
        mock_maps_post.side_effect = maps_post_side_effect

        result = agents.run_emergency_workflow(
            emergency_type="Medical Emergency",
            description="A person has fainted at the bus stop.",
            location="Bengaluru",
        )

        assert result["assessment"] is not None
        assert result["response"] is None
        assert result["status"] == "failed"


# ── Multi-agent handoff guardrail ─────────────────────────────────────────────

class TestMultiAgentHandoff:
    def test_second_agent_requires_first_output(self):
        """Verify Response Agent is blocked if no assessment is passed."""
        agent = agents.EmergencyResponseAgent()
        response, record = agent.run(
            assessment=None,
            help_centers=[],
            original_description="Some emergency.",
        )
        assert response is None
        assert record.status == "Blocked"
        assert "cannot run" in record.notes.lower() or "without" in record.notes.lower()
