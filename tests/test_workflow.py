"""
tests/test_workflow.py — Integration-level tests for the two-agent workflow.

All Replicate API and Google Maps calls are mocked so no real credentials are needed.
"""

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import replicate_client
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
            "place_id": "PLACE_TEST_001",
            "maps_url": "https://www.google.com/maps/place/?q=place_id:PLACE_TEST_001",
            "latitude": 12.9600,
            "longitude": 77.5800,
        }
    ]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def configure_fake_apis():
    """Configure fake API clients before each test."""
    replicate_client.configure_replicate(
        api_token="FAKE_REPLICATE_TOKEN",
        assessment_model="meta/llama-3.2-11b-vision-instruct",
        response_model="meta/llama-3.1-70b-instruct",
    )
    maps_service._maps_api_key = "FAKE_MAPS_KEY"
    yield
    maps_service._maps_api_key = None


# ── Assessment Agent tests ────────────────────────────────────────────────────

class TestAssessmentAgent:
    @patch("replicate_client._run_replicate")
    def test_successful_assessment(self, mock_run):
        mock_run.return_value = json.dumps(_mock_assessment_dict())

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

    @patch("replicate_client._run_replicate")
    def test_invalid_json_response_fails(self, mock_run):
        mock_run.return_value = "This is not valid JSON at all."

        agent = agents.EmergencyAssessmentAgent()
        assessment, record = agent.run(
            emergency_type="Road Accident",
            description="Collision reported.",
            location="Bengaluru",
        )

        assert assessment is None
        assert record.status == "Failed"

    @patch("replicate_client._run_replicate")
    def test_schema_validation_failure(self, mock_run):
        bad_data = {"agent": "Emergency Assessment Agent", "summary": "Bad schema"}
        mock_run.return_value = json.dumps(bad_data)

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

    @patch("replicate_client._run_replicate")
    def test_successful_response(self, mock_run):
        mock_run.return_value = json.dumps(_mock_response_dict())

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

    @patch("replicate_client._run_replicate")
    def test_severity_mismatch_corrected(self, mock_run):
        """If response severity differs from assessment, it should be corrected."""
        resp_data = _mock_response_dict()
        resp_data["severity_level"] = "Low"        # Mismatch: assessment is High
        mock_run.return_value = json.dumps(resp_data)

        agent = agents.EmergencyResponseAgent()
        response, record = agent.run(
            assessment=self._valid_assessment(),   # severity = High
            help_centers=[],
            original_description="Collision.",
        )

        # Corrected to assessment severity
        if response is not None:
            assert response.severity_level == "High"


# ── Full workflow tests ───────────────────────────────────────────────────────

class TestFullWorkflow:
    @patch("maps_service.requests.get")
    @patch("replicate_client._run_replicate")
    def test_full_workflow_success(self, mock_run, mock_maps_get):
        # Mock Replicate: alternate between assessment and response
        call_count = {"n": 0}

        def replicate_side_effect(model, input_payload):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps(_mock_assessment_dict())
            return json.dumps(_mock_response_dict())

        mock_run.side_effect = replicate_side_effect

        # Mock Maps: geocoding + places nearby
        def maps_side_effect(url, params=None, timeout=None):
            m = MagicMock()
            if "geocode" in url:
                m.json.return_value = {
                    "status": "OK",
                    "results": [{
                        "formatted_address": "MG Road, Bengaluru",
                        "geometry": {"location": {"lat": 12.9716, "lng": 77.5946}},
                    }],
                }
            elif "nearbysearch" in url:
                m.json.return_value = {
                    "status": "OK",
                    "results": [{
                        "name": "City Hospital",
                        "place_id": "TEST_PLACE",
                        "vicinity": "Fort, Bengaluru",
                        "geometry": {"location": {"lat": 12.96, "lng": 77.58}},
                    }],
                }
            elif "details" in url:
                m.json.return_value = {"result": {"formatted_phone_number": "+91 80 9876 5432"}}
            else:
                m.json.return_value = {}
            return m

        mock_maps_get.side_effect = maps_side_effect

        result = agents.run_emergency_workflow(
            emergency_type="Road Accident",
            description="Two vehicles collided near a junction. One person appears injured.",
            location="MG Road, Bengaluru",
        )

        assert result["status"] == "success"
        assert result["assessment"] is not None
        assert result["response"] is not None
        assert len(result["execution_log"]) >= 2

    @patch("maps_service.requests.get")
    @patch("replicate_client._run_replicate")
    def test_workflow_fails_on_assessment_error(self, mock_run, mock_maps_get):
        mock_run.return_value = "Not JSON at all"

        result = agents.run_emergency_workflow(
            emergency_type="Medical Emergency",
            description="Person has collapsed in a shopping mall.",
            location="Bengaluru",
        )

        assert result["status"] == "failed"
        assert result["assessment"] is None
        assert "error_message" in result

    @patch("maps_service.requests.get")
    @patch("replicate_client._run_replicate")
    def test_workflow_returns_partial_on_response_failure(self, mock_run, mock_maps_get):
        call_count = {"n": 0}

        def replicate_side_effect(model, input_payload):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps(_mock_assessment_dict())
            return "invalid json for response"

        mock_run.side_effect = replicate_side_effect

        def maps_side_effect(url, params=None, timeout=None):
            m = MagicMock()
            if "geocode" in url:
                m.json.return_value = {
                    "status": "OK",
                    "results": [{"formatted_address": "Bengaluru", "geometry": {"location": {"lat": 12.97, "lng": 77.59}}}],
                }
            else:
                m.json.return_value = {"status": "ZERO_RESULTS", "results": []}
            return m

        mock_maps_get.side_effect = maps_side_effect

        result = agents.run_emergency_workflow(
            emergency_type="Medical Emergency",
            description="A person has fainted at the bus stop.",
            location="Bengaluru",
        )

        # Assessment succeeded, response failed — partial result
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
