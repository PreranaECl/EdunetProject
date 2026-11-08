"""
agents.py — Emergency Assessment and Response Agents for Emergency Assist AI.

Orchestrates the two-agent workflow:
  1. Emergency Assessment Agent  → analyzes the situation
  2. Emergency Response Agent    → generates safety guidance

Both agents make genuine Replicate API calls. The second cannot run without
validated output from the first.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Tuple

import groq_client as replicate_client  # drop-in replacement
import guardrails
import maps_service
from schemas import (
    AgentExecutionRecord,
    AgentState,
    EmergencyAssessment,
    EmergencyResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build a plain-text summary of help centers for the prompt
# ---------------------------------------------------------------------------

def _summarise_help_centers(help_centers: list) -> str:
    """Format help center results as plain text for the Response Agent prompt."""
    if not help_centers:
        return "No verified nearby help centers were found."

    lines = []
    for i, c in enumerate(help_centers[:10], 1):
        dist = f"{c['distance_km']} km away" if c.get("distance_km") is not None else "distance unknown"
        phone = c.get("phone") or "phone unavailable"
        lines.append(
            f"{i}. {c['name']} ({c['category']}) — {c.get('address', '')} | {dist} | {phone}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emergency Assessment Agent
# ---------------------------------------------------------------------------

class EmergencyAssessmentAgent:
    """
    Agent 1: Analyzes the emergency description and optional image.
    Returns a validated EmergencyAssessment.
    """

    AGENT_NAME = "Emergency Assessment Agent"

    def run(
        self,
        emergency_type: str,
        description: str,
        location: str,
        image_bytes: Optional[bytes] = None,
        image_mime: str = "image/jpeg",
        max_retries: int = 1,
    ) -> Tuple[Optional[EmergencyAssessment], AgentExecutionRecord]:
        """
        Execute the assessment workflow.

        Returns
        -------
        (EmergencyAssessment | None, AgentExecutionRecord)
        """
        record = AgentExecutionRecord(
            component=self.AGENT_NAME,
            state=AgentState.PENDING,
            status="Pending",
            execution_time_s=0.0,
            validation="Pending",
        )

        start = time.perf_counter()

        for attempt in range(max_retries + 1):
            record.state = AgentState.RUNNING if attempt == 0 else AgentState.RETRYING
            record.status = "Running" if attempt == 0 else f"Retrying (attempt {attempt + 1})"
            try:
                logger.info(
                    "%s — attempt %d/%d", self.AGENT_NAME, attempt + 1, max_retries + 1
                )
                raw, latency, in_tok, out_tok = replicate_client.analyze_emergency(
                    emergency_type=emergency_type,
                    description=description,
                    location=location,
                    image_bytes=image_bytes,
                    image_mime=image_mime,
                )

                # Validate output schema
                assessment, err = guardrails.validate_assessment_output(raw)
                if assessment is None:
                    logger.warning("Assessment validation failed (attempt %d): %s", attempt + 1, err)
                    if attempt < max_retries:
                        continue
                    record.state = AgentState.FAILED
                    record.status = "Failed"
                    record.execution_time_s = round(time.perf_counter() - start, 3)
                    record.validation = "Failed"
                    record.notes = err
                    return None, record

                record.state = AgentState.COMPLETED
                record.status = "Completed"
                record.execution_time_s = round(time.perf_counter() - start, 3)
                record.validation = "Passed"
                record.input_tokens = in_tok
                record.output_tokens = out_tok
                record.notes = f"latency={latency:.2f}s"
                logger.info("%s completed successfully.", self.AGENT_NAME)
                return assessment, record

            except Exception as exc:
                logger.error("%s error (attempt %d): %s", self.AGENT_NAME, attempt + 1, exc)
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                record.state = AgentState.FAILED
                record.status = "Failed"
                record.execution_time_s = round(time.perf_counter() - start, 3)
                record.validation = "Failed"
                record.notes = str(exc)
                return None, record

        # Should not reach here
        record.state = AgentState.FAILED
        record.status = "Failed"
        record.execution_time_s = round(time.perf_counter() - start, 3)
        return None, record


# ---------------------------------------------------------------------------
# Emergency Response Agent
# ---------------------------------------------------------------------------

class EmergencyResponseAgent:
    """
    Agent 2: Generates safety guidance based on the validated assessment
    and real nearby help center data.
    """

    AGENT_NAME = "Emergency Response Agent"

    def run(
        self,
        assessment: EmergencyAssessment,
        help_centers: list,
        original_description: str,
        max_retries: int = 1,
    ) -> Tuple[Optional[EmergencyResponse], AgentExecutionRecord]:
        """
        Execute the response workflow.

        Parameters
        ----------
        assessment : EmergencyAssessment
            Validated output from the Assessment Agent.
        help_centers : list
            Actual results from Google Maps Places API.
        original_description : str
            User's original emergency description.

        Returns
        -------
        (EmergencyResponse | None, AgentExecutionRecord)
        """
        # Guardrail: block if no valid assessment
        guard = guardrails.require_valid_assessment(assessment)
        if not guard:
            record = AgentExecutionRecord(
                component=self.AGENT_NAME,
                state=AgentState.BLOCKED,
                status="Blocked",
                execution_time_s=0.0,
                validation="Failed",
                notes=guard.message,
            )
            return None, record

        record = AgentExecutionRecord(
            component=self.AGENT_NAME,
            state=AgentState.PENDING,
            status="Pending",
            execution_time_s=0.0,
            validation="Pending",
        )

        help_center_summary = _summarise_help_centers(help_centers)
        assessment_dict = assessment.model_dump()

        start = time.perf_counter()

        for attempt in range(max_retries + 1):
            record.state = AgentState.RUNNING if attempt == 0 else AgentState.RETRYING
            record.status = "Running" if attempt == 0 else f"Retrying (attempt {attempt + 1})"
            try:
                logger.info(
                    "%s — attempt %d/%d", self.AGENT_NAME, attempt + 1, max_retries + 1
                )
                raw, latency, in_tok, out_tok = replicate_client.generate_emergency_guidance(
                    assessment=assessment_dict,
                    help_centers_summary=help_center_summary,
                    original_description=original_description,
                )

                # Schema validation
                response, err = guardrails.validate_response_output(raw)
                if response is None:
                    logger.warning("Response validation failed (attempt %d): %s", attempt + 1, err)
                    if attempt < max_retries:
                        continue
                    record.state = AgentState.FAILED
                    record.status = "Failed"
                    record.execution_time_s = round(time.perf_counter() - start, 3)
                    record.validation = "Failed"
                    record.notes = err
                    return None, record

                # Consistency check with assessment
                consistent, consistency_msg = guardrails.check_response_consistency(
                    assessment, response
                )
                if not consistent:
                    logger.warning(
                        "Severity mismatch between agents. Overwriting response severity "
                        "with assessment severity for safety."
                    )
                    response.severity_level = assessment.severity_level

                # Safety content check
                passed, issues = guardrails.run_response_safety_check(response)
                if not passed:
                    logger.warning(
                        "%s safety check found issues (attempt %d): %s",
                        self.AGENT_NAME, attempt + 1, issues,
                    )
                    if attempt < max_retries:
                        continue
                    record.state = AgentState.FAILED
                    record.status = "Failed"
                    record.execution_time_s = round(time.perf_counter() - start, 3)
                    record.validation = "Failed"
                    record.notes = "; ".join(issues)
                    return None, record

                record.state = AgentState.COMPLETED
                record.status = "Completed"
                record.execution_time_s = round(time.perf_counter() - start, 3)
                record.validation = "Passed"
                record.input_tokens = in_tok
                record.output_tokens = out_tok
                record.notes = f"latency={latency:.2f}s"
                logger.info("%s completed successfully.", self.AGENT_NAME)
                return response, record

            except Exception as exc:
                logger.error("%s error (attempt %d): %s", self.AGENT_NAME, attempt + 1, exc)
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                record.status = "Failed"
                record.execution_time_s = round(time.perf_counter() - start, 3)
                record.validation = "Failed"
                record.notes = str(exc)
                return None, record

        record.status = "Failed"
        record.execution_time_s = round(time.perf_counter() - start, 3)
        return None, record


# ---------------------------------------------------------------------------
# Full orchestrated workflow
# ---------------------------------------------------------------------------

def run_emergency_workflow(
    emergency_type: str,
    description: str,
    location: str,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> dict:
    """
    Execute the complete two-agent emergency assistance workflow.

    Workflow
    --------
    1. Emergency Assessment Agent (Gemini)
    2. Google Maps nearby help center search
    3. Emergency Response Agent (Gemini)
    4. Combine results into final report data

    Returns a dict with keys:
      assessment, response, help_centers, execution_log,
      total_time_s, status, error_message
    """
    workflow_start = time.perf_counter()
    execution_log = []
    result = {
        "assessment": None,
        "response": None,
        "help_centers": [],
        "execution_log": execution_log,
        "total_time_s": 0.0,
        "status": "failed",
        "error_message": None,
        "maps_result": None,
    }

    # ------------------------------------------------------------------
    # Step 1: Emergency Assessment Agent
    # ------------------------------------------------------------------
    assessment_agent = EmergencyAssessmentAgent()
    assessment, assessment_record = assessment_agent.run(
        emergency_type=emergency_type,
        description=description,
        location=location,
        image_bytes=image_bytes,
        image_mime=image_mime,
    )
    execution_log.append(assessment_record)

    if assessment is None:
        result["error_message"] = (
            f"Emergency Assessment Agent failed: {assessment_record.notes}"
        )
        result["total_time_s"] = round(time.perf_counter() - workflow_start, 3)
        return result

    result["assessment"] = assessment

    # ------------------------------------------------------------------
    # Step 2: Google Maps — find nearest help centers
    # ------------------------------------------------------------------
    maps_start = time.perf_counter()
    maps_record = AgentExecutionRecord(
        component="Google Maps Service",
        status="Running",
        execution_time_s=0.0,
        validation="N/A",
    )

    help_centers = []
    maps_result = None

    if maps_service.is_configured() and location:
        try:
            lat, lon, formatted_addr = maps_service.geocode_location(location)
            if lat is not None and lon is not None:
                maps_result = maps_service.find_nearest_help_centers(
                    latitude=lat,
                    longitude=lon,
                    emergency_category=emergency_type,
                )
                raw_centers = maps_result.get("help_centers", [])
                help_centers = guardrails.validate_help_center_data(raw_centers)
                maps_result["help_centers"] = help_centers
                maps_record.status = "Completed"
                maps_record.notes = (
                    f"{len(help_centers)} centers found | "
                    f"api_calls={maps_result.get('api_calls', 0)}"
                )
            else:
                maps_record.status = "Failed"
                maps_record.notes = f"Geocoding failed: {formatted_addr}"
        except Exception as exc:
            maps_record.status = "Failed"
            maps_record.notes = str(exc)
            logger.error("Maps service error: %s", exc)
    else:
        maps_record.status = "Skipped"
        maps_record.notes = "Maps not configured or no location provided."

    maps_record.execution_time_s = round(time.perf_counter() - maps_start, 3)
    execution_log.append(maps_record)
    result["help_centers"] = help_centers
    result["maps_result"] = maps_result

    # ------------------------------------------------------------------
    # Step 3: Emergency Response Agent
    # ------------------------------------------------------------------
    response_agent = EmergencyResponseAgent()
    response, response_record = response_agent.run(
        assessment=assessment,
        help_centers=help_centers,
        original_description=description,
    )
    execution_log.append(response_record)

    if response is None:
        result["error_message"] = (
            f"Emergency Response Agent failed: {response_record.notes}"
        )
        result["total_time_s"] = round(time.perf_counter() - workflow_start, 3)
        # Even on response failure, return the assessment so partial results show
        return result

    result["response"] = response
    result["status"] = "success"
    result["total_time_s"] = round(time.perf_counter() - workflow_start, 3)
    logger.info(
        "Emergency workflow completed in %.2fs", result["total_time_s"]
    )
    return result
