"""
utils.py — Utility functions for Emergency Assist AI.

Covers:
  - Image processing
  - PII redaction (auto-strip before sending to AI)
  - Report text generation
  - Session metrics management (token aggregation, cost estimation)
  - Rate limiting
  - Token budget alerts
  - Latency history for trending charts
"""

from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from schemas import (
    AgentExecutionRecord,
    EmergencyAssessment,
    EmergencyResponse,
    LatencyPoint,
    SessionMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Replicate per-model cost table
# Prices in USD per million tokens (input / output)
# Source: https://replicate.com/pricing (approximate as of 2025)
# ---------------------------------------------------------------------------

REPLICATE_PRICING: dict[str, dict] = {
    # model_id_fragment → {input_per_mtok, output_per_mtok}
    "llama-3.2-11b":       {"input": 0.06,  "output": 0.06},
    "llama-3.2-90b":       {"input": 0.90,  "output": 0.90},
    "llama-3.1-8b":        {"input": 0.05,  "output": 0.25},
    "llama-3.1-70b":       {"input": 0.65,  "output": 2.75},
    "llama-3.1-405b":      {"input": 9.50,  "output": 9.50},
    "llama-3-8b":          {"input": 0.05,  "output": 0.25},
    "llama-3-70b":         {"input": 0.65,  "output": 2.75},
    "mistral-7b":          {"input": 0.05,  "output": 0.25},
    "mixtral-8x7b":        {"input": 0.30,  "output": 1.00},
    "default":             {"input": 0.10,  "output": 0.30},
}


def estimate_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate USD cost for a single inference call.

    Prices are approximate — Replicate charges per second of GPU compute
    which correlates with token count but is not exactly per-token.
    Always label estimates as approximate in the UI.
    """
    pricing = REPLICATE_PRICING.get("default")
    for fragment, p in REPLICATE_PRICING.items():
        if fragment in model_id.lower():
            pricing = p
            break

    cost = (input_tokens / 1_000_000) * pricing["input"] + \
           (output_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)


# ---------------------------------------------------------------------------
# PII redaction — auto-strip before sending to AI
# ---------------------------------------------------------------------------

# Patterns → replacement labels
_PII_REDACT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),        "[AADHAAR_REDACTED]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                  "[PAN_REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"),                   "[CARD_REDACTED]"),
    (re.compile(r"(?i)\bpassword\s*[:=]\s*\S+"),              "[PASSWORD_REDACTED]"),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),
]


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Replace detected PII patterns with safe placeholder labels.

    Returns (redacted_text, list_of_redacted_types).
    """
    redacted_types: List[str] = []
    result = text
    for pattern, label in _PII_REDACT_RULES:
        new_result, n = pattern.subn(label, result)
        if n > 0:
            redacted_types.append(label)
            result = new_result
    return result, redacted_types


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_REQUESTS_PER_MINUTE = 10   # max requests in a rolling 60-second window


def check_rate_limit(metrics: SessionMetrics) -> Tuple[bool, str]:
    """
    Check whether the session has exceeded the per-minute request rate limit.

    Returns (allowed: bool, message: str).
    Resets the counter if more than 60 seconds have elapsed.
    """
    now = time.time()
    elapsed = now - metrics.last_request_time

    if elapsed >= 60.0:
        # New minute window — reset counter
        metrics.requests_this_minute = 0
        metrics.last_request_time = now

    if metrics.requests_this_minute >= RATE_LIMIT_REQUESTS_PER_MINUTE:
        wait_s = int(60 - elapsed) + 1
        return False, (
            f"Rate limit reached: {RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute. "
            f"Please wait approximately {wait_s} second(s) before retrying."
        )

    # First request of the window — set the start time
    if metrics.requests_this_minute == 0:
        metrics.last_request_time = now

    metrics.requests_this_minute += 1
    return True, ""


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

def check_token_budget(
    metrics: SessionMetrics,
    budget: int,
) -> Tuple[bool, str]:
    """
    Return (within_budget, warning_message).
    If total tokens used this session exceed `budget`, return False.
    """
    used = metrics.input_tokens + metrics.output_tokens
    if budget > 0 and used >= budget:
        return False, (
            f"Token budget exceeded: {used:,} tokens used (limit {budget:,}). "
            "Reset the session or increase the budget in the sidebar to continue."
        )
    if budget > 0 and used >= budget * 0.80:
        return True, (
            f"⚠️ Approaching token budget: {used:,} / {budget:,} tokens used ({int(used/budget*100)}%)."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def prepare_image(
    file_bytes: bytes,
    max_dimension: int = 1600,
    quality: int = 85,
) -> tuple[bytes, str]:
    """
    Resize (if needed) and re-encode an image for upload to Replicate.
    Strips EXIF metadata.

    Returns (image_bytes, mime_type).
    """
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(file_bytes))
    original_format = (img.format or "JPEG").upper()

    w, h = img.size
    if w > max_dimension or h > max_dimension:
        img.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
        logger.info("Image resized from %dx%d to %dx%d", w, h, *img.size)

    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
        output_format = "JPEG"
    elif original_format == "PNG":
        output_format = "PNG"
    else:
        output_format = "JPEG"

    output = io.BytesIO()
    if output_format == "JPEG":
        img.save(output, format="JPEG", quality=quality, exif=b"")
        mime = "image/jpeg"
    else:
        img.save(output, format="PNG")
        mime = "image/png"

    return output.getvalue(), mime


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_emergency_report_text(
    assessment: EmergencyAssessment,
    response: EmergencyResponse,
    help_centers: List[dict],
    location_input: str,
    emergency_type: str,
) -> str:
    """
    Build a plain-text emergency assistance report suitable for download.

    Never includes API keys, raw prompts, or internal configuration data.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "=" * 65,
        "   EMERGENCY ASSIST AI — EMERGENCY ASSISTANCE REPORT",
        "=" * 65,
        f"Generated: {ts}",
        "",
        "IMPORTANT: This report is generated using AI and provides",
        "   preliminary emergency assistance information only.",
        "   It is NOT an official emergency assessment.",
        "   For immediate danger, call 112 in India directly.",
        "=" * 65,
        "",
        f"Emergency Category : {emergency_type}",
        f"Emergency Location : {location_input or 'Not provided'}",
        f"Severity Level     : {assessment.severity_level}",
        "",
        "EMERGENCY SUMMARY",
        "-" * 40,
        assessment.emergency_summary,
        "",
    ]

    if assessment.observed_indicators:
        lines += ["OBSERVED INDICATORS", "-" * 40]
        for item in assessment.observed_indicators:
            lines.append(f"  * {item}")
        lines.append("")

    if assessment.potential_risks:
        lines += ["POTENTIAL RISKS", "-" * 40]
        for item in assessment.potential_risks:
            lines.append(f"  * {item}")
        lines.append("")

    lines += ["IMMEDIATE SAFETY ACTIONS", "-" * 40]
    for action in response.immediate_actions:
        lines.append(f"  > {action}")
    lines.append("")

    lines += ["RECOMMENDED EMERGENCY SERVICES", "-" * 40]
    for svc in response.recommended_services:
        lines.append(f"  * {svc}")
    lines.append("")

    lines += [
        "EMERGENCY CONTACT",
        "-" * 40,
        "  112 -- India Emergency Response Support",
        "  (Integrated number for Police / Fire / Ambulance / Disaster)",
        "",
    ]

    if help_centers:
        lines += ["NEAREST HELP CENTERS (from Google Maps)", "-" * 40]
        for i, c in enumerate(help_centers[:10], 1):
            dist = f"{c['distance_km']} km (straight-line)" if c.get("distance_km") is not None else "distance unknown"
            phone = c.get("phone") or "phone unavailable"
            lines += [
                f"  {i}. {c['name']} [{c['category']}]",
                f"     Address  : {c.get('address', 'N/A')}",
                f"     Distance : {dist}",
                f"     Phone    : {phone}",
                f"     Map      : {c.get('maps_url', 'N/A')}",
                "",
            ]
    else:
        lines += [
            "NEAREST HELP CENTERS",
            "-" * 40,
            "  No verified nearby help centers found.",
            "  Please contact 112 for emergency assistance.",
            "",
        ]

    if response.safety_warnings:
        lines += ["SAFETY WARNINGS", "-" * 40]
        for warning in response.safety_warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    if assessment.information_gaps:
        lines += ["DIAGNOSTIC LIMITATIONS", "-" * 40]
        for gap in assessment.information_gaps:
            lines.append(f"  * {gap}")
        lines.append("")

    lines += [
        "DISCLAIMER",
        "-" * 40,
        response.disclaimer,
        "",
        "=" * 65,
        "  Data source for help centers: Google Maps Places API",
        "  AI platform: Replicate (Llama models)",
        "  Application: Emergency Assist AI",
        "=" * 65,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session metrics helpers
# ---------------------------------------------------------------------------

def init_session_metrics() -> SessionMetrics:
    """Return a fresh SessionMetrics instance."""
    return SessionMetrics()


def update_metrics_from_workflow(
    metrics: SessionMetrics,
    workflow_result: dict,
    emergency_type: str = "",
) -> SessionMetrics:
    """
    Update session metrics after a completed workflow.

    Correctly aggregates structured token counts, estimates cost,
    and appends a LatencyPoint for trending charts.
    Mutates metrics in place and returns it.
    """
    metrics.total_requests += 1

    if workflow_result.get("status") == "success":
        metrics.successful_workflows += 1
    else:
        metrics.failed_workflows += 1

    total_time = workflow_result.get("total_time_s", 0.0)
    metrics.total_response_time_s += total_time

    # Snapshot for the trending chart
    point = LatencyPoint(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_s=round(total_time, 3),
        emergency_type=emergency_type,
    )

    # Per-component breakdown from structured execution records
    for record in workflow_result.get("execution_log", []):
        comp = record.component.lower()
        t = record.execution_time_s or 0.0

        if "assessment" in comp:
            metrics.assessment_latency_s = t
            metrics.gemini_api_calls += 1
            # Structured token fields (Fix 1 — no more string parsing)
            metrics.input_tokens += record.input_tokens
            metrics.output_tokens += record.output_tokens
            point.assessment_s = round(t, 3)
            point.input_tokens += record.input_tokens
            point.output_tokens += record.output_tokens

        elif "response" in comp:
            metrics.response_latency_s = t
            metrics.gemini_api_calls += 1
            metrics.input_tokens += record.input_tokens
            metrics.output_tokens += record.output_tokens
            point.response_s = round(t, 3)
            point.input_tokens += record.input_tokens
            point.output_tokens += record.output_tokens

        elif "maps" in comp or "google" in comp:
            metrics.maps_latency_s = t
            maps_result = workflow_result.get("maps_result") or {}
            metrics.maps_api_calls += maps_result.get("api_calls", 0)
            point.maps_s = round(t, 3)

    # Cost estimation (Fix 2)
    from replicate_client import _assessment_model, _response_model
    workflow_cost = (
        estimate_cost_usd(_assessment_model, point.input_tokens // 2, point.output_tokens // 2)
        + estimate_cost_usd(_response_model, point.input_tokens // 2, point.output_tokens // 2)
    )
    metrics.total_cost_usd = round(metrics.total_cost_usd + workflow_cost, 6)
    point.cost_usd = round(workflow_cost, 6)

    # Append trending point and execution log
    metrics.latency_history.append(point)
    metrics.execution_log.extend(workflow_result.get("execution_log", []))

    return metrics


def record_guardrail_intervention(metrics: SessionMetrics) -> None:
    """Increment the guardrail intervention counter."""
    metrics.guardrail_interventions += 1


def average_response_time(metrics: SessionMetrics) -> float:
    """Return the average workflow time in seconds, or 0.0 if no requests yet."""
    if metrics.total_requests == 0:
        return 0.0
    return round(metrics.total_response_time_s / metrics.total_requests, 2)


# ---------------------------------------------------------------------------
# Severity colour helper
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "Critical": "#dc2626",
    "High":     "#f97316",
    "Moderate": "#eab308",
    "Low":      "#22c55e",
    "Unknown":  "#6b7280",
}


def severity_color(level: str) -> str:
    return SEVERITY_COLORS.get(level, "#6b7280")


# ---------------------------------------------------------------------------
# Category → emoji helper
# ---------------------------------------------------------------------------

CATEGORY_ICONS = {
    "Medical Emergency":        "🏥",
    "Road Accident":            "🚗",
    "Fire Emergency":           "🔥",
    "Crime or Security Threat": "🚔",
    "Natural Disaster":         "⛈️",
    "Flood or Water Emergency": "🌊",
    "Building Collapse":        "🏚️",
    "Missing Person":           "🔍",
    "Other Emergency":          "🆘",
}


def category_icon(category: str) -> str:
    return CATEGORY_ICONS.get(category, "🆘")
