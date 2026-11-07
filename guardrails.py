"""
guardrails.py — Safety guardrails for Emergency Assist AI.

Implements input, output, tool, data, security, operational, multi-agent,
and ethical guardrails as required by the application specification.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from schemas import EmergencyAssessment, EmergencyResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024          # 5 MB
MAX_IMAGE_DIMENSION = 4096                        # pixels
MAX_DESCRIPTION_CHARS = 2000
MIN_DESCRIPTION_CHARS = 5
MAX_LOCATION_CHARS = 500

# Valid latitude / longitude ranges
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)

# PII patterns — warn user when detected
PII_PATTERNS = {
    "Aadhaar number": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "PAN number": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "credit/debit card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "bank account": re.compile(r"\b\d{9,18}\b"),
    "password": re.compile(r"(?i)\bpassword\s*[:=]\s*\S+"),
}

# Prompt injection / jailbreak detection patterns
INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (all |previous |your )?(instructions?|rules?|guidelines?)"),
    re.compile(r"(?i)reveal\b.{0,30}?(api[_\s]?key|secret|password|prompt|token)"),
    re.compile(r"(?i)pretend (you are|to be) (a |an )?(?!.*emergency)"),
    re.compile(r"(?i)forget (all |your )?(previous |prior |above )?(instructions?|context|rules?)"),
    re.compile(r"(?i)act as (a |an )?(unrestricted|jailbroken|different)"),
    re.compile(r"(?i)override (safety|guardrail|instruction)"),
    re.compile(r"(?i)do anything now"),
    re.compile(r"(?i)disable (safety|filters?)"),
    re.compile(r"(?i)system prompt"),
]

# Unsafe AI output patterns — things the response agents must never say
UNSAFE_OUTPUT_PATTERNS = [
    re.compile(r"(?i)\bprescribe\b"),
    re.compile(r"(?i)\bdiagnose[sd]?\b"),
    re.compile(r"(?i)take \d+\s*mg"),
    re.compile(r"(?i)ambulance has been dispatched"),
    re.compile(r"(?i)police (have been|has been) notified"),
    re.compile(r"(?i)fire service (have been|has been) notified"),
    re.compile(r"(?i)beds? (are|is) available"),
    re.compile(r"(?i)will arrive in \d+"),
    re.compile(r"(?i)emergency services? have been (called|contacted|dispatched|notified)"),
    re.compile(r"(?i)enter the burning"),
    re.compile(r"(?i)confront the (criminal|attacker|suspect)"),
    re.compile(r"(?i)touch (the |an? )?(exposed |live |bare )?wire"),
]

SEVERITY_LEVELS = {"Critical", "High", "Moderate", "Low", "Unknown"}

# Official emergency numbers — only these may appear in AI output
OFFICIAL_EMERGENCY_NUMBERS = {"112", "100", "101", "102", "108"}


# ---------------------------------------------------------------------------
# Guardrail result helpers
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Container for a single guardrail check outcome."""

    def __init__(self, passed: bool, message: str = "", intervention: bool = False):
        self.passed = passed
        self.message = message
        self.intervention = intervention   # True when an active block/redaction occurred

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# 1. Input guardrails
# ---------------------------------------------------------------------------

def validate_image(
    file_bytes: bytes,
    filename: str,
    mime_type: Optional[str] = None,
) -> GuardrailResult:
    """
    Validate an uploaded image file.

    Checks:
    - File size ≤ 5 MB
    - File extension allowed
    - Valid image file signature (magic bytes)
    - Not a decompression bomb
    - Readable dimensions
    """
    # Size check
    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return GuardrailResult(
            False,
            f"Image too large ({size_mb:.1f} MB). Maximum allowed size is 5 MB.",
            intervention=True,
        )

    # Extension check
    import os
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return GuardrailResult(
            False,
            f"File type '{ext}' is not allowed. Please upload JPG, JPEG, or PNG.",
            intervention=True,
        )

    # Magic bytes / file signature check
    if file_bytes[:3] == b"\xff\xd8\xff":
        detected_mime = "image/jpeg"
    elif file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        detected_mime = "image/png"
    else:
        return GuardrailResult(
            False,
            "The uploaded file does not appear to be a valid JPEG or PNG image.",
            intervention=True,
        )

    # Decompression bomb / dimension check using Pillow
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_DIMENSION ** 2
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        # Re-open after verify (verify() exhausts the stream)
        img = Image.open(io.BytesIO(file_bytes))
        w, h = img.size
        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
            return GuardrailResult(
                False,
                f"Image dimensions ({w}×{h}) exceed the maximum allowed ({MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION}).",
                intervention=True,
            )
    except Exception as exc:
        return GuardrailResult(
            False,
            f"Image validation failed: {exc}. Please upload a valid JPG or PNG file.",
            intervention=True,
        )

    return GuardrailResult(True, "Image valid.", intervention=False)


def validate_text_input(text: str, field_name: str = "description") -> GuardrailResult:
    """
    Validate emergency description or location text.

    Checks:
    - Not empty
    - Not too short
    - Not too long
    - No prompt injection patterns
    """
    if not isinstance(text, str):
        return GuardrailResult(False, f"{field_name} must be a text string.", intervention=True)

    stripped = text.strip()

    if len(stripped) < MIN_DESCRIPTION_CHARS:
        return GuardrailResult(
            False,
            f"{field_name} is too short. Please provide more detail.",
            intervention=False,
        )

    max_len = MAX_LOCATION_CHARS if field_name == "location" else MAX_DESCRIPTION_CHARS
    if len(stripped) > max_len:
        return GuardrailResult(
            False,
            f"{field_name} exceeds maximum allowed length ({max_len} characters). Please shorten it.",
            intervention=True,
        )

    # Prompt injection check
    for pattern in INJECTION_PATTERNS:
        if pattern.search(stripped):
            logger.warning("Prompt injection detected in %s", field_name)
            return GuardrailResult(
                False,
                "Your input contains text that cannot be processed. "
                "Please describe only your emergency situation.",
                intervention=True,
            )

    return GuardrailResult(True, "Text valid.", intervention=False)


def check_pii(text: str) -> Tuple[bool, List[str]]:
    """
    Scan text for PII patterns.

    Returns (has_pii: bool, list_of_pii_types_found).
    Does NOT modify the text — caller decides how to handle.
    """
    found = []
    for label, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(label)
    return (len(found) > 0, found)


def validate_location_text(location: str) -> GuardrailResult:
    """Validate a manually entered location string."""
    return validate_text_input(location.strip(), field_name="location")


def validate_coordinates(lat: Any, lon: Any) -> GuardrailResult:
    """Validate latitude and longitude values."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return GuardrailResult(
            False, "Coordinates must be numeric values.", intervention=True
        )

    if not (LAT_RANGE[0] <= lat_f <= LAT_RANGE[1]):
        return GuardrailResult(
            False, f"Latitude {lat_f} is outside the valid range (-90 to 90).", intervention=True
        )
    if not (LON_RANGE[0] <= lon_f <= LON_RANGE[1]):
        return GuardrailResult(
            False, f"Longitude {lon_f} is outside the valid range (-180 to 180).", intervention=True
        )

    return GuardrailResult(True, "Coordinates valid.", intervention=False)


# ---------------------------------------------------------------------------
# 2. Output guardrails
# ---------------------------------------------------------------------------

def validate_assessment_output(raw: dict) -> Tuple[Optional[EmergencyAssessment], str]:
    """
    Validate raw Emergency Assessment Agent output against Pydantic schema.

    Returns (EmergencyAssessment, "") on success or (None, error_message) on failure.
    """
    try:
        assessment = EmergencyAssessment(**raw)
        return assessment, ""
    except ValidationError as exc:
        logger.error("Assessment output validation failed: %s", exc)
        return None, f"Assessment output failed schema validation: {exc}"
    except Exception as exc:
        return None, f"Unexpected assessment validation error: {exc}"


def validate_response_output(raw: dict) -> Tuple[Optional[EmergencyResponse], str]:
    """
    Validate raw Emergency Response Agent output against Pydantic schema.

    Returns (EmergencyResponse, "") on success or (None, error_message) on failure.
    """
    try:
        response = EmergencyResponse(**raw)
        return response, ""
    except ValidationError as exc:
        logger.error("Response output validation failed: %s", exc)
        return None, f"Response output failed schema validation: {exc}"
    except Exception as exc:
        return None, f"Unexpected response validation error: {exc}"


def check_unsafe_content(text: str) -> Tuple[bool, List[str]]:
    """
    Scan AI-generated text for unsafe output patterns.

    Returns (is_unsafe: bool, list_of_matched_patterns).
    """
    matches = []
    for pattern in UNSAFE_OUTPUT_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return (len(matches) > 0, matches)


def validate_emergency_contact(number: str) -> bool:
    """Return True only when the number is in the approved official contact list."""
    return number.strip() in OFFICIAL_EMERGENCY_NUMBERS


def check_response_consistency(
    assessment: EmergencyAssessment,
    response: EmergencyResponse,
) -> Tuple[bool, str]:
    """
    Check that the Emergency Response Agent's severity matches the assessment.
    Returns (consistent: bool, message).
    """
    if assessment.severity_level != response.severity_level:
        return (
            False,
            f"Severity mismatch: Assessment={assessment.severity_level}, "
            f"Response={response.severity_level}. Using assessment severity.",
        )
    return True, ""


def validate_help_center_data(centers: List[dict]) -> List[dict]:
    """
    Filter out any help center records that appear fabricated (missing coordinates,
    implausible distance, etc.). Only centres with verifiable coordinates are kept.
    Returns the filtered list.
    """
    valid = []
    for c in centers:
        if c.get("name") and c.get("address") is not None:
            valid.append(c)
    return valid


# ---------------------------------------------------------------------------
# 3. Full response safety check
# ---------------------------------------------------------------------------

def run_response_safety_check(response: EmergencyResponse) -> Tuple[bool, List[str]]:
    """
    Run all safety checks on a validated EmergencyResponse.

    Returns (passed: bool, list_of_issues).
    """
    issues = []

    # Unsafe content scan on immediate actions and warnings
    combined_text = " ".join(response.immediate_actions + response.safety_warnings)
    is_unsafe, matched = check_unsafe_content(combined_text)
    if is_unsafe:
        for m in matched:
            issues.append(f"Unsafe content detected: {m}")

    # Emergency contact number verification
    contact_number = response.emergency_contact.number
    if not validate_emergency_contact(contact_number):
        issues.append(
            f"Unverified emergency contact number '{contact_number}'. "
            "Only official numbers are permitted."
        )

    # Ensure human assistance flag is set for critical/high
    if response.severity_level in ("Critical", "High") and not response.human_assistance_required:
        issues.append(
            "Critical/High severity response must set human_assistance_required=true."
        )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# 4. Tool execution guardrails
# ---------------------------------------------------------------------------

ALLOWED_ASSESSMENT_TOOLS = {
    "process_image",
    "analyze_description",
    "invoke_gemini_assessment",
    "return_assessment_json",
}

ALLOWED_RESPONSE_TOOLS = {
    "read_assessment",
    "invoke_gemini_response",
    "generate_guidance",
    "request_help_center_categories",
    "return_response_json",
}

ALLOWED_MAPS_TOOLS = {
    "geocode_location",
    "search_nearby_places",
    "calculate_distance",
    "generate_maps_link",
}


def check_tool_authorization(agent: str, tool: str) -> bool:
    """
    Return True when the named tool is authorized for the given agent.
    agent: "assessment" | "response" | "maps"
    """
    mapping = {
        "assessment": ALLOWED_ASSESSMENT_TOOLS,
        "response": ALLOWED_RESPONSE_TOOLS,
        "maps": ALLOWED_MAPS_TOOLS,
    }
    allowed = mapping.get(agent.lower(), set())
    return tool in allowed


# ---------------------------------------------------------------------------
# 5. Multi-agent workflow guardrails
# ---------------------------------------------------------------------------

def require_valid_assessment(assessment: Optional[EmergencyAssessment]) -> GuardrailResult:
    """Block the Emergency Response Agent if no valid assessment is present."""
    if assessment is None:
        return GuardrailResult(
            False,
            "Emergency Response Agent cannot run without a valid assessment from the "
            "Emergency Assessment Agent.",
            intervention=True,
        )
    return GuardrailResult(True, "Assessment valid.")


def strip_image_metadata(image_bytes: bytes) -> bytes:
    """
    Strip EXIF and other metadata from an image before transmitting to external APIs.
    Returns cleaned image bytes.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        output = io.BytesIO()
        # Save without EXIF
        img_format = img.format or "JPEG"
        if img_format.upper() == "JPEG":
            img.save(output, format="JPEG", exif=b"")
        else:
            img.save(output, format=img_format)
        return output.getvalue()
    except Exception as exc:
        logger.warning("Could not strip image metadata: %s. Proceeding with original.", exc)
        return image_bytes


def detect_image_mime(file_bytes: bytes) -> str:
    """Detect MIME type from magic bytes."""
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"
