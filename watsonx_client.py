"""
watsonx_client.py — IBM watsonx.ai LLM client for Emergency Assist AI.

Uses the ibm-watsonx-ai SDK to call IBM foundation models.
Credentials are read from environment variables (or .env file):
  WATSONX_API_KEY   — IBM Cloud API key
  WATSONX_PROJECT_ID — watsonx.ai project ID
  WATSONX_URL       — watsonx.ai endpoint (default: us-south)
  WATSONX_MODEL_ID  — foundation model to use (default: ibm/granite-3-8b-instruct)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — ibm-watsonx-ai may not be installed in all environments
# ---------------------------------------------------------------------------

_WatsonxAI = None          # ibm_watsonx_ai.foundation_models.ModelInference
_credentials_cls = None    # ibm_watsonx_ai.Credentials

def _import_sdk() -> bool:
    """Attempt to import the IBM watsonx.ai SDK. Returns True on success."""
    global _WatsonxAI, _credentials_cls
    if _WatsonxAI is not None:
        return True
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        _WatsonxAI = ModelInference
        _credentials_cls = Credentials
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_model: Optional[object] = None       # ModelInference instance
_model_id: str = "ibm/granite-3-8b-instruct"
_project_id: str = ""

WATSONX_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure_watsonx(
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    model_id: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """
    Initialise the watsonx.ai ModelInference client.
    Falls back to environment variables when arguments are omitted.
    """
    global _model, _model_id, _project_id

    if not _import_sdk():
        raise RuntimeError(
            "ibm-watsonx-ai package is not installed. "
            "Run: pip install ibm-watsonx-ai"
        )

    _api_key    = api_key    or os.environ.get("WATSONX_API_KEY", "")
    _project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
    _url        = url        or os.environ.get("WATSONX_URL", WATSONX_DEFAULT_URL)
    _model_id   = model_id   or os.environ.get("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

    if not _api_key:
        raise ValueError("WATSONX_API_KEY is required.")
    if not _project_id:
        raise ValueError("WATSONX_PROJECT_ID is required.")

    creds = _credentials_cls(url=_url, api_key=_api_key)
    _model = _WatsonxAI(
        model_id=_model_id,
        credentials=creds,
        project_id=_project_id,
        params={
            "max_new_tokens": 1024,
            "temperature": 0.3,
            "repetition_penalty": 1.1,
        },
    )
    logger.info("watsonx.ai client configured — model: %s | project: %s", _model_id, _project_id)


def is_configured() -> bool:
    return _model is not None


def get_model():
    if _model is None:
        raise RuntimeError(
            "watsonx.ai client not configured. Call configure_watsonx() first."
        )
    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_text(prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Call the watsonx.ai foundation model with the given prompt.
    Returns the generated text string.
    """
    model = get_model()
    response = model.generate_text(prompt=prompt, params={"max_new_tokens": max_new_tokens})
    return response or ""
