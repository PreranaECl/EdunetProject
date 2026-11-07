"""
memory.py — Cross-session memory storage for Emergency Assist AI.

Uses SQLite (stdlib — no extra dependencies) to persist workflow history
across Streamlit page refreshes and application restarts.

Only stores non-sensitive metadata — no API keys, descriptions, images,
or PII are ever written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

_DB_DIR = Path(__file__).parent / ".data"
_DB_PATH = _DB_DIR / "emergency_assist_memory.db"

MAX_HISTORY_ROWS = 200   # keep last N workflows to prevent unbounded growth


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_WORKFLOWS_TABLE = """
CREATE TABLE IF NOT EXISTS workflow_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    emergency_type  TEXT    NOT NULL,
    severity_level  TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    total_time_s    REAL    DEFAULT 0.0,
    assessment_s    REAL    DEFAULT 0.0,
    response_s      REAL    DEFAULT 0.0,
    maps_s          REAL    DEFAULT 0.0,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cost_usd        REAL    DEFAULT 0.0,
    guardrail_hits  INTEGER DEFAULT 0,
    maps_results    INTEGER DEFAULT 0,
    notes           TEXT    DEFAULT ''
);
"""

_CREATE_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS session_stats (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    total_requests      INTEGER DEFAULT 0,
    successful          INTEGER DEFAULT 0,
    failed              INTEGER DEFAULT 0,
    total_input_tokens  INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd      REAL    DEFAULT 0.0,
    guardrail_total     INTEGER DEFAULT 0,
    last_updated        TEXT    DEFAULT ''
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn():
    """Yield a SQLite connection, always closing it afterwards."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("SQLite error: %s", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    try:
        with _get_conn() as conn:
            conn.execute(_CREATE_WORKFLOWS_TABLE)
            conn.execute(_CREATE_STATS_TABLE)
            # Upsert the single stats row
            conn.execute(
                "INSERT OR IGNORE INTO session_stats (id) VALUES (1)"
            )
        logger.info("Memory DB initialised at %s", _DB_PATH)
    except Exception as exc:
        logger.warning("Could not initialise memory DB: %s", exc)


# ---------------------------------------------------------------------------
# Write a workflow result
# ---------------------------------------------------------------------------

def save_workflow(
    emergency_type: str,
    severity_level: str,
    status: str,
    total_time_s: float = 0.0,
    assessment_s: float = 0.0,
    response_s: float = 0.0,
    maps_s: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    guardrail_hits: int = 0,
    maps_results: int = 0,
    notes: str = "",
) -> None:
    """
    Persist one workflow record to SQLite.
    Trims oldest rows when table exceeds MAX_HISTORY_ROWS.
    """
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO workflow_history
                    (created_at, emergency_type, severity_level, status,
                     total_time_s, assessment_s, response_s, maps_s,
                     input_tokens, output_tokens, cost_usd,
                     guardrail_hits, maps_results, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (now, emergency_type, severity_level, status,
                 total_time_s, assessment_s, response_s, maps_s,
                 input_tokens, output_tokens, cost_usd,
                 guardrail_hits, maps_results, notes),
            )
            # Update cumulative stats row
            conn.execute(
                """
                UPDATE session_stats SET
                    total_requests      = total_requests + 1,
                    successful          = successful + ?,
                    failed              = failed + ?,
                    total_input_tokens  = total_input_tokens + ?,
                    total_output_tokens = total_output_tokens + ?,
                    total_cost_usd      = total_cost_usd + ?,
                    guardrail_total     = guardrail_total + ?,
                    last_updated        = ?
                WHERE id = 1
                """,
                (
                    1 if status == "success" else 0,
                    1 if status != "success" else 0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    guardrail_hits,
                    now,
                ),
            )
            # Trim old rows
            conn.execute(
                """
                DELETE FROM workflow_history WHERE id NOT IN (
                    SELECT id FROM workflow_history ORDER BY id DESC LIMIT ?
                )
                """,
                (MAX_HISTORY_ROWS,),
            )
    except Exception as exc:
        logger.warning("Could not save workflow to memory DB: %s", exc)


# ---------------------------------------------------------------------------
# Read history
# ---------------------------------------------------------------------------

def load_workflow_history(limit: int = 50) -> List[dict]:
    """Return the most recent `limit` workflow records as list of dicts."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM workflow_history
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Could not load workflow history: %s", exc)
        return []


def load_cumulative_stats() -> dict:
    """Return the lifetime cumulative stats row."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM session_stats WHERE id = 1"
            ).fetchone()
        return dict(row) if row else {}
    except Exception as exc:
        logger.warning("Could not load cumulative stats: %s", exc)
        return {}


def clear_history() -> None:
    """Delete all stored workflow records (stats are preserved)."""
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM workflow_history")
        logger.info("Workflow history cleared.")
    except Exception as exc:
        logger.warning("Could not clear history: %s", exc)


def db_exists() -> bool:
    """Return True if the database file exists."""
    return _DB_PATH.exists()
