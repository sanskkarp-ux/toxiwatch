"""
database.py — SQLite Database Layer (REPAIRED)
===============================================
This file handles all database operations for ToxiWatch.

We use SQLite — a lightweight, file-based database that needs no server setup.
The entire database is stored in a single file: `toxiwatch.db`.

FIXES APPLIED:
  - Replaced relative DATABASE_PATH = "toxiwatch.db" with an absolute path
    using pathlib.Path(__file__).parent.parent to place the DB in the project root.
    This prevents the database being created in random directories depending on
    where uvicorn is launched from.
  - Replaced emoji characters in log messages with ASCII-safe equivalents to
    prevent UnicodeEncodeError on Windows consoles.
  - Added explicit check_same_thread=False for SQLite connections to prevent
    threading errors when FastAPI's async handlers access the database.

TABLE SCHEMA (moderation_logs):
  id                INTEGER PRIMARY KEY AUTOINCREMENT
  original_comment  TEXT    NOT NULL
  translated_comment TEXT
  language          TEXT    NOT NULL
  toxicity_score    REAL    NOT NULL
  action            TEXT    NOT NULL
  scores_json       TEXT
  timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Absolute path to the SQLite database file ─────────────────────────────────
# FIX: Replaced relative "toxiwatch.db" with an absolute path anchored to the
# project root directory (parent of the app/ package directory).
# This ensures the DB file is always created in the project root regardless of
# which directory the server is started from.
BASE_DIR = Path(__file__).parent.parent.resolve()   # e.g. E:\toxiwatch\toxiwatch
DATABASE_PATH = str(BASE_DIR / "toxiwatch.db")


def get_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the SQLite database.

    sqlite3.connect() opens (or creates) the database file.
    Setting row_factory = sqlite3.Row makes query results accessible
    like dictionaries: row["column_name"] instead of row[0].

    FIX: Added check_same_thread=False to allow FastAPI async endpoints
    to access the database across different OS threads safely.

    Returns:
        An active SQLite database connection.
    """
    conn = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False   # FIX: Required for FastAPI's threaded async model
    )
    # Row factory: makes each row behave like a dict with named columns
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize the database — create tables if they don't exist.

    This is called ONCE at server startup (from main.py lifespan).
    The `IF NOT EXISTS` clause ensures we don't accidentally delete
    existing data when restarting the server.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Enable WAL mode for better concurrent read performance
        cursor.execute("PRAGMA journal_mode=WAL")

        # CREATE TABLE IF NOT EXISTS: Only create if table doesn't already exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moderation_logs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                original_comment  TEXT    NOT NULL,
                translated_comment TEXT,
                language          TEXT    NOT NULL,
                toxicity_score    REAL    NOT NULL,
                action            TEXT    NOT NULL,
                scores_json       TEXT,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create an index on (action, timestamp) for faster stats/history queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_action_timestamp
            ON moderation_logs (action, timestamp)
        """)

        conn.commit()
        logger.info("[DB] Database initialized at '%s'", DATABASE_PATH)

    except Exception as e:
        logger.error("[DB ERROR] Database init failed: %s", str(e))
        raise  # Re-raise so the app startup fails loudly

    finally:
        conn.close()


def save_moderation_result(
    original_comment: str,
    translated_comment: Optional[str],
    language: str,
    toxicity_score: float,
    action: str,
    scores: Dict
) -> int:
    """
    Save a moderation result to the database.

    This is called after every successful /moderate request.
    We store all fields so that the /history endpoint can replay them.

    Args:
        original_comment:   The raw comment submitted by the user.
        translated_comment: The English translation (or None if English).
        language:           Detected language code ("en", "hi", "hinglish").
        toxicity_score:     The primary toxicity float score (0.0-1.0).
        action:             Moderation decision ("APPROVED", "REVIEW", "BLOCKED").
        scores:             Full dict of all category scores.

    Returns:
        The auto-generated ID of the newly inserted row.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # The ? placeholders prevent SQL injection attacks
        cursor.execute("""
            INSERT INTO moderation_logs
                (original_comment, translated_comment, language,
                 toxicity_score, action, scores_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            original_comment,
            translated_comment,
            language,
            toxicity_score,
            action,
            json.dumps(scores)   # Store dict as JSON string
        ))

        conn.commit()

        new_id = cursor.lastrowid
        logger.info("[DB] Saved moderation log (id=%d, action=%s)", new_id, action)
        return new_id

    except Exception as e:
        logger.error("[DB ERROR] Failed to save moderation result: %s", str(e))
        conn.rollback()
        raise

    finally:
        conn.close()


def get_moderation_history(limit: int = 100, offset: int = 0) -> List[Dict]:
    """
    Retrieve moderation history from the database.

    Uses ORDER BY id DESC to return newest results first (most recent at top).

    Args:
        limit:  Maximum number of records to return (pagination).
        offset: How many records to skip (for page 2, 3, etc.).

    Returns:
        List of dicts, each representing one moderation log entry.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id, original_comment, translated_comment, language,
                toxicity_score, action, scores_json, timestamp
            FROM moderation_logs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        rows = cursor.fetchall()

        # Convert each SQLite Row object to a plain Python dict
        results = []
        for row in rows:
            record = dict(row)
            # Parse JSON string back into a Python dict
            if record.get("scores_json"):
                try:
                    record["scores"] = json.loads(record["scores_json"])
                except (json.JSONDecodeError, TypeError):
                    record["scores"] = {}
                del record["scores_json"]
            else:
                record["scores"] = {}
            results.append(record)

        return results

    except Exception as e:
        logger.error("[DB ERROR] Failed to fetch history: %s", str(e))
        return []

    finally:
        conn.close()


def get_total_count() -> int:
    """
    Return the total number of moderation logs in the database.
    Used to show "X comments moderated" in the frontend dashboard.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM moderation_logs")
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        logger.error("[DB ERROR] Count query failed: %s", str(e))
        return 0
    finally:
        conn.close()


def get_action_counts() -> Dict[str, int]:
    """
    Return counts grouped by moderation action.
    Used for dashboard statistics.

    Returns something like:
        {"APPROVED": 45, "REVIEW": 12, "BLOCKED": 8}
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM moderation_logs
            GROUP BY action
        """)
        rows = cursor.fetchall()
        return {row["action"]: row["count"] for row in rows}
    except Exception as e:
        logger.error("[DB ERROR] Action counts query failed: %s", str(e))
        return {}
    finally:
        conn.close()
