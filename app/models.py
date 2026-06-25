"""
models.py — Pydantic Data Models (Request & Response Schemas)
=============================================================
Pydantic v2 compatible models for ToxiWatch.

FIXES APPLIED:
  - Replaced deprecated `@validator` with `@field_validator` (Pydantic v2)
  - Replaced `class Config` with `model_config = ConfigDict(...)` (Pydantic v2)
  - Replaced `schema_extra` with `json_schema_extra` (Pydantic v2)
  - `@field_validator` requires `@classmethod` decorator in Pydantic v2
  - `mode='before'` must be explicitly set for pre-processing validators

Think of Pydantic models like a FORM:
  - The form has required fields and optional fields
  - If you submit the form with missing required fields → error
  - If you submit the wrong data type → error
  - FastAPI uses these to auto-validate incoming JSON
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Optional
from datetime import datetime


class CommentRequest(BaseModel):
    """
    Schema for POST /moderate request body.

    The user sends JSON like:
        {"comment": "Tu bahut bada bewakoof hai"}

    Field constraints:
        - comment must be a string
        - minimum 1 character (can't be empty)
        - maximum 2000 characters (reasonable comment length)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "comment": "Tu bahut bada bewakoof hai"
            }
        }
    )

    comment: str = Field(
        ...,                          # The `...` means "required field"
        min_length=1,                 # Can't be an empty string
        max_length=2000,              # Limit comment length
        description="The comment text to moderate. Supports English, Hindi, Hinglish.",
        examples=["Tu bahut bada bewakoof hai"],
    )

    @field_validator("comment", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """
        Automatically strip leading/trailing whitespace from comments.
        Runs BEFORE the min_length check so "   " becomes "" and fails min_length.

        E.g. "  hello  " → "hello"
             "   "       → "" → triggers min_length=1 validation error
        """
        if isinstance(v, str):
            return v.strip()
        return v


class ScoresDetail(BaseModel):
    """
    Schema for the detailed per-category toxicity scores.
    All scores are floats in the range [0.0, 1.0].
    """
    toxicity: float = Field(0.0, ge=0.0, le=1.0)
    severe_toxicity: float = Field(0.0, ge=0.0, le=1.0)
    obscene: float = Field(0.0, ge=0.0, le=1.0)
    threat: float = Field(0.0, ge=0.0, le=1.0)
    insult: float = Field(0.0, ge=0.0, le=1.0)
    identity_attack: float = Field(0.0, ge=0.0, le=1.0)
    # Some Detoxify variants include sexual_explicit
    sexual_explicit: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModerationResponse(BaseModel):
    """
    Schema for POST /moderate response body.

    The API returns JSON like:
        {
          "original_comment": "Tu bahut bada bewakoof hai",
          "translated_comment": "You are a very big idiot",
          "detected_language": "hinglish",
          "text_analyzed": "You are a very big idiot",
          "toxicity_score": 0.89,
          "scores": {...},
          "action": "BLOCKED",
          "log_id": 42
        }
    """

    original_comment: str = Field(..., description="The original submitted comment.")

    translated_comment: Optional[str] = Field(
        None,
        description="English translation (only present for Hindi/Hinglish comments)."
    )

    detected_language: str = Field(
        ...,
        description="Detected language: 'en', 'hi', 'hinglish', or 'other'."
    )

    text_analyzed: str = Field(
        ...,
        description="The actual text that was scored (translated version if applicable)."
    )

    toxicity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Primary toxicity score between 0.0 and 1.0."
    )

    scores: Dict[str, float] = Field(
        ...,
        description="Detailed scores for each toxicity category."
    )

    action: str = Field(
        ...,
        description="Moderation decision: APPROVED, REVIEW, or BLOCKED.",
        examples=["BLOCKED"]
    )

    log_id: Optional[int] = Field(
        None,
        description="Database ID of the saved moderation log entry."
    )

    # ── Hybrid Filter metadata ────────────────────────────────────────────────
    matched_word: Optional[str] = Field(
        None,
        description="The abuse word/phrase matched by the Keyword Filter (None if Detoxify decided)."
    )
    matched_rule: Optional[str] = Field(
        None,
        description="The specific rule that triggered in the Keyword Filter."
    )
    blocked_by: Optional[str] = Field(
        None,
        description="Which system made the BLOCKED decision: 'Keyword Filter' or 'Detoxify'."
    )


class ModerationLogEntry(BaseModel):
    """
    Schema for a single entry in the moderation history log.
    Used in the GET /history response.
    """
    id: int
    original_comment: str
    translated_comment: Optional[str] = None
    language: str
    toxicity_score: float
    action: str
    scores: Optional[Dict[str, float]] = None
    matched_word: Optional[str] = None
    matched_rule: Optional[str] = None
    blocked_by: Optional[str] = None
    timestamp: Optional[str] = None


class HistoryResponse(BaseModel):
    """
    Schema for GET /history response.
    Returns a list of log entries plus pagination metadata.
    """
    total: int = Field(..., description="Total number of records in database.")
    page: int = Field(..., description="Current page number.")
    per_page: int = Field(..., description="Records per page.")
    logs: list = Field(..., description="List of moderation log entries.")


class StatsResponse(BaseModel):
    """
    Schema for GET /stats response.
    Summary statistics for the dashboard.
    """
    total_moderated: int
    approved: int
    review: int
    blocked: int
    approval_rate: float   # Percentage of APPROVED (0–100)
    block_rate: float      # Percentage of BLOCKED (0–100)
