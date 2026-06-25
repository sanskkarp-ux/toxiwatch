"""
routes.py — API Endpoints (Route Handlers)
==========================================
This file defines all the API endpoints for ToxiWatch.

An "endpoint" is a URL that the frontend (or any client) can call.
FastAPI uses Python decorators to map URLs to functions:

    @router.post("/moderate")   → handles POST requests to /moderate
    @router.get("/history")     → handles GET requests to /history

Think of it like a restaurant menu:
  - Each endpoint is a menu item
  - The client "orders" by sending a request to a URL
  - The handler function prepares and returns the "meal" (response)

Available endpoints:
  POST /moderate  → Analyze a comment for toxicity
  GET  /history   → Retrieve past moderation logs
  GET  /stats     → Summary statistics
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models import (
    CommentRequest,
    ModerationResponse,
    HistoryResponse,
    StatsResponse
)
from app.ml_engine import analyze_comment
from app.database import (
    save_moderation_result,
    get_moderation_history,
    get_total_count,
    get_action_counts
)

logger = logging.getLogger(__name__)

# APIRouter groups related endpoints together.
# In main.py, we use app.include_router(router) to attach them to the app.
router = APIRouter()


# ── POST /moderate ────────────────────────────────────────────────────────────
@router.post(
    "/moderate",
    response_model=ModerationResponse,
    tags=["Moderation"],
    summary="Analyze a comment for toxicity",
    description=(
        "Submit a comment (English, Hindi, or Hinglish) to be analyzed. "
        "First runs the Hybrid Keyword Abuse Filter (instant BLOCK for explicit abuse). "
        "If not caught, runs Detoxify ML scoring for nuanced decisions."
    )
)
async def moderate_comment(request: CommentRequest):
    """
    Main moderation endpoint — Hybrid Pipeline.

    Flow:
        1. Receive comment from frontend (JSON body).
        2. Run Hybrid Abuse Filter (keyword/regex). If blocked → skip step 3.
        3. Run Detoxify ML pipeline (language detection + translation + scoring).
        4. Save result to SQLite database.
        5. Return structured response.
    """
    logger.info("[Moderate] Received: '%.80s'...", request.comment)

    try:
        # ── Run the Hybrid Moderation Pipeline ───────────────────────────────
        # Phase 1: Keyword abuse filter (app/abuse_filter.py)
        # Phase 2: Detoxify ML model (only if Phase 1 did not block)
        result = analyze_comment(request.comment)

        # ── Save to database ──────────────────────────────────────────────────
        log_id = save_moderation_result(
            original_comment=result["original_comment"],
            translated_comment=result["translated_comment"],
            language=result["detected_language"],
            toxicity_score=result["toxicity_score"],
            action=result["action"],
            scores=result["scores"],
            matched_word=result.get("matched_word"),
            matched_rule=result.get("matched_rule"),
            blocked_by=result.get("blocked_by"),
        )

        # ── Build and return the response ─────────────────────────────────────
        response_data = {
            "original_comment": result["original_comment"],
            "translated_comment": result["translated_comment"],
            "detected_language": result["detected_language"],
            "text_analyzed": result["text_analyzed"],
            "toxicity_score": result["toxicity_score"],
            "scores": result["scores"],
            "action": result["action"],
            "log_id": log_id,
            "matched_word": result.get("matched_word"),
            "matched_rule": result.get("matched_rule"),
            "blocked_by": result.get("blocked_by"),
        }

        logger.info(
            "[Moderate] Done | lang=%s | score=%.4f | action=%s | blocked_by=%s",
            result["detected_language"],
            result["toxicity_score"],
            result["action"],
            result.get("blocked_by", "N/A"),
        )

        return response_data

    except Exception as e:
        logger.error("[Moderate] Failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Moderation processing failed: {str(e)}"
        )


# ── GET /history ──────────────────────────────────────────────────────────────
@router.get(
    "/history",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Retrieve moderation history",
    description="Returns a paginated list of all past moderation logs, newest first."
)
async def get_history(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page (max 100)")
):
    """
    Retrieve paginated moderation history from SQLite.

    Query parameters (appended to URL):
        /history?page=1&per_page=20

    Pagination example:
        Total 50 records, per_page=20:
          Page 1: records 1–20  (offset=0)
          Page 2: records 21–40 (offset=20)
          Page 3: records 41–50 (offset=40)
    """
    try:
        # Calculate offset for SQL LIMIT/OFFSET
        offset = (page - 1) * per_page

        logs = get_moderation_history(limit=per_page, offset=offset)
        total = get_total_count()

        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "logs": logs,
        }

    except Exception as e:
        logger.error("[History] Fetch failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


# ── GET /stats ─────────────────────────────────────────────────────────────────
@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Statistics"],
    summary="Get moderation statistics",
    description="Returns summary counts and rates for moderation actions."
)
async def get_stats():
    """
    Dashboard statistics endpoint.

    Returns:
        - Total comments moderated
        - Count per action (APPROVED, REVIEW, BLOCKED)
        - Approval rate and block rate as percentages
    """
    try:
        total = get_total_count()
        action_counts = get_action_counts()

        approved = action_counts.get("APPROVED", 0)
        review = action_counts.get("REVIEW", 0)
        blocked = action_counts.get("BLOCKED", 0)

        # Calculate rates (avoid division by zero when total=0)
        approval_rate = round((approved / total * 100) if total > 0 else 0, 1)
        block_rate = round((blocked / total * 100) if total > 0 else 0, 1)

        return {
            "total_moderated": total,
            "approved": approved,
            "review": review,
            "blocked": blocked,
            "approval_rate": approval_rate,
            "block_rate": block_rate,
        }

    except Exception as e:
        logger.error("[Stats] Fetch failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


# ── GET /test-comments ────────────────────────────────────────────────────────
@router.get(
    "/test-comments",
    tags=["Utility"],
    summary="Get sample test comments",
    description="Returns 30 pre-defined test comments for development and testing."
)
async def get_test_comments():
    """
    Returns a curated set of 30 test comments covering:
    - English (safe, mild, severe)
    - Hindi (Devanagari script)
    - Hinglish (Hindi in Roman script)
    - Edge cases (mixed language, punctuation-heavy, etc.)
    """
    test_comments = [
        # ── ENGLISH: SAFE / APPROVED ──────────────────────────────────────────
        {"id": 1,  "text": "This movie was absolutely brilliant! The cinematography was stunning.", "expected": "APPROVED", "lang": "en"},
        {"id": 2,  "text": "I loved the storyline and character development in this film.", "expected": "APPROVED", "lang": "en"},
        {"id": 3,  "text": "Great acting by the lead actor. Definitely worth watching.", "expected": "APPROVED", "lang": "en"},
        {"id": 4,  "text": "The screenplay was well-crafted. Enjoyed every scene.", "expected": "APPROVED", "lang": "en"},
        {"id": 5,  "text": "One of the best films I have seen this year. Highly recommended.", "expected": "APPROVED", "lang": "en"},

        # ── ENGLISH: MILD TOXICITY / REVIEW ───────────────────────────────────
        {"id": 6,  "text": "This movie was pretty bad. The director has no idea what he is doing.", "expected": "REVIEW", "lang": "en"},
        {"id": 7,  "text": "Terrible casting choices. Whoever approved this needs their eyes checked.", "expected": "REVIEW", "lang": "en"},
        {"id": 8,  "text": "What a waste of money. The producers clearly do not care about quality.", "expected": "REVIEW", "lang": "en"},

        # ── ENGLISH: SEVERE TOXICITY / BLOCKED ───────────────────────────────
        {"id": 9,  "text": "You are an idiot if you liked this garbage film.", "expected": "BLOCKED", "lang": "en"},
        {"id": 10, "text": "The director is a complete moron who should never make movies again.", "expected": "BLOCKED", "lang": "en"},
        {"id": 11, "text": "Anyone who enjoyed this stupid movie is absolutely brain-dead.", "expected": "BLOCKED", "lang": "en"},
        {"id": 12, "text": "This film is trash and the lead actor should be ashamed of himself.", "expected": "BLOCKED", "lang": "en"},

        # ── HINDI (DEVANAGARI): SAFE ──────────────────────────────────────────
        {"id": 13, "text": "यह फिल्म बहुत अच्छी थी। मुझे बहुत पसंद आई।", "expected": "APPROVED", "lang": "hi"},
        {"id": 14, "text": "इस फिल्म की कहानी बहुत सुंदर है।", "expected": "APPROVED", "lang": "hi"},
        {"id": 15, "text": "अभिनेता का प्रदर्शन शानदार था।", "expected": "APPROVED", "lang": "hi"},

        # ── HINDI (DEVANAGARI): MILD TOXICITY ────────────────────────────────
        {"id": 16, "text": "यह फिल्म बकवास है, पैसे बर्बाद हो गए।", "expected": "REVIEW", "lang": "hi"},
        {"id": 17, "text": "निर्देशक को कुछ नहीं पता, बहुत खराब फिल्म बनाई।", "expected": "REVIEW", "lang": "hi"},

        # ── HINDI (DEVANAGARI): SEVERE TOXICITY ──────────────────────────────
        {"id": 18, "text": "तू बेवकूफ है, इस फिल्म को पसंद करने वाला।", "expected": "BLOCKED", "lang": "hi"},
        {"id": 19, "text": "यह बेकार फिल्म है, बनाने वाले मूर्ख हैं।", "expected": "BLOCKED", "lang": "hi"},

        # ── HINGLISH: SAFE ────────────────────────────────────────────────────
        {"id": 20, "text": "Yeh movie bahut acchi thi yaar, maza aa gaya!", "expected": "APPROVED", "lang": "hinglish"},
        {"id": 21, "text": "Bilkul sahi movie thi, director ne bahut accha kaam kiya.", "expected": "APPROVED", "lang": "hinglish"},
        {"id": 22, "text": "Bhai is movie ki story toh ekdum mast hai.", "expected": "APPROVED", "lang": "hinglish"},

        # ── HINGLISH: MILD TOXICITY ────────────────────────────────────────────
        {"id": 23, "text": "Tera review bakwaas hai, movie toh acchi thi.", "expected": "REVIEW", "lang": "hinglish"},
        {"id": 24, "text": "Yeh faltu movie hai, timepass bhi nahi.", "expected": "REVIEW", "lang": "hinglish"},
        {"id": 25, "text": "Director ko kuch nahi aata, ganda kaam kiya hai.", "expected": "REVIEW", "lang": "hinglish"},

        # ── HINGLISH: SEVERE TOXICITY / BLOCKED ──────────────────────────────
        {"id": 26, "text": "Tu bewakoof hai, aisi movie ko like karta hai.", "expected": "BLOCKED", "lang": "hinglish"},
        {"id": 27, "text": "Tu bahut bada bewakoof hai yaar, kuch samajh nahi aata tujhe.", "expected": "BLOCKED", "lang": "hinglish"},
        {"id": 28, "text": "Kamina director ne paisa barbad karwa diya, ullu ke patthe.", "expected": "BLOCKED", "lang": "hinglish"},

        # ── MIXED / EDGE CASES ─────────────────────────────────────────────────
        {"id": 29, "text": "This movie is bakwaas, total waste of time yaar.", "expected": "REVIEW", "lang": "hinglish"},
        {"id": 30, "text": "Amazing film! Bilkul sahi tha, must watch for everyone!", "expected": "APPROVED", "lang": "mixed"},
    ]

    return {"test_comments": test_comments, "total": len(test_comments)}
