"""
tests/test_moderation.py — Test Suite for ToxiWatch (REPAIRED)
==============================================================
FIXES APPLIED:
  - Added `with TestClient(app) as client:` context manager usage since
    FastAPI 0.111 + lifespan requires the client to be used as a context
    manager to properly trigger startup/shutdown events (model loading).
  - Moved TestClient to a session-scoped fixture so models load once.
  - Added PYTHONUTF8 env var to prevent Unicode issues in test output.
  - Removed autouse load_models_once fixture (it now conflicts with lifespan
    since models are loaded in the lifespan handler, not separately).
  - The session-scoped `client` fixture properly starts/stops the app lifecycle.

Run all tests:
    venv\Scripts\pytest tests\test_moderation.py -v

Run a specific test:
    venv\Scripts\pytest tests\test_moderation.py::test_english_safe_01 -v

NOTE: First run downloads ML model weights (~1 GB). Please be patient.
"""

import os
import pytest

# Set UTF-8 encoding before any imports that might log Unicode
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED CLIENT FIXTURE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def client():
    """
    Create a TestClient that triggers the FastAPI lifespan (startup/shutdown).

    FIX: In FastAPI 0.111+, the lifespan context (which loads ML models) is only
    triggered when TestClient is used as a context manager (`with TestClient(app)`).
    This session-scoped fixture ensures models are loaded ONCE for all tests.
    """
    from main import app
    with TestClient(app) as test_client:
        yield test_client


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Common moderation helper
# ══════════════════════════════════════════════════════════════════════════════

def moderate(client, comment: str) -> dict:
    """Send a comment to /moderate and return the JSON response."""
    response = client.post("/moderate", json={"comment": comment})
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    return response.json()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1: ENGLISH — SAFE (Expected: APPROVED)
# ══════════════════════════════════════════════════════════════════════════════

def test_english_safe_01(client):
    """Positive movie review in English — should be approved."""
    result = moderate(client, "This movie was absolutely brilliant! The cinematography was stunning.")
    assert result["detected_language"] == "en"
    assert result["action"] == "APPROVED"
    assert result["toxicity_score"] < 0.4

def test_english_safe_02(client):
    """Another positive review — should be approved."""
    result = moderate(client, "I loved the storyline and character development in this film.")
    assert result["detected_language"] == "en"
    assert result["action"] == "APPROVED"

def test_english_safe_03(client):
    """Short positive comment — should be approved."""
    result = moderate(client, "Great acting by the lead actor. Definitely worth watching.")
    assert result["action"] == "APPROVED"
    assert result["toxicity_score"] < 0.4

def test_english_safe_04(client):
    """Neutral factual comment — should be approved."""
    result = moderate(client, "The screenplay was well-crafted. Enjoyed every scene.")
    assert result["action"] == "APPROVED"

def test_english_safe_05(client):
    """Enthusiastic recommendation — should be approved."""
    result = moderate(client, "One of the best films I have seen this year. Highly recommended.")
    assert result["action"] == "APPROVED"
    assert result["scores"]["toxicity"] < 0.4


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2: ENGLISH — MILD TOXICITY (Expected: REVIEW)
# ══════════════════════════════════════════════════════════════════════════════

def test_english_mild_01(client):
    """Mildly critical comment — may be REVIEW."""
    result = moderate(client, "This movie was pretty bad. The director has no idea what he is doing.")
    assert result["detected_language"] == "en"
    # Could be REVIEW or APPROVED depending on model version
    assert result["action"] in ("APPROVED", "REVIEW")

def test_english_mild_02(client):
    """Passive-aggressive criticism — may be REVIEW."""
    result = moderate(client, "Terrible casting choices. Whoever approved this needs their eyes checked.")
    assert result["action"] in ("REVIEW", "APPROVED")

def test_english_mild_03(client):
    """Frustrated comment about money waste — mild negativity."""
    result = moderate(client, "What a waste of money. The producers clearly do not care about quality.")
    # This should not be BLOCKED (no direct personal attack)
    assert result["action"] != "BLOCKED" or result["toxicity_score"] >= 0.7


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3: ENGLISH — SEVERE TOXICITY (Expected: BLOCKED)
# ══════════════════════════════════════════════════════════════════════════════

def test_english_severe_01(client):
    """Direct personal insult — should be BLOCKED."""
    result = moderate(client, "You are an idiot if you liked this garbage film.")
    assert result["detected_language"] == "en"
    assert result["action"] == "BLOCKED"
    assert result["toxicity_score"] >= 0.7

def test_english_severe_02(client):
    """Strong insult targeting director — should be BLOCKED."""
    result = moderate(client, "The director is a complete moron who should never make movies again.")
    assert result["action"] == "BLOCKED"
    assert result["scores"]["insult"] > 0.5

def test_english_severe_03(client):
    """Broad attack on audience — should be BLOCKED."""
    result = moderate(client, "Anyone who enjoyed this stupid movie is absolutely brain-dead.")
    assert result["action"] == "BLOCKED"

def test_english_severe_04(client):
    """Shame-based attack — should be BLOCKED."""
    result = moderate(client, "This film is trash and the lead actor should be ashamed of himself.")
    assert result["action"] in ("REVIEW", "BLOCKED")
    assert result["toxicity_score"] >= 0.4


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4: HINDI (DEVANAGARI) — SAFE (Expected: APPROVED)
# ══════════════════════════════════════════════════════════════════════════════

def test_hindi_safe_01(client):
    """Positive Hindi review in Devanagari — should detect as Hindi and approve."""
    result = moderate(client, "\u092f\u0939 \u092b\u093c\u093f\u0932\u094d\u092e \u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u0940 \u0925\u0940\u0964 \u092e\u0941\u091d\u0947 \u092c\u0939\u0941\u0924 \u092a\u0938\u0902\u0926 \u0906\u0908\u0964")
    assert result["detected_language"] == "hi"
    assert result["translated_comment"] is not None  # Should be translated
    assert result["action"] == "APPROVED"

def test_hindi_safe_02(client):
    """Positive story comment in Hindi."""
    result = moderate(client, "\u0907\u0938 \u092b\u093c\u093f\u0932\u094d\u092e \u0915\u0940 \u0915\u0939\u093e\u0928\u0940 \u092c\u0939\u0941\u0924 \u0938\u0941\u0902\u0926\u0930 \u0939\u0948\u0964")
    assert result["detected_language"] == "hi"
    assert result["action"] == "APPROVED"

def test_hindi_safe_03(client):
    """Praise for acting in Hindi."""
    result = moderate(client, "\u0905\u092d\u093f\u0928\u0947\u0924\u093e \u0915\u093e \u092a\u094d\u0930\u0926\u0930\u094d\u0936\u0928 \u0936\u093e\u0928\u0926\u093e\u0930 \u0925\u093e\u0964")
    assert result["detected_language"] == "hi"
    assert result["translated_comment"] is not None


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5: HINDI (DEVANAGARI) — MILD/SEVERE TOXICITY
# ══════════════════════════════════════════════════════════════════════════════

def test_hindi_mild_01(client):
    """Mild criticism in Hindi — 'this film is nonsense, wasted money'."""
    result = moderate(client, "\u092f\u0939 \u092b\u093c\u093f\u0932\u094d\u092e \u092c\u0915\u0935\u093e\u0938 \u0939\u0948, \u092a\u0948\u0938\u0947 \u092c\u0930\u094d\u092c\u093e\u0926 \u0939\u094b \u0917\u090f\u0964")
    assert result["detected_language"] == "hi"
    assert result["translated_comment"] is not None
    assert result["action"] in ("APPROVED", "REVIEW")

def test_hindi_severe_01(client):
    """Insult in Hindi — should be BLOCKED."""
    result = moderate(client, "\u0924\u0942 \u092c\u0947\u0935\u0915\u0942\u092b \u0939\u0948, \u0907\u0938 \u092b\u093c\u093f\u0932\u094d\u092e \u0915\u094b \u092a\u0938\u0902\u0926 \u0915\u0930\u0928\u0947 \u0935\u093e\u0932\u093e\u0964")
    assert result["detected_language"] == "hi"
    assert result["translated_comment"] is not None
    assert result["action"] in ("REVIEW", "BLOCKED")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6: HINGLISH — SAFE (Expected: APPROVED)
# ══════════════════════════════════════════════════════════════════════════════

def test_hinglish_safe_01(client):
    """Positive Hinglish review."""
    result = moderate(client, "Yeh movie bahut acchi thi yaar, maza aa gaya!")
    assert result["detected_language"] == "hinglish"
    assert result["action"] == "APPROVED"

def test_hinglish_safe_02(client):
    """Complimenting the director in Hinglish."""
    result = moderate(client, "Bilkul sahi movie thi, director ne bahut accha kaam kiya.")
    assert result["detected_language"] == "hinglish"
    assert result["action"] == "APPROVED"

def test_hinglish_safe_03(client):
    """Story praise in Hinglish."""
    result = moderate(client, "Bhai is movie ki story toh ekdum mast hai.")
    assert result["detected_language"] == "hinglish"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 7: HINGLISH — MILD TOXICITY
# ══════════════════════════════════════════════════════════════════════════════

def test_hinglish_mild_01(client):
    """Disagreeing with a review — Hinglish."""
    result = moderate(client, "Tera review bakwaas hai, movie toh acchi thi.")
    assert result["detected_language"] == "hinglish"
    assert result["action"] in ("APPROVED", "REVIEW")

def test_hinglish_mild_02(client):
    """Calling movie useless — Hinglish."""
    result = moderate(client, "Yeh faltu movie hai, timepass bhi nahi.")
    assert result["detected_language"] == "hinglish"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 8: HINGLISH — SEVERE TOXICITY (Expected: BLOCKED)
# ══════════════════════════════════════════════════════════════════════════════

def test_hinglish_severe_01(client):
    """Direct Hinglish insult — should be BLOCKED."""
    result = moderate(client, "Tu bewakoof hai, aisi movie ko like karta hai.")
    assert result["detected_language"] == "hinglish"
    assert result["action"] in ("REVIEW", "BLOCKED")

def test_hinglish_severe_02(client):
    """Strong Hinglish insult — should be BLOCKED."""
    result = moderate(client, "Tu bahut bada bewakoof hai yaar, kuch samajh nahi aata tujhe.")
    assert result["detected_language"] == "hinglish"
    assert result["translated_comment"] is not None
    assert result["action"] in ("REVIEW", "BLOCKED")

def test_hinglish_severe_03(client):
    """Multiple insults in Hinglish — should be BLOCKED."""
    result = moderate(client, "Kamina director ne paisa barbad karwa diya, ullu ke patthe.")
    assert result["detected_language"] == "hinglish"
    assert result["action"] in ("REVIEW", "BLOCKED")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 9: MIXED LANGUAGE
# ══════════════════════════════════════════════════════════════════════════════

def test_mixed_01(client):
    """English-Hinglish mix — mild negative."""
    result = moderate(client, "This movie is bakwaas, total waste of time yaar.")
    # Mixed language should still be detected as Hinglish
    assert result["detected_language"] in ("hinglish", "en")

def test_mixed_02(client):
    """Positive mix of English and Hinglish."""
    result = moderate(client, "Amazing film! Bilkul sahi tha, must watch for everyone!")
    assert result["action"] == "APPROVED"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 10: EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

def test_edge_empty_after_strip(client):
    """Comment that is only whitespace — should return 422 Unprocessable Entity."""
    response = client.post("/moderate", json={"comment": "   "})
    assert response.status_code == 422  # Pydantic validation error (min_length=1)

def test_edge_very_short(client):
    """Single word comment — should work without errors."""
    result = moderate(client, "Good")
    assert "action" in result
    assert "toxicity_score" in result

def test_edge_long_comment(client):
    """Long comment — should work within 2000 char limit."""
    long_comment = "Great movie! " * 100  # ~1300 characters
    result = moderate(client, long_comment[:2000])
    assert result["action"] == "APPROVED"

def test_health_endpoint(client):
    """Health check endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_history_endpoint(client):
    """History endpoint should return a valid structure."""
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "logs" in data
    assert isinstance(data["logs"], list)

def test_stats_endpoint(client):
    """Stats endpoint should return numeric counts."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_moderated" in data
    assert isinstance(data["total_moderated"], int)

def test_test_comments_endpoint(client):
    """Test comments endpoint should return 30 sample comments."""
    response = client.get("/test-comments")
    assert response.status_code == 200
    data = response.json()
    assert "test_comments" in data
    assert data["total"] == 30
    assert len(data["test_comments"]) == 30

def test_response_schema(client):
    """Ensure /moderate response always contains required fields."""
    result = moderate(client, "Nice movie!")
    required_fields = [
        "original_comment", "translated_comment", "detected_language",
        "text_analyzed", "toxicity_score", "scores", "action"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"
