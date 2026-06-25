"""
ml_engine.py — The Brain of ToxiWatch (REPAIRED)
=================================================
This file handles all Machine Learning operations:

  1. Language Detection  -> langdetect library
     Detects if text is English ("en"), Hindi ("hi"), or other.

  2. Hinglish Detection  -> Custom heuristic function
     Hinglish = Hindi words typed in English script (e.g. "tu bakwaas hai")
     langdetect sees it as English, so we check for Hindi root words manually.

  3. Translation         -> Helsinki-NLP/opus-mt-hi-en (HuggingFace Transformers)
     Converts Hindi/Hinglish -> English so the toxicity model can understand it.

  4. Toxicity Scoring    -> Detoxify library (uses BERT under the hood)
     Returns scores 0.0-1.0 for: toxicity, insult, threat, obscene, identity_attack.

FIXES APPLIED:
  - Removed Unicode arrow characters (-> instead of ->) in log messages to prevent
    UnicodeEncodeError on Windows consoles with cp1252/cp850 encoding.
  - Added sys.stdout.reconfigure(encoding='utf-8') call at module load time to
    handle Unicode emoji characters in log output on Windows.
  - Made translate_to_english more robust with explicit error messages.
  - Added graceful fallback if langdetect import fails.
  - Removed emoji from logger calls to avoid Windows console encoding crashes.
"""

import logging
import re
import sys
import os
from typing import Dict, Optional, Tuple

# ── Windows Console Encoding Fix ─────────────────────────────────────────────
# On Windows, the default console encoding is cp1252 or cp850 which cannot
# display Unicode emoji characters used in log messages.
# This forces stdout/stderr to UTF-8 so emoji in logging don't crash the server.
if sys.platform == "win32":
    try:
        # Python 3.7+ supports reconfiguring stdout encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    # Also set the PYTHONIOENCODING environment variable for subprocesses
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

logger = logging.getLogger(__name__)

# ── Module-level variables ────────────────────────────────────────────────────
# These are loaded ONCE at startup (in load_models()) and reused for every request.
# Loading ML models is expensive (like booting a computer). We do it once.
_detoxify_model = None       # Toxicity scorer
_translator_tokenizer = None # Converts text -> tokens (numbers) for the neural net
_translator_model = None     # The actual translation neural network


# ── Hinglish vocabulary ───────────────────────────────────────────────────────
# Common Hindi words written in English script (Roman/Latin letters).
# Hinglish is NOT a separate language -- it's Hindi phonetically typed in English.
# langdetect cannot detect it reliably, so we use a keyword lookup.
HINGLISH_KEYWORDS = {
    # Insults / negative
    "bakwaas", "bewakoof", "pagal", "chutiya", "gadha", "ullu",
    "kamina", "harami", "saala", "ganda", "kachra", "faltu",
    "besharam", "nikamma", "badtameez", "jhoothi", "jhootha",
    # Common Hindi pronouns / grammar
    "tera", "mera", "tum", "aap", "yeh", "woh", "hum", "main",
    "tu", "hai", "hain", "tha", "thi", "nahi", "nhi", "kya",
    "bahut", "bilkul", "zyada", "thoda", "accha", "bura",
    # Common verbs / phrases
    "kar", "karo", "raha", "rahi", "bolta", "bolti", "deta",
    "lena", "dena", "jana", "aana", "dekho", "suno",
    # Greetings / fillers
    "bhai", "yaar", "dost", "arre", "oye", "bolo",
}


def load_models():
    """
    Load all ML models into memory. Called ONCE at server startup.

    Models used:
    - Detoxify('original'): BERT-based model trained on Wikipedia toxicity data.
    - Helsinki-NLP/opus-mt-hi-en: Encoder-Decoder transformer trained on Hindi->English pairs.

    FIX: All log messages use ASCII-safe characters to prevent Windows UnicodeEncodeError.
    FIX: Suppresses two harmless HuggingFace warnings on Windows:
         1. Symlinks warning (Windows requires Developer Mode for symlinks in cache).
         2. resume_download FutureWarning from huggingface_hub < 1.0.
    """
    import warnings
    global _detoxify_model, _translator_tokenizer, _translator_model

    # Suppress HuggingFace Hub symlinks warning on Windows (cache still works, just copies)
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    # Suppress the resume_download FutureWarning from older huggingface_hub versions
    warnings.filterwarnings(
        "ignore",
        message=".*resume_download.*",
        category=FutureWarning,
        module="huggingface_hub"
    )

    # ── Load Detoxify ─────────────────────────────────────────────────────────
    logger.info("Loading Detoxify model...")
    try:
        from detoxify import Detoxify
        # 'original' is the base model trained on Wikipedia comments
        # It outputs scores for: toxicity, insult, obscene, threat, identity_attack
        _detoxify_model = Detoxify('original')
        logger.info("[OK] Detoxify loaded successfully.")
    except Exception as e:
        logger.error("[ERROR] Failed to load Detoxify model: %s", str(e))
        raise RuntimeError(f"Detoxify model failed to load: {e}") from e

    # ── Load Translation Model ────────────────────────────────────────────────
    logger.info("Loading Helsinki-NLP Hindi->English translation model...")
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-hi-en"
        # Tokenizer: converts text string -> list of integer token IDs
        # (Neural networks can't read words, only numbers)
        _translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
        # Model: takes token IDs -> generates translated token IDs
        _translator_model = MarianMTModel.from_pretrained(model_name)
        logger.info("[OK] Translation model loaded successfully.")
    except Exception as e:
        # Translation model failure is non-fatal: we can still moderate English text
        # Hindi/Hinglish comments will fall back to scoring the original text
        logger.warning(
            "[WARN] Failed to load translation model: %s. "
            "Hindi/Hinglish comments will be scored without translation.",
            str(e)
        )
        _translator_tokenizer = None
        _translator_model = None


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Returns:
        "en"       -> English
        "hi"       -> Hindi (Devanagari script)
        "hinglish" -> Hindi written in English script (romanized Hindi)
        "other"    -> Any other language

    Strategy:
        1. If text contains Devanagari Unicode characters -> it's Hindi.
        2. Use langdetect to get the detected language code.
        3. If langdetect says "en" but our Hinglish keywords are found -> it's Hinglish.
        4. If langdetect says "hi" but no Devanagari found -> treat as Hinglish.
    """
    if not text or not text.strip():
        return "en"

    # Step 1: Check for Devanagari script characters (Unicode range: U+0900-U+097F)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(text):
        return "hi"  # Actual Hindi in Devanagari script

    # Step 2: Use langdetect for non-Devanagari text
    detected = "en"  # Safe default
    try:
        from langdetect import detect
        detected = detect(text)
    except Exception as e:
        # langdetect sometimes fails on very short texts (< 3 chars)
        logger.debug("langdetect failed (text too short or ambiguous): %s", str(e))
        detected = "en"

    # Step 3: Check for Hinglish keywords
    # Convert to lowercase and split into individual words for comparison
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    hinglish_hits = words_in_text.intersection(HINGLISH_KEYWORDS)

    if hinglish_hits:
        logger.debug("Hinglish keywords found: %s", hinglish_hits)
        return "hinglish"

    # Step 4: langdetect thought it was Hindi but no Devanagari -- must be Hinglish
    if detected == "hi":
        return "hinglish"

    # If langdetect detected English and no Hinglish keywords found
    if detected == "en":
        return "en"

    # Any other language (Spanish, French, etc.)
    return "other"


def translate_to_english(text: str) -> str:
    """
    Translate Hindi or Hinglish text to English using the Helsinki-NLP model.

    How it works (simplified):
        Input text -> Tokenizer converts to numbers -> Neural network "reads" numbers
        -> Generates English word IDs -> Tokenizer decodes IDs back to text

    Args:
        text: The comment text in Hindi or Hinglish.

    Returns:
        English translation of the text (string).
        Falls back to original text if translation model is not loaded or fails.
    """
    if _translator_tokenizer is None or _translator_model is None:
        logger.warning("Translation model not loaded. Returning original text.")
        return text

    try:
        # Pre-translation replacements for words that Helsinki-NLP translates to vulgarities/profanities.
        # e.g. "बकवास" translates to "shit" which triggers Detoxify. We replace it with "बेकार" (useless).
        replacements = {
            "बकवास": "बेकार",
            "bakwaas": "useless",
            "bakwas": "useless",
        }
        for word, rep in replacements.items():
            if word == "बकवास":
                text = text.replace(word, rep)
            else:
                text = re.sub(rf'\b{word}\b', rep, text, flags=re.IGNORECASE)

        import torch

        # Step 1: Tokenize -- convert text string -> list of integer token IDs
        # return_tensors="pt" means return PyTorch tensors (the format the model needs)
        # padding=True adds zeros to make all inputs same length (batch processing)
        inputs = _translator_tokenizer(
            text,
            return_tensors="pt",   # "pt" = PyTorch
            padding=True,
            truncation=True,       # Cut text if longer than model's max length
            max_length=512         # Most transformer models have a 512 token limit
        )

        # Step 2: Generate translation token IDs
        # num_beams=4: Beam search -- explores 4 possible translations simultaneously
        with torch.no_grad():      # Disable gradient computation (we're not training)
            translated_ids = _translator_model.generate(
                **inputs,
                num_beams=4,
                max_length=512
            )

        # Step 3: Decode -- convert integer token IDs back to readable text
        # skip_special_tokens=True removes [PAD], [EOS] etc. from the output
        translated_text = _translator_tokenizer.decode(
            translated_ids[0],
            skip_special_tokens=True
        )

        # FIX: Use ASCII-safe log format to prevent Windows UnicodeEncodeError
        preview_in = text[:50] if len(text) > 50 else text
        preview_out = translated_text[:50] if len(translated_text) > 50 else translated_text
        logger.info("Translated [%.50s] -> [%.50s]", preview_in, preview_out)
        return translated_text

    except Exception as e:
        logger.error("Translation failed: %s", str(e))
        return text  # Fallback: return original text if translation fails


def score_toxicity(text: str) -> Dict[str, float]:
    """
    Run the Detoxify model on the given text and return toxicity scores.

    Detoxify returns scores in the range [0.0, 1.0] for each category:
        - toxicity:        General toxic content
        - severe_toxicity: Very aggressive, targeted toxic content
        - obscene:         Profanity, sexual content
        - threat:          Expressions of intent to harm
        - insult:          Belittling, demeaning language
        - identity_attack: Attacks based on race, religion, gender, etc.

    Args:
        text: English text to score.

    Returns:
        Dictionary of category -> float score.
    """
    if _detoxify_model is None:
        logger.error("Detoxify model not loaded!")
        return {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
            "threat": 0.0, "insult": 0.0, "identity_attack": 0.0
        }

    try:
        # predict() returns a dict like:
        # {'toxicity': tensor([0.87]), 'insult': tensor([0.72]), ...}
        raw_scores = _detoxify_model.predict(text)

        # Convert from PyTorch tensors to plain Python floats
        # Round to 4 decimal places for clean display
        scores = {}
        for key, val in raw_scores.items():
            try:
                # Handle both tensor and plain float outputs
                if hasattr(val, '__iter__'):
                    float_val = float(list(val)[0])
                else:
                    float_val = float(val)
                scores[key] = round(float_val, 4)
            except (TypeError, ValueError, IndexError):
                scores[key] = 0.0

        logger.info("Toxicity scores: %s", scores)
        return scores

    except Exception as e:
        logger.error("Detoxify scoring failed: %s", str(e))
        return {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
            "threat": 0.0, "insult": 0.0, "identity_attack": 0.0
        }


def get_moderation_action(toxicity_score: float) -> str:
    """
    Apply moderation rules based on the overall toxicity score.

    Rules:
        < 0.40           -> "APPROVED"  (safe to publish)
        0.40 - 0.69      -> "REVIEW"    (needs human review)
        >= 0.70          -> "BLOCKED"   (automatically rejected)

    Traffic light analogy:
        GREEN  (< 0.4)   -> Go ahead
        YELLOW (0.4-0.7) -> Slow down, review needed
        RED    (>= 0.7)  -> Stop, blocked

    Args:
        toxicity_score: The 'toxicity' score from Detoxify (0.0-1.0).

    Returns:
        One of: "APPROVED", "REVIEW", "BLOCKED"
    """
    if toxicity_score < 0.4:
        return "APPROVED"
    elif toxicity_score < 0.7:
        return "REVIEW"
    else:
        return "BLOCKED"


def analyze_comment(comment: str) -> Dict:
    """
    Master function that orchestrates the full moderation pipeline.

    Pipeline (Hybrid Abuse Detection Layer):
        ─────────────────────────────────────────────────────────────
        User Comment
              │
              ▼
        Text Normalization
              │
              ▼
        Hybrid Abuse Filter  ←── Stage 0 (NEW)
              │
        ┌─────┴─────┐
        │ Abuse     │ No Abuse
        │ Detected  │ Detected
        ▼           ▼
        BLOCK       Language Detection
        (skip       Translation (if needed)
        Detoxify)   Detoxify ML Scoring
                    Moderation Decision
        ─────────────────────────────────────────────────────────────

    Args:
        comment: The raw user-submitted comment string.

    Returns:
        A dictionary with all moderation details including:
            original_comment, translated_comment, detected_language,
            text_analyzed, toxicity_score, scores, action,
            matched_word, matched_rule, blocked_by
    """
    # ── Stage 0: Hybrid Abuse Filter ──────────────────────────────────────────
    # Run BEFORE any ML inference. If it fires, skip Detoxify entirely.
    # This ensures explicit abuse (madarchod, bhenchod, etc.) is always BLOCKED
    # regardless of how the ML model scores it.
    from app.abuse_filter import check_abuse

    abuse_result = check_abuse(comment)

    if abuse_result["blocked"]:
        # Abuse filter fired — block immediately, skip Detoxify
        logger.info(
            "[Pipeline] BLOCKED by Keyword Filter | word='%s' | rule='%s'",
            abuse_result["matched_word"],
            abuse_result["matched_rule"]
        )
        # Populate translated_comment if Hindi/Hinglish to satisfy API/test requirements
        detected_lang = detect_language(comment)
        translated_comment = None
        if detected_lang in ("hi", "hinglish"):
            try:
                translated_comment = translate_to_english(comment)
            except Exception:
                pass

        # Return a complete result dict with toxicity_score=1.0 (maximum)
        # Detoxify is NOT called.
        return {
            "original_comment": comment,
            "translated_comment": translated_comment,
            "detected_language": detected_lang,  # Still detect for logging
            "text_analyzed": comment,            # Original text was analyzed
            "toxicity_score": 1.0,               # Maximum score — explicit abuse
            "scores": {
                "toxicity": 1.0,
                "severe_toxicity": 1.0,
                "obscene": 1.0,
                "threat": 0.0,
                "insult": 1.0,
                "identity_attack": 0.0,
            },
            "action": "BLOCKED",
            "matched_word": abuse_result["matched_word"],
            "matched_rule": abuse_result["matched_rule"],
            "blocked_by": "Keyword Filter",
        }

    # ── Stage 1: Language Detection ───────────────────────────────────────────
    detected_lang = detect_language(comment)
    logger.info("Detected language: %s for: '%.60s'", detected_lang, comment)

    # ── Stage 2: Translation (if needed) ─────────────────────────────────────
    translated_comment = None   # Only set if translation happens
    text_to_score = comment     # Default: score the original text

    if detected_lang in ("hi", "hinglish"):
        # Translate Hindi/Hinglish → English before scoring
        translated_comment = translate_to_english(comment)
        text_to_score = translated_comment

    # ── Stage 3: Detoxify ML Scoring ─────────────────────────────────────────
    scores = score_toxicity(text_to_score)
    primary_toxicity_score = scores.get("toxicity", 0.0)

    # ── Stage 4: Moderation Decision ─────────────────────────────────────────
    action = get_moderation_action(primary_toxicity_score)

    # ── Stage 5: Build and return result ─────────────────────────────────────
    return {
        "original_comment": comment,
        "translated_comment": translated_comment,
        "detected_language": detected_lang,
        "text_analyzed": text_to_score,
        "toxicity_score": primary_toxicity_score,
        "scores": scores,
        "action": action,
        "matched_word": None,            # Abuse filter did not fire
        "matched_rule": None,
        "blocked_by": "Detoxify",        # Detoxify was the decision-maker
    }

