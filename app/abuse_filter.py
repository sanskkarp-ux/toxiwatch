"""
app/abuse_filter.py — Production-Grade Hybrid Lexical Abuse Detection Filter
=============================================================================
This module implements a multi-stage lexical abuse detector that runs BEFORE
the Detoxify ML model. If it fires, Detoxify is skipped entirely.

DETECTION PIPELINE (per request):
    1. Text Normalization
       - Unicode NFKC normalization
       - Zero-width / invisible character removal
       - Emoji removal
       - Lowercase
       - Leetspeak substitution (4→a, 0→o, 1→i, 3→e, 5→s, 7→t, @→a, $→s)
       - Non-alpha characters → space
       - Collapse repeated whitespace

    2. Stage A — Single-Word Pattern Matching (compiled regex)
       - Each abuse word compiled into a regex that allows:
           * 1+ repetitions of each character (maaaad → mad)
           * 0+ whitespace between characters (m a d a r c h o d)
           * Word boundaries (gandhi does NOT match gand)
       - Matches against normalized text

    3. Stage B — Compound Phrase Pattern Matching
       - Multi-word abuse phrases compiled into flexible regex patterns
       - Words separated by 1+ whitespace, chars allow repetitions

    4. Stage C — Censored Pattern Matching
       - Hardcoded patterns for heavily censored forms
       - e.g., m*****chod, bh****dike, m*c, b*c

PERFORMANCE:
    All patterns are compiled ONCE at module import time.
    Per-request cost: normalization (O(n)) + pattern scans (O(k·n))
    where k = number of patterns (~250), n = text length.
    Typical latency: < 2ms per request.

RETURN VALUE:
    {
        "blocked":      bool,
        "matched_word": str | None,   # The canonical abuse word matched
        "matched_rule": str | None,   # Which stage/rule triggered
        "reason":       str | None,   # Human-readable reason
        "confidence":   float         # 1.0 if blocked, 0.0 if safe
    }
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ABUSE LEXICON
# ══════════════════════════════════════════════════════════════════════════════

# Single-word abuse terms (no internal spaces).
# Used in Stage A pattern matching.
ABUSE_WORDS: set = {
    # ── Core abbreviations ─────────────────────────────────────────────────
    "mc", "bc", "bkl", "bsdk",

    # ── Madarchod family ───────────────────────────────────────────────────
    "madarchod", "madharchod", "maderchod", "madarchood", "madarchut",
    "maderchut", "madarchodd", "maadarchod", "makarchod", "maaderchod",
    "maderfucker", "maderfukr", "madarchodbc", "motherfucker", "motherphuker",
    "motherfukr", "mothrfkr",

    # ── Behenchod / Bhenchod family ────────────────────────────────────────
    "behenchod", "bhenchod", "bahenchod", "bhenchodd", "behanchod",
    "bhainchod", "betichod", "bhenchood", "bhenchud", "bhenchow",
    "bhaanchod", "makkarchod", "bhaenchod", "penchod", "pinchod",
    "penchelled", "pencho",

    # ── Chut / Bhosda / Gaand family ───────────────────────────────────────
    "chut", "choot", "chutiya", "chutia", "chutiye", "chutmar",
    "chutiyaa", "chutiyapa", "chodu", "chudakkad",
    "bhosda", "bhosdi", "bhosdike", "bhosad", "bhosadike",
    "bhosdiwala", "bhosdiwale", "bhosadchod", "bhosdiki", "bhosdee",
    "bhosadika", "bhosadike", "bhosari", "bhosdik", "bhosdke",
    "gaand", "gand", "gandu", "gaandu", "gandfat", "gandfut",
    "gaandmar", "chuttad", "chutad", "gaanduu",

    # ── Lund / Lauda family ────────────────────────────────────────────────
    "lund", "lauda", "loda", "laude", "lode", "laundey", "launda",
    "lounda", "lulli", "gadhalund",

    # ── Randi / Haramzada family ───────────────────────────────────────────
    "randi", "rand", "randy", "randibaaz", "kanjari", "gashti",
    "randii",

    # ── Harami / Kameena family ────────────────────────────────────────────
    "harami", "haramzada", "haramjada", "haramkhor", "haraamjaade",

    # ── Bhadwa / Kutte family ─────────────────────────────────────────────
    "bhadwa", "bhadva", "bhaduaa", "dalal", "dalla", "fattu",
    "kutta", "kutiya", "kuttiya", "suar",

    # ── Common insults ─────────────────────────────────────────────────────
    "bakchod", "bakchodi", "bevda", "bewda", "bewakoof", "charsi",
    "jhatu", "kamina", "kameena", "saala", "saale", "gadha", "gadhe",
    "hijra", "napunsak", "kaminey",

    # ── Regional / Punjabi-Haryanvi ───────────────────────────────────────
    "fuddi", "phuddi", "bulla",

    # ── Body-part terms used as abuse ──────────────────────────────────────
    "tatte", "gote", "aand", "jhaat", "moot", "mut", "hag", "haggu",
    "paad", "peshab", "chooche", "choochi", "mammey", "bur", "burr",
    "laundiya", "loundiya",

    # ── Phonetic / online spelling variants ───────────────────────────────
    "bhosri",  # bhosari variant
}


# Multi-word compound phrases used in Stage B pattern matching.
# Sorted longest-first so longer phrases match before shorter sub-phrases.
COMPOUND_PHRASES: List[str] = sorted([
    # ── Long phrases ──────────────────────────────────────────────────────
    "teri maa ki chut",
    "teri behen ki chut",
    "tera baap ka lund",
    "maa ki chut mein",
    "gaand mein danda",
    "randi ka putra",
    "randi ka bacha",
    "madarchod bhenchod",
    "ullu ka pattha",
    "ullu ke pathe",
    "laude ke baal",
    "lund ke baal",
    "chut ke pasine",
    "bhosdi ke chutiye",
    "bhen ke laude",
    "behen ke laude",
    "bhen ke lode",
    "bhen ka loda",
    "lund chod",

    # ── Medium phrases ────────────────────────────────────────────────────
    "maa ki chut",
    "maa ka lund",
    "bhen ki chut",
    "teri maa ki",
    "teri behen ki",
    "teri maa di",
    "tera baap",
    "bhen ke",
    "maa ke",
    "bhosde mein",
    "bhosdi wale",
    "bhosdi da",
    "gaand da",
    "lund da",
    "maa dar chod",
    "madar chod",
    "behan chod",
    "maa chod",
    "beti chod",

    # ── Short compound patterns ────────────────────────────────────────────
    "saala kutta",
    "saali kutti",
    "kamine saale",
    "harami kutte",
    "gandu chutiya",
    "bkl chutiya",
    "bc mc",
    "chudwa le",
    "chudwane ka",
    "maro bc",
    "marunga mc",
    "kameena sala",
    "madarchod sala",
    "bhenchod sala",
    "chutiya sala",
    "ka bhosda",
    "ka lund",
    "bhaiya ke",
    "beti ke",
    "saala kutta",
    "saali kutti",
    "harami kutta",
    "bhen ke lode",
    "bhen ka loda",
    "bhen ke laude",
    "behen ke laude",
    "ullu ke pathe",
], key=len, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# Leetspeak / symbol → letter substitution table.
# IMPORTANT: Only map characters that are UNAMBIGUOUSLY used as letter
# substitutes in evasion. Do NOT map: !, *, # (these are punctuation and
# should become spaces to preserve word boundaries).
_LEET_TABLE = str.maketrans({
    '4': 'a',
    '@': 'a',
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '5': 's',
    '$': 's',
    '7': 't',
    # NOTE: '!' is intentionally excluded — it is most commonly used as
    # punctuation (madarchod!!!) not as a letter substitute. If mapped to 'i',
    # it would extend the word and break word boundaries (madarchodiii).
    # '|': 'i' also excluded for same reason.
})

# Regex to strip zero-width / invisible Unicode characters.
_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f'
    r'\u202a\u202b\u202c\u202d\u202e'
    r'\ufeff\u00ad\u034f\u2060\u2061\u2062\u2063\u2064]'
)

# Regex to strip emoji (major Unicode emoji ranges).
_EMOJI_RE = re.compile(
    r'[\U0001F300-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F'
    r'\U0001FA70-\U0001FAFF'
    r'\U00002600-\U000027BF'
    r'\U0001F004-\U0001F0CF'
    r'\U00002702-\U000027B0'
    r'\uFE00-\uFE0F'          # Variation selectors
    r'\U0001F1E0-\U0001F1FF]', # Regional indicator symbols (flags)
    re.UNICODE
)

# Regex for extra punctuation at word edges ("madarchod!!!" → "madarchod")
_TRAILING_PUNCT_RE = re.compile(r'[^\w\s]+$|^[^\w\s]+', re.MULTILINE)


def normalize(text: str) -> str:
    """
    Normalize text for abuse pattern matching.

    Transformation pipeline:
        1. Unicode NFKC (handles fullwidth, decomposed, accented chars)
        2. Remove zero-width / invisible characters
        3. Remove emoji
        4. Lowercase
        5. Leetspeak substitution (4→a, 0→o, 1→i, 3→e, 5→s, 7→t, @→a, $→s)
        6. Replace ALL non-[a-z] characters with a space
           (handles: dots, dashes, underscores, *, #, !, @, $, etc.)
        7. Collapse multiple consecutive spaces into one

    Returns:
        Normalized string containing only [a-z] and single spaces.
    """
    # Step 1: Unicode normalization — handles accented, fullwidth, ligatures
    text = unicodedata.normalize('NFKC', text)

    # Step 2: Remove invisible / zero-width characters
    text = _ZERO_WIDTH_RE.sub('', text)

    # Step 3: Remove emoji
    text = _EMOJI_RE.sub('', text)

    # Step 4: Lowercase
    text = text.lower()

    # Step 5: Leetspeak substitution (single char at a time via translate)
    text = text.translate(_LEET_TABLE)

    # Step 6: Any character that is NOT [a-z] or space → space
    # This handles: dots, dashes, underscores, *, #, !, $, %, &, (, ), numbers, etc.
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Step 7: Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATTERN COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_word_regex(word: str) -> str:
    """
    Build a regex string for a single-word abuse term.

    Design:
        Each character in the word is expressed as `char+` (matches 1 or more
        repetitions of that char). Characters are separated by `\\s*` (matches
        zero or more whitespace characters).

    This makes the pattern match:
        - "madarchod"              (exact)
        - "maaaadarchod"           (repeated chars)
        - "madar chod"             (space between characters)
        - "m a d a r c h o d"     (space between every character)

    Word boundaries (\\b) prevent matching abuse words inside unrelated words:
        e.g., "gand" does NOT match "gandhi"
        e.g., "chut" does NOT match "chutney"
    """
    chars = [re.escape(c) for c in word]
    inner = r'\s*'.join(f'{c}+' for c in chars)
    return r'\b' + inner + r'\b'


def _build_phrase_regex(phrase: str) -> str:
    """
    Build a regex string for a compound abuse phrase.

    Each word in the phrase is itself expanded with char+ and \\s* between chars.
    Words in the phrase are separated by \\s+ (one or more spaces).
    """
    words = phrase.split()
    word_patterns = [
        r'\s*'.join(f'{re.escape(c)}+' for c in word)
        for word in words
    ]
    inner = r'\s+'.join(word_patterns)
    return r'\b' + inner + r'\b'


# ── Stage A: Single-word patterns ─────────────────────────────────────────────
# Each entry: (compiled_pattern, canonical_word)
_WORD_PATTERNS: List[Tuple[re.Pattern, str]] = []

# ── Stage B: Compound phrase patterns ────────────────────────────────────────
# Each entry: (compiled_pattern, canonical_phrase)
_PHRASE_PATTERNS: List[Tuple[re.Pattern, str]] = []

# ── Stage C: Censored patterns ────────────────────────────────────────────────
# Hardcoded patterns for heavily censored forms where core letters are replaced.
# e.g., m*****chod (the letters 'adar' are replaced by asterisks)
# These CANNOT be caught by the word patterns because core letters are missing.
# Applied to the NORMALIZED text.
# Each entry: (compiled_pattern, canonical_word, rule_name)
_CENSORED_PATTERNS: List[Tuple[re.Pattern, str, str]] = []


def _compile_all_patterns() -> None:
    """
    Compile all abuse patterns at module load time.
    Called once — never called again during request handling.
    """
    global _WORD_PATTERNS, _PHRASE_PATTERNS, _CENSORED_PATTERNS

    # ── Stage A: Single-word patterns ─────────────────────────────────────
    compiled_words = []
    for word in ABUSE_WORDS:
        try:
            pattern = re.compile(_build_word_regex(word), re.IGNORECASE)
            compiled_words.append((pattern, word))
        except re.error as exc:
            logger.warning("Failed to compile pattern for word '%s': %s", word, exc)
    _WORD_PATTERNS = compiled_words
    logger.debug("[AbuseFilter] Compiled %d single-word patterns.", len(_WORD_PATTERNS))

    # ── Stage B: Compound phrase patterns ─────────────────────────────────
    compiled_phrases = []
    for phrase in COMPOUND_PHRASES:
        try:
            pattern = re.compile(_build_phrase_regex(phrase), re.IGNORECASE)
            compiled_phrases.append((pattern, phrase))
        except re.error as exc:
            logger.warning("Failed to compile pattern for phrase '%s': %s", phrase, exc)
    _PHRASE_PATTERNS = compiled_phrases
    logger.debug("[AbuseFilter] Compiled %d compound phrase patterns.", len(_PHRASE_PATTERNS))

    # ── Stage C: Censored patterns ─────────────────────────────────────────
    # These match on NORMALIZED text (after normalize()) where separators
    # have already been replaced by spaces. Censored chars become spaces.
    # Pattern structure: first_letters + \s+ + remaining_letters
    # (the censored middle becomes a whitespace gap)
    _CENSORED_PATTERNS = [
        # m*****chod → "m     chod" in normalized text (letters 'adar' censored)
        (re.compile(r'\bm\s{1,8}chod\b', re.IGNORECASE), 'madarchod', 'censored_m_chod'),
        # m*darchod → "m darchod" ('a' censored)
        (re.compile(r'\bm\s+darchod\b', re.IGNORECASE), 'madarchod', 'censored_m_darchod'),
        # bh****dike / bh***di → "bh    dike"
        (re.compile(r'\bbh\s{1,6}(?:sd\s*)?(?:ike|di|dike)\b', re.IGNORECASE), 'bhosdike', 'censored_bh_dike'),
        # bh*nchod → "bh nchod" (the 'e' between bh and nchod is censored)
        # The pattern: bh + optional space + optional letters + chod
        (re.compile(r'\bbh\s*\w*\s+\w*chod\b', re.IGNORECASE), 'bhenchod', 'censored_bh_nchod'),
        # ch*tiya / ch***iya → "ch tiya"
        (re.compile(r'\bch\s{0,5}(?:u\s*)?t\s*(?:i\s*y\s*a|oot?)\b', re.IGNORECASE), 'chutiya', 'censored_ch_tiya'),
        # g**nd → "g nd", g**and → "g and" (the 'aa' between g and nd is censored)
        (re.compile(r'\bg\s+n?\s*d\b', re.IGNORECASE), 'gaand', 'censored_gaand_1'),
        (re.compile(r'\bg\s+a+\s*n\s*d\b', re.IGNORECASE), 'gaand', 'censored_gaand_2'),
        # r*ndi → "r ndi"
        (re.compile(r'\br\s+(?:an?\s*)?(?:di|nd?i?)\b', re.IGNORECASE), 'randi', 'censored_randi'),
        # l***d / l**nd → "l nd"
        (re.compile(r'\bl\s{1,4}(?:n\s*)?d\b', re.IGNORECASE), 'lund', 'censored_lund'),
    ]
    logger.debug("[AbuseFilter] Compiled %d censored patterns.", len(_CENSORED_PATTERNS))


# Run compilation immediately at module import — one-time cost.
_compile_all_patterns()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def check_abuse(text: str) -> Dict:
    """
    Run the full hybrid abuse detection pipeline against the input text.

    Args:
        text: Raw user-submitted comment text (any language, any encoding).

    Returns:
        A dictionary with the following keys:
            blocked      (bool)  — True if abuse was detected.
            matched_word (str)   — The canonical abuse word/phrase matched, or None.
            matched_rule (str)   — The detection rule that triggered, or None.
            reason       (str)   — Human-readable explanation, or None.
            confidence   (float) — 1.0 if blocked, 0.0 if safe.

    Detection Stages:
        Stage A: Single-word pattern matching (regex, word boundaries)
        Stage B: Compound phrase pattern matching
        Stage C: Censored pattern matching (e.g., m*c, m*****chod)
    """
    if not text or not text.strip():
        return _safe_result()

    # Normalize the text once — reused across all stages
    norm = normalize(text)

    if not norm:
        return _safe_result()

    # ── Stage A: Single-word patterns ─────────────────────────────────────
    for pattern, word in _WORD_PATTERNS:
        if pattern.search(norm):
            logger.info(
                "[AbuseFilter] BLOCKED by Stage A (word pattern) | word='%s' | input='%.80s'",
                word, text
            )
            return _block_result(
                matched_word=word,
                matched_rule=f"word_pattern:{word}",
                reason=f"Matched abuse term: '{word}'"
            )

    # ── Stage B: Compound phrase patterns ─────────────────────────────────
    for pattern, phrase in _PHRASE_PATTERNS:
        if pattern.search(norm):
            logger.info(
                "[AbuseFilter] BLOCKED by Stage B (phrase pattern) | phrase='%s' | input='%.80s'",
                phrase, text
            )
            return _block_result(
                matched_word=phrase,
                matched_rule=f"phrase_pattern:{phrase}",
                reason=f"Matched compound abuse phrase: '{phrase}'"
            )

    # ── Stage C: Censored patterns ─────────────────────────────────────────
    for pattern, word, rule in _CENSORED_PATTERNS:
        if pattern.search(norm):
            logger.info(
                "[AbuseFilter] BLOCKED by Stage C (censored pattern) | rule='%s' | input='%.80s'",
                rule, text
            )
            return _block_result(
                matched_word=word,
                matched_rule=f"censored:{rule}",
                reason=f"Matched censored abuse pattern ('{word}')"
            )

    # Nothing detected — safe to proceed to Detoxify
    return _safe_result()


def _block_result(matched_word: str, matched_rule: str, reason: str) -> Dict:
    """Return a standardized BLOCKED result."""
    return {
        "blocked": True,
        "matched_word": matched_word,
        "matched_rule": matched_rule,
        "reason": reason,
        "confidence": 1.0,
    }


def _safe_result() -> Dict:
    """Return a standardized SAFE (not blocked) result."""
    return {
        "blocked": False,
        "matched_word": None,
        "matched_rule": None,
        "reason": None,
        "confidence": 0.0,
    }
