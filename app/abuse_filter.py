"""
app/abuse_filter.py — Enterprise-Grade Hybrid Abuse Detection Engine v2.0
==========================================================================

ARCHITECTURE — 5-Layer Detection Pipeline
------------------------------------------

  Raw Text
    │
    ├─ normalize_soft()       → [a-z\s] — leet applied, separators→space
    └─ normalize_stripped()   → [a-z]   — leet applied, everything removed,
                                           2+ repeated chars collapsed to 1

Detection passes (each returns on first match, short-circuits subsequent layers):

    Layer 1 │ REGEX      │ Compiled pattern per abuse term. Allows char repetition
            │            │ (char+) and any separators (\\s*) between characters.
            │            │ Applies word boundaries to prevent false positives.
            │            │ Applied to: normalize_soft(text)
            │            │ Catches: "m4darch0d", "m.a.d.a.r.c.h.o.d",
            │            │          "maaaadarchod", "madar-chod", "madar chod"
            │            │
    Layer 2 │ EXACT      │ Single compiled alternation regex of every pre-normalized
            │ STRIPPED   │ family variant (>= 7 chars). No boundary constraint —
            │            │ substring match on the fully stripped text.
            │            │ Applied to: normalize_stripped(text)
            │            │ Catches: variants after stripping all obfuscation chars
            │            │          e.g. "m...a...d...a...r...c...h...o...d"
            │            │
    Layer 3 │ FUZZY      │ difflib.SequenceMatcher on every token (word) AND every
            │            │ adjacent token pair from the soft-normalized text.
            │            │ Threshold: 0.82 similarity.
            │            │ Applied to: tokens from normalize_soft(text)
            │            │ Catches: "madarxhod", "mdarchod", "madhar4d",
            │            │          "madarxchod", novel misspellings
            │            │
    Layer 4 │ PHRASE     │ Compiled regex for multi-word compound abuse phrases.
            │            │ Applied to: normalize_soft(text)
            │            │
    Layer 5 │ CENSORED   │ Hardcoded patterns for heavily censored forms where
            │            │ internal letters are replaced by asterisks.
            │            │ Applied to: normalize_soft(text)

PERFORMANCE:
    All patterns compiled at import time. Zero per-request compilation.
    Typical latency: < 5ms for lexical detection on a 100-word comment.
    Fuzzy layer: O(tokens × families) ≈ 20 × 18 × 5μs ≈ 2ms

RETURN VALUE (check_abuse):
    {
        "blocked":        bool,
        "matched_family": str | None,   # Canonical family name (e.g. "madarchod")
        "matched_variant": str | None,  # Specific form that triggered (e.g. "m4darch0d")
        "matched_word":   str | None,   # Alias for matched_family (backward compat)
        "matched_rule":   str | None,   # Detection layer + detail
        "reason":         str | None,   # Human-readable explanation
        "confidence":     float,        # 1.0 for exact/regex, 0.xx for fuzzy
        "normalised_text": str,         # normalize_stripped(original) — what was seen
        "blocked_by":     str,          # "Hybrid Abuse Filter" or None
    }
"""

import re
import unicodedata
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Fuzzy matching threshold (0.0-1.0). Inputs with similarity >= this value
# against a canonical family form will be flagged.
# 0.82 catches: "madarxhod" (89%), "mdarchod" (94%), "madhar4d" (78% against
# "madharchod" — relies on "madharchod" variant being close enough).
FUZZY_THRESHOLD: float = 0.82

# Only apply exact-stripped substring matching for variants of at least this
# length, to prevent short abuse words ("gand", "lund") matching inside
# legitimate longer words ("gandhi", "lunar").
MIN_EXACT_STRIPPED_LEN: int = 7

# Minimum token length for fuzzy matching. Very short tokens are noise.
MIN_FUZZY_TOKEN_LEN: int = 4

# Minimum family canonical length for fuzzy matching.
MIN_FUZZY_CANONICAL_LEN: int = 5


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ABUSE FAMILY LEXICON
# ══════════════════════════════════════════════════════════════════════════════
#
# Each family is keyed by its CANONICAL name.
# Values are lists of KNOWN VARIANTS (including raw typed forms; they are
# normalized at startup and added to the exact lookup table).
#
# Why families instead of a flat list?
#   - Groups semantically related terms for better reporting
#   - Enables fuzzy matching against ALL variants, not just canonical
#   - Makes the lexicon maintainable and extensible
#
# Important: List every known typed variant (pre-normalization), including
# leetspeak forms like "m4darchod" and obfuscated forms like "madhar4d".
# The startup normalizer will collapse them all to their canonical stripped form.

ABUSE_FAMILIES: Dict[str, List[str]] = {

    # ── Madarchod family ───────────────────────────────────────────────────────
    # All motherfucker/maderchod equivalents and phonetic variants.
    "madarchod": [
        "madarchod", "maderchod", "madharchod", "maadarchod", "madarchood",
        "madarchut", "maderchut", "madarchodd", "makarchod", "maaderchod",
        "maderfucker", "maderfukr", "madarchodbc", "motherfucker", "motherphuker",
        "motherfukr", "mothrfkr",
        # Typed obfuscations (normalized at startup to their stripped forms)
        "madhar4d",          # → "madhared"  (common obfuscation)
        "m4darchod",         # → "madarchod"
        "m@darchod",         # → "madarchod"
        "madarxhod",         # → "madarxhod" (fuzzy catches this)
    ],

    # ── Bhenchod / Behenchod family ────────────────────────────────────────────
    "bhenchod": [
        "bhenchod", "behenchod", "bahenchod", "bhenchodd", "behanchod",
        "bhainchod", "betichod", "bhenchood", "bhenchud", "bhenchow",
        "bhaanchod", "makkarchod", "bhaenchod", "penchod", "pinchod",
        "penchelled", "pencho",
    ],

    # ── Chutiya family ─────────────────────────────────────────────────────────
    "chutiya": [
        "chutiya", "chutia", "chutiye", "chutiyaa", "chutiyapa", "chutmar",
        "chodu", "chudakkad",
        "chut1ya",           # → "chutiya"
    ],

    # ── Chut (standalone) ─────────────────────────────────────────────────────
    "chut": [
        "chut", "choot",
    ],

    # ── Bhosdike family ────────────────────────────────────────────────────────
    "bhosdike": [
        "bhosdike", "bhosdi", "bhosda", "bhosad", "bhosadike", "bhosdiwala",
        "bhosdiwale", "bhosadchod", "bsdk", "bhosdiki", "bhosdee", "bhosadika",
        "bhosadike", "bhosari", "bhosdik", "bhosdke",
        "bh0sdike",          # → "bhosdike"
        "bh0sdi",            # → "bhosdi"
    ],

    # ── Gaand family ──────────────────────────────────────────────────────────
    "gaand": [
        "gaand", "gand", "gandu", "gaandu", "gandfat", "gandfut",
        "gaandmar", "chuttad", "chutad", "gaanduu",
        "g@ndu",             # → "gandu"
        "g@@nd",             # → "gaand"
    ],

    # ── Lund family ───────────────────────────────────────────────────────────
    "lund": [
        "lund", "lauda", "loda", "laude", "lode", "laundey", "launda",
        "lounda", "lulli", "gadhalund",
    ],

    # ── Randi family ──────────────────────────────────────────────────────────
    "randi": [
        "randi", "rand", "randy", "randibaaz", "kanjari", "gashti", "randii",
    ],

    # ── Harami family ─────────────────────────────────────────────────────────
    "harami": [
        "harami", "haramzada", "haramjada", "haramkhor", "haraamjaade",
    ],

    # ── Bhadwa / Dalla family ─────────────────────────────────────────────────
    "bhadwa": [
        "bhadwa", "bhadva", "bhaduaa", "dalal", "dalla",
    ],

    # ── Kutta / Kutiya family ─────────────────────────────────────────────────
    "kutta": [
        "kutta", "kutiya", "kuttiya", "suar",
    ],

    # ── Bakchod family ────────────────────────────────────────────────────────
    "bakchod": [
        "bakchod", "bakchodi",
    ],

    # ── Kamina / Kameena family ───────────────────────────────────────────────
    "kamina": [
        "kamina", "kameena", "kaminey",
    ],

    # ── Fuddi family (Punjabi) ────────────────────────────────────────────────
    "fuddi": [
        "fuddi", "phuddi", "bulla",
    ],

    # ── Jhatu / Fattu family ──────────────────────────────────────────────────
    "jhatu": [
        "jhatu", "fattu",
    ],

    # ── Core abbreviations (treated as single-word families) ──────────────────
    "mc":   ["mc"],
    "bc":   ["bc"],
    "bkl":  ["bkl"],
    "bsdk": ["bsdk"],
}


# Words that do not belong to a semantic family but are standalone abuse terms.
# These are compiled into the regex pattern system individually.
STANDALONE_WORDS: Set[str] = {
    "bevda", "bewda", "bewakoof", "charsi", "saala", "saale", "gadha", "gadhe",
    "hijra", "napunsak", "tatte", "gote", "aand", "jhaat", "moot", "mut",
    "hag", "haggu", "paad", "peshab", "chooche", "choochi", "mammey", "bur", "burr",
    "laundiya", "loundiya",
}


# Multi-word compound abuse phrases.
# Sorted longest-first so longer phrases take priority over sub-phrases.
COMPOUND_PHRASES: List[str] = sorted([
    "teri maa ki chut", "teri behen ki chut", "tera baap ka lund",
    "maa ki chut mein", "gaand mein danda", "randi ka putra", "randi ka bacha",
    "madarchod bhenchod", "ullu ka pattha", "ullu ke pathe", "laude ke baal",
    "lund ke baal", "chut ke pasine", "bhosdi ke chutiye", "bhen ke laude",
    "behen ke laude", "bhen ke lode", "bhen ka loda", "lund chod",
    "maa ki chut", "maa ka lund", "bhen ki chut", "teri maa ki", "teri behen ki",
    "teri maa di", "tera baap", "bhosde mein", "bhosdi wale", "bhosdi da",
    "gaand da", "lund da", "maa dar chod", "madar chod", "behan chod",
    "maa chod", "beti chod", "saala kutta", "saali kutti", "kamine saale",
    "harami kutte", "gandu chutiya", "bkl chutiya", "bc mc", "chudwa le",
    "chudwane ka", "maro bc", "marunga mc", "kameena sala", "madarchod sala",
    "bhenchod sala", "chutiya sala", "ka bhosda", "ka lund", "bhaiya ke",
    "beti ke", "harami kutta", "bhen ke lode", "bhen ka loda", "bhen ke laude",
    "behen ke laude", "ullu ke pathe", "bhen ki chut",
], key=len, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Leetspeak substitution ────────────────────────────────────────────────────
#
# SAFE table: used in normalize_soft (word boundaries must be preserved).
# Does NOT include '!' because "madarchod!!!" → soft → "madarchod!!!" → ! stays
# as a non-alpha → replaced by space → "madarchod " → word boundary intact ✓
# If we mapped ! → i, "madarchod!!!" would become "madarchodiii" breaking \b.
#
# FULL table: used in normalize_stripped (no word boundaries).
# Includes '!' → 'i' because after stripping ALL non-alpha, "chut!ya" → "chutya"
# (which is still close enough to "chutiya" for fuzzy matching at 92%).
# Includes 8 → b, 9 → g (new in v2).

_LEET_SAFE = str.maketrans({
    '4': 'a', '@': 'a',
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '5': 's', '$': 's',
    '7': 't',
    '8': 'b',
    '9': 'g',
})

_LEET_FULL = str.maketrans({
    '4': 'a', '@': 'a',
    '0': 'o',
    '1': 'i',   '!': 'i',
    '3': 'e',
    '5': 's',   '$': 's',
    '7': 't',
    '8': 'b',
    '9': 'g',
})

# ── Unicode zero-width / invisible character removal ─────────────────────────
_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f'
    r'\u202a\u202b\u202c\u202d\u202e'
    r'\ufeff\u00ad\u034f\u2060\u2061\u2062\u2063\u2064]'
)

# ── Emoji removal (major Unicode ranges) ─────────────────────────────────────
_EMOJI_RE = re.compile(
    r'[\U0001F300-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F'
    r'\U0001FA70-\U0001FAFF'
    r'\U00002600-\U000027BF'
    r'\U0001F004-\U0001F0CF'
    r'\U00002702-\U000027B0'
    r'\uFE00-\uFE0F'
    r'\U0001F1E0-\U0001F1FF]',
    re.UNICODE
)

# ── Repeated char collapse (for stripped normalization) ───────────────────────
# Collapses 2+ consecutive identical characters to 1.
# "maaaad" → "mad",  "gaand" → "gand",  "madarchood" → "madarchod"
_REPEAT_COLLAPSE_RE = re.compile(r'(.)\1+')


def normalize_soft(text: str) -> str:
    """
    Phase 1 + Phase 2 normalization — preserves word boundaries.

    Transforms text so that the regex layer (with \\b word boundaries) can
    safely operate on it. Separators become spaces, leet chars become letters.

    Pipeline:
        1. Unicode NFKC (handles fullwidth, decomposed, ligatures)
        2. Strip zero-width / invisible chars
        3. Strip emojis
        4. Lowercase
        5. SAFE leet substitution (does NOT map ! → i to preserve boundaries)
        6. Non-[a-z] chars → space  (dots, dashes, *, !, etc. become spaces)
        7. Collapse multiple consecutive spaces

    Returns:
        String containing only [a-z] and single spaces.

    Examples:
        "m4darchod"          → "madarchod"
        "madar-chod"         → "madar chod"
        "m.a.d.a.r.c.h.o.d" → "m a d a r c h o d"
        "maaaadarchod!!!"    → "maaaadarchod"  (! becomes space, then trimmed)
        "g@ndu"              → "gandu"
    """
    text = unicodedata.normalize('NFKC', text)
    text = _ZERO_WIDTH_RE.sub('', text)
    text = _EMOJI_RE.sub('', text)
    text = text.lower()
    text = text.translate(_LEET_SAFE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_stripped(text: str) -> str:
    """
    Phase 1 + Phase 2 + Phase 1-ext normalization — maximum compression.

    Produces the most compact form of the text for substring matching and
    fuzzy comparison. All separators are removed, all repeated chars collapsed.

    Pipeline:
        1–4. Same as normalize_soft
        5. FULL leet substitution (includes ! → i)
        6. Remove ALL non-[a-z] chars (no spaces — everything stripped)
        7. Collapse 2+ consecutive identical chars to 1
           "maaaad" → "mad",  "madarchood" → "madarchod"

    Returns:
        String containing only [a-z], with no spaces and no repeated chars.

    Examples:
        "maaaadarchod"       → "madarchod"
        "m.a.d.a.r.c.h.o.d" → "madarchod"
        "m a d a r c h o d" → "madarchod"
        "madar-chod"         → "madarchod"
        "m4darchod"          → "madarchod"
        "m@darchod"          → "madarchod"
        "mad*rchod"          → "madrchod"   (fuzzy: 94% ≈ madarchod)
        "bh0sdike"           → "bhosdike"
        "g@ndu"              → "gandu"
        "chut!ya"            → "chutiya"    (! → i via FULL leet)
        "madarchood"         → "madarchod"  (double-o collapsed)
    """
    text = unicodedata.normalize('NFKC', text)
    text = _ZERO_WIDTH_RE.sub('', text)
    text = _EMOJI_RE.sub('', text)
    text = text.lower()
    text = text.translate(_LEET_FULL)
    text = re.sub(r'[^a-z]', '', text)           # Strip EVERYTHING non-alpha
    text = _REPEAT_COLLAPSE_RE.sub(r'\1', text)  # Collapse repeated chars
    return text


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PATTERN BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_word_regex(word: str) -> str:
    """
    Build a regex string for a single-word abuse term (Layer 1).

    Design:
        Each character expressed as `char+` (1+ repetitions — catches "maaaad").
        Characters separated by `\\s*` (0+ whitespace — catches "m a d a r c h o d").
        Word boundaries `\\b` prevent matching inside longer legitimate words.

    The pattern operates on normalize_soft() output, which only contains [a-z\\s].

    Examples:
        "madarchod" → \\bm+\\s*a+\\s*d+\\s*a+\\s*r+\\s*c+\\s*h+\\s*o+\\s*d+\\b
        "mc"        → \\bm+\\s*c+\\b

    What the pattern catches:
        "madarchod"              (exact) ✓
        "maaaadarchod"           (repeated chars) ✓  — via a+
        "m a d a r c h o d"     (spaces between chars) ✓  — via \\s*
        "m.a.d.a.r.c.h.o.d"    (dots → spaces after normalize) ✓
        "madar chod"             (space in middle) ✓
        "madar-chod"             (dash → space after normalize) ✓

    What it does NOT catch (handled by other layers):
        "madarxhod"   (wrong letter x) → fuzzy layer
        "mdarchod"    (missing letter) → fuzzy layer
    """
    chars = [re.escape(c) for c in word if c != ' ']
    inner = r'\s*'.join(f'{c}+' for c in chars)
    return r'\b' + inner + r'\b'


def _build_phrase_regex(phrase: str) -> str:
    """
    Build a regex string for a compound multi-word abuse phrase (Layer 4).

    Each word in the phrase is itself char+ with \\s* between chars.
    Words are separated by \\s+ (one or more whitespace).

    Example:
        "teri maa ki chut" →
        \\bt+\\s*e+\\s*r+\\s*i+\\s+m+\\s*a+\\s*a+\\s+k+\\s*i+\\s+c+\\s*h+\\s*u+\\s*t+\\b
    """
    words = phrase.split()
    word_pats = [
        r'\s*'.join(f'{re.escape(c)}+' for c in word)
        for word in words
    ]
    inner = r'\s+'.join(word_pats)
    return r'\b' + inner + r'\b'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STARTUP COMPILATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Layer 1: Word regex patterns ──────────────────────────────────────────────
# Each entry: (compiled_pattern, family_name)
_REGEX_PATTERNS: List[Tuple[re.Pattern, str]] = []

# ── Layer 2: Exact stripped substring matching ────────────────────────────────
# All long (>= MIN_EXACT_STRIPPED_LEN) normalized variants compiled into ONE
# alternation regex applied to the stripped text. This is much faster than
# iterating a dict — Python's regex engine is optimized for alternation.
_EXACT_LONG_RE: Optional[re.Pattern] = None
_EXACT_LONG_MAP: Dict[str, str] = {}     # normalized_variant → family_name

# ── Layer 3: Fuzzy match targets ──────────────────────────────────────────────
# Each entry: (family_name, normalized_canonical_of_family_variants)
# We include the normalized form of EACH variant (not just canonical) so that
# "madharchod" can be a fuzzy target in addition to "madarchod".
_FUZZY_TARGETS: List[Tuple[str, str]] = []

# ── Layer 4: Phrase patterns ──────────────────────────────────────────────────
_PHRASE_PATTERNS: List[Tuple[re.Pattern, str]] = []

# ── Layer 5: Censored patterns ────────────────────────────────────────────────
# Applied to normalize_soft() text. The censored characters become spaces,
# so patterns match the spaced-out remnants.
# Each entry: (compiled_pattern, canonical_family_name, rule_label)
_CENSORED_PATTERNS: List[Tuple[re.Pattern, str, str]] = []


def _compile_all_patterns() -> None:
    """
    Compile all detection patterns at module import time.

    This function runs ONCE. All compiled patterns are stored in module-level
    variables and reused for every subsequent check_abuse() call.

    Calling this at import time means:
        - Zero regex compilation cost per request
        - All patterns ready before the first HTTP request arrives
        - Any compilation errors surface immediately on server startup
    """
    global _REGEX_PATTERNS, _EXACT_LONG_RE, _EXACT_LONG_MAP
    global _FUZZY_TARGETS, _PHRASE_PATTERNS, _CENSORED_PATTERNS

    # ── Collect ALL unique words across families + standalone ─────────────────
    all_words: Set[str] = set()
    for variants in ABUSE_FAMILIES.values():
        all_words.update(variants)
    all_words.update(STANDALONE_WORDS)

    # ── Layer 1: Compile one regex pattern per unique word ────────────────────
    compiled_word_pats = []
    for word in all_words:
        # Only compile patterns for simple single-word terms (no spaces).
        # Phrases are handled by Layer 4.
        if ' ' in word:
            continue
        # Use the CANONICAL family name if the word belongs to a family,
        # otherwise use the word itself.
        family = _word_to_family(word)
        try:
            pat = re.compile(_build_word_regex(word), re.IGNORECASE)
            compiled_word_pats.append((pat, family))
        except re.error as exc:
            logger.warning("[AbuseFilter] Failed to compile regex for '%s': %s", word, exc)
    _REGEX_PATTERNS = compiled_word_pats
    logger.debug("[AbuseFilter] Compiled %d word regex patterns.", len(_REGEX_PATTERNS))

    # ── Layer 2: Build exact stripped lookup ──────────────────────────────────
    # Pre-normalize every variant from every family.
    # Store: normalized_form → family_name
    exact_map: Dict[str, str] = {}
    for family_name, variants in ABUSE_FAMILIES.items():
        for raw_variant in variants:
            norm = normalize_stripped(raw_variant)
            if norm and len(norm) >= MIN_EXACT_STRIPPED_LEN:
                exact_map[norm] = family_name
    for word in STANDALONE_WORDS:
        norm = normalize_stripped(word)
        if norm and len(norm) >= MIN_EXACT_STRIPPED_LEN:
            exact_map[norm] = word

    _EXACT_LONG_MAP = exact_map

    if exact_map:
        # Sort longest-first so longer patterns take priority in alternation.
        sorted_variants = sorted(exact_map.keys(), key=len, reverse=True)
        _EXACT_LONG_RE = re.compile(
            '|'.join(re.escape(v) for v in sorted_variants)
        )
    logger.debug("[AbuseFilter] Built exact stripped map: %d entries.", len(exact_map))

    # ── Layer 3: Build fuzzy targets ──────────────────────────────────────────
    # Include the normalized form of EACH variant as a fuzzy target,
    # so similar-to-variant inputs get caught (e.g., "madhared" ≈ "madharchod").
    seen_fuzzy: Set[str] = set()
    fuzzy_targets = []
    for family_name, variants in ABUSE_FAMILIES.items():
        for raw_variant in variants:
            norm = normalize_stripped(raw_variant)
            # Only use variants long enough to be distinctive
            if norm and len(norm) >= MIN_FUZZY_CANONICAL_LEN and norm not in seen_fuzzy:
                seen_fuzzy.add(norm)
                fuzzy_targets.append((family_name, norm))
    _FUZZY_TARGETS = fuzzy_targets
    logger.debug("[AbuseFilter] Built %d fuzzy targets.", len(_FUZZY_TARGETS))

    # ── Layer 4: Compile phrase patterns ──────────────────────────────────────
    compiled_phrase_pats = []
    for phrase in COMPOUND_PHRASES:
        try:
            pat = re.compile(_build_phrase_regex(phrase), re.IGNORECASE)
            compiled_phrase_pats.append((pat, phrase))
        except re.error as exc:
            logger.warning("[AbuseFilter] Failed to compile phrase regex for '%s': %s", phrase, exc)
    _PHRASE_PATTERNS = compiled_phrase_pats
    logger.debug("[AbuseFilter] Compiled %d phrase patterns.", len(_PHRASE_PATTERNS))

    # ── Layer 5: Censored patterns ────────────────────────────────────────────
    # After normalize_soft(), censored chars (* # ! etc.) become spaces.
    # Patterns match the space-separated remnants of heavily censored words.
    _CENSORED_PATTERNS = [
        # "m*****chod" → soft → "m     chod" → "m chod" after collapse
        (re.compile(r'\bm\s{1,8}chod\b', re.I), 'madarchod', 'censored_m_chod'),
        # "m*darchod" → soft → "m darchod"
        (re.compile(r'\bm\s+darchod\b', re.I), 'madarchod', 'censored_m_darchod'),
        # "bh****dike" → soft → "bh    dike"
        (re.compile(r'\bbh\s{1,6}(?:sd\s*)?(?:ike|di|dike)\b', re.I), 'bhosdike', 'censored_bh_dike'),
        # "bh*nchod" → soft → "bh nchod"
        (re.compile(r'\bbh\s*\w*\s+\w*chod\b', re.I), 'bhenchod', 'censored_bh_nchod'),
        # "ch*tiya" → soft → "ch tiya"
        (re.compile(r'\bch\s{0,5}(?:u\s*)?t\s*(?:i\s*y\s*a|oot?)\b', re.I), 'chutiya', 'censored_ch_tiya'),
        # "g**nd" → soft → "g nd"
        (re.compile(r'\bg\s+n?\s*d\b', re.I), 'gaand', 'censored_gaand_1'),
        (re.compile(r'\bg\s+a+\s*n\s*d\b', re.I), 'gaand', 'censored_gaand_2'),
        # "r*ndi" → soft → "r ndi"
        (re.compile(r'\br\s+(?:an?\s*)?(?:di|nd?i?)\b', re.I), 'randi', 'censored_randi'),
        # "l***d" → soft → "l nd" (or "l d")
        (re.compile(r'\bl\s{1,4}(?:n\s*)?d\b', re.I), 'lund', 'censored_lund'),
        # "m*c" → soft → "m c"
        (re.compile(r'\bm\s+c\b', re.I), 'mc', 'censored_mc'),
        # "b*c" → soft → "b c"
        (re.compile(r'\bb\s+c\b', re.I), 'bc', 'censored_bc'),
    ]
    logger.debug("[AbuseFilter] Compiled %d censored patterns.", len(_CENSORED_PATTERNS))


def _word_to_family(word: str) -> str:
    """
    Look up the canonical family name for a given variant word.
    Falls back to the word itself if not found in any family.
    """
    for family_name, variants in ABUSE_FAMILIES.items():
        if word in variants:
            return family_name
    return word  # Standalone word is its own family


# Run compilation immediately when the module is imported.
_compile_all_patterns()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DETECTION LAYERS
# ══════════════════════════════════════════════════════════════════════════════

def _match_regex(soft: str) -> Optional[Dict]:
    """
    Layer 1: Regex pattern matching on soft-normalized text.

    Checks every compiled single-word pattern against the soft-normalized
    input. Patterns use char+ (handles repeated chars) and \\s* between
    characters (handles separators). Word boundaries prevent false positives.

    Args:
        soft: normalize_soft(original_text)

    Returns:
        Match dict or None.
    """
    for pattern, family_name in _REGEX_PATTERNS:
        m = pattern.search(soft)
        if m:
            return {
                "family": family_name,
                "variant": m.group(),          # The actual matched string
                "rule": f"regex:{family_name}",
                "confidence": 1.0,
            }
    return None


def _match_exact_stripped(stripped: str) -> Optional[Dict]:
    """
    Layer 2: Exact substring matching on the fully stripped normalized text.

    The stripped text has ALL separators removed and ALL repeated chars
    collapsed. We check if any pre-normalized family variant (>= 7 chars)
    appears as a substring.

    This catches patterns that the regex layer might miss when abuse words
    are embedded in text without natural word boundaries, or when the
    normalization collapses unusual variants.

    Args:
        stripped: normalize_stripped(original_text)

    Returns:
        Match dict or None.
    """
    if _EXACT_LONG_RE is None:
        return None
    m = _EXACT_LONG_RE.search(stripped)
    if m:
        norm_variant = m.group()
        family_name = _EXACT_LONG_MAP.get(norm_variant, norm_variant)
        return {
            "family": family_name,
            "variant": norm_variant,
            "rule": f"exact_stripped:{norm_variant}",
            "confidence": 1.0,
        }
    return None


def _match_fuzzy(soft: str) -> Optional[Dict]:
    """
    Layer 3: Fuzzy string matching on individual tokens and adjacent pairs.

    Phase 3 implementation — uses difflib.SequenceMatcher to compare each
    word (token) in the soft-normalized text against pre-computed canonical
    normalized forms of each abuse family.

    Why token-level (not full-text)?
        - Social media comments are short; most abuse appears as standalone words
        - Avoids comparing "I hate this person madarchod" as a whole string
          against short abuse words (would give low similarity)
        - Adjacent-pair checking handles "madar chod" → combined "madarchod"

    Threshold: FUZZY_THRESHOLD (0.82)
        - At 0.82: "madarxhod" vs "madarchod" → 0.89 → MATCH  ✓
        - At 0.82: "mdarchod" vs "madarchod" → 0.94 → MATCH   ✓
        - At 0.82: "madhared" vs "madharchod" → 0.78 → close (handled by
          explicit "madhar4d" variant in the exact lookup)
        - Safe words: "documentation" vs any → < 0.70 → safe  ✓

    Args:
        soft: normalize_soft(original_text)

    Returns:
        Match dict or None.
    """
    tokens = soft.split()
    if not tokens:
        return None

    def _check_token(token: str) -> Optional[Dict]:
        if len(token) < MIN_FUZZY_TOKEN_LEN:
            return None
        best_ratio = 0.0
        best_family = None
        best_canonical = None
        for family_name, canonical_norm in _FUZZY_TARGETS:
            if len(canonical_norm) < MIN_FUZZY_CANONICAL_LEN:
                continue
            # Skip if length difference is too large (quick pre-filter)
            len_ratio = len(token) / len(canonical_norm)
            if len_ratio < 0.6 or len_ratio > 1.6:
                continue
            ratio = SequenceMatcher(None, token, canonical_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_family = family_name
                best_canonical = canonical_norm
        if best_ratio >= FUZZY_THRESHOLD and best_family:
            return {
                "family": best_family,
                "variant": token,
                "rule": f"fuzzy:{best_canonical}@{best_ratio:.2f}",
                "confidence": round(best_ratio, 3),
            }
        return None

    # Pass 1: Individual tokens
    for token in tokens:
        result = _check_token(token)
        if result:
            return result

    # Pass 2: Adjacent token pairs (handles "madar chod" → "madarchod")
    for i in range(len(tokens) - 1):
        pair = tokens[i] + tokens[i + 1]
        result = _check_token(pair)
        if result:
            return result

    return None


def _match_phrase(soft: str) -> Optional[Dict]:
    """
    Layer 4: Compound phrase matching on soft-normalized text.

    Checks multi-word abuse phrases. These cannot be caught by single-word
    patterns because they span multiple tokens.

    Args:
        soft: normalize_soft(original_text)

    Returns:
        Match dict or None.
    """
    for pattern, phrase in _PHRASE_PATTERNS:
        if pattern.search(soft):
            # Determine which family this phrase belongs to
            first_word = phrase.split()[0]
            family = _word_to_family(first_word)
            return {
                "family": family,
                "variant": phrase,
                "rule": f"phrase:{phrase}",
                "confidence": 1.0,
            }
    return None


def _match_censored(soft: str) -> Optional[Dict]:
    """
    Layer 5: Censored pattern matching.

    Catches heavily censored forms where core letters are replaced by
    asterisks or other punctuation:
        "m*c"          → mc
        "m***c"        → mc
        "m*****chod"   → madarchod
        "bh****dike"   → bhosdike
        "g**nd"        → gaand
        "r*ndi"        → randi

    After normalize_soft(), these censored chars become spaces, so we
    match the spaced-out remnants with specific patterns.

    Args:
        soft: normalize_soft(original_text)

    Returns:
        Match dict or None.
    """
    for pattern, family_name, rule_label in _CENSORED_PATTERNS:
        if pattern.search(soft):
            return {
                "family": family_name,
                "variant": family_name,
                "rule": f"censored:{rule_label}",
                "confidence": 1.0,
            }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def check_abuse(text: str) -> Dict:
    """
    Main entry point — run the full 5-layer hybrid abuse detection pipeline.

    Call this BEFORE the Detoxify ML model. If it returns blocked=True,
    skip Detoxify entirely and return BLOCKED with toxicity_score=1.0.

    Detection order (short-circuits on first match):
        1. Regex patterns      — handles separators, leet, repeated chars
        2. Exact stripped      — handles fully obfuscated variants (>= 7 chars)
        3. Fuzzy matching      — handles misspellings and novel variants
        4. Phrase patterns     — handles multi-word compound abuse
        5. Censored patterns   — handles m*c, bh****dike, g**nd etc.

    Args:
        text: Raw user-submitted comment text (any encoding/language).

    Returns:
        {
            "blocked":         bool    — True if abuse detected
            "matched_family":  str     — Canonical family name ("madarchod")
            "matched_variant": str     — The form that triggered ("m4darch0d")
            "matched_word":    str     — Alias for matched_family (backward compat)
            "matched_rule":    str     — Layer + detail ("regex:madarchod")
            "reason":          str     — Human-readable explanation
            "confidence":      float   — 1.0 for exact/regex, 0.xx for fuzzy
            "normalised_text": str     — normalize_stripped(text) (what we saw)
            "blocked_by":      str     — "Hybrid Abuse Filter" | None
        }

    False positive protection:
        - Word boundaries in regex layer: "gandhi" ≠ "gand", "chutney" ≠ "chut"
        - Fuzzy length ratio filter: very different lengths skip similarity check
        - Exact stripped only for variants >= 7 chars: avoids short-word substrings
    """
    if not text or not text.strip():
        return _safe_result("")

    # Compute both normalized forms once — reused across all layers
    soft = normalize_soft(text)
    stripped = normalize_stripped(text)

    if not soft:
        return _safe_result(stripped)

    # ── Layer 1: Regex (fastest, catches most cases) ──────────────────────────
    match = _match_regex(soft)

    # ── Layer 2: Phrase patterns (catches compound phrases) ───────────────────
    if not match:
        match = _match_phrase(soft)

    # ── Layer 3: Censored patterns (catches starred/dotted forms) ─────────────
    if not match:
        match = _match_censored(soft)

    # ── Layer 4: Exact stripped (catches complex obfuscations) ────────────────
    if not match:
        match = _match_exact_stripped(stripped)

    # ── Layer 5: Fuzzy (catches misspellings and novel variants) ─────────────
    if not match:
        match = _match_fuzzy(soft)

    if match:
        family  = match["family"]
        variant = match["variant"]
        rule    = match["rule"]
        conf    = match["confidence"]

        logger.info(
            "[AbuseFilter] BLOCKED | family='%s' | variant='%s' | rule='%s' | conf=%.2f",
            family, variant, rule, conf
        )
        return {
            "blocked":          True,
            "matched_family":   family,
            "matched_variant":  variant,
            "matched_word":     family,         # backward compat
            "matched_rule":     rule,
            "reason":           f"Matched abuse family '{family}' via {rule}",
            "confidence":       conf,
            "normalised_text":  stripped,
            "blocked_by":       "Hybrid Abuse Filter",
        }

    return _safe_result(stripped)


def _safe_result(normalised_text: str = "") -> Dict:
    """Return a standardized SAFE (not blocked) result."""
    return {
        "blocked":          False,
        "matched_family":   None,
        "matched_variant":  None,
        "matched_word":     None,
        "matched_rule":     None,
        "reason":           None,
        "confidence":       0.0,
        "normalised_text":  normalised_text,
        "blocked_by":       None,
    }


# Convenience alias for test suite compatibility
normalize = normalize_soft

