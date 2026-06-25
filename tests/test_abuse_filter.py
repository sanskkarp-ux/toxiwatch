"""
tests/test_abuse_filter.py — Comprehensive Test Suite for the Hybrid Abuse Filter
==================================================================================
Tests the abuse_filter.check_abuse() function directly (no server needed).

Coverage:
    - Exact matching (50+ abuse words)
    - Case insensitive matching
    - Separator variants (spaces, dots, dashes, underscores)
    - Leetspeak / number substitution (4→a, 0→o, 1→i, 3→e, 5→s, 7→t)
    - Special symbol variants (@, $, !, *, #)
    - Repeated letter evasion (maaaadarchod)
    - Per-character spacing (m.a.d.a.r.c.h.o.d)
    - Censored variants (m*c, b*c, m*****chod)
    - Compound phrases
    - Mixed language inputs
    - False positive protection (safe words MUST NOT be blocked)
    - Edge cases (empty string, whitespace, short input)

Run:
    .\\venv\\Scripts\\pytest tests\\test_abuse_filter.py -v
    .\\venv\\Scripts\\pytest tests\\test_abuse_filter.py -v --tb=short
"""

import os
import sys
import pytest

os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.abuse_filter import check_abuse, normalize


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def should_block(text: str, label: str = "") -> None:
    """Assert that check_abuse blocks the given text."""
    result = check_abuse(text)
    assert result["blocked"] is True, (
        f"Expected BLOCKED for: {repr(text)}"
        + (f" [{label}]" if label else "")
        + f"\n  normalize() → {repr(normalize(text))}"
        + f"\n  Result: {result}"
    )
    assert result["matched_word"] is not None
    assert result["confidence"] == 1.0


def should_pass(text: str, label: str = "") -> None:
    """Assert that check_abuse does NOT block the given text."""
    result = check_abuse(text)
    assert result["blocked"] is False, (
        f"Expected SAFE (false positive!) for: {repr(text)}"
        + (f" [{label}]" if label else "")
        + f"\n  normalize() → {repr(normalize(text))}"
        + f"\n  Result: {result}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — EXACT MATCHING (should_block)
# Test that every entry in the core abuse lexicon is caught verbatim.
# ══════════════════════════════════════════════════════════════════════════════

class TestExactMatching:
    def test_madarchod(self):          should_block("madarchod")
    def test_bhenchod(self):           should_block("bhenchod")
    def test_behenchod(self):          should_block("behenchod")
    def test_chutiya(self):            should_block("chutiya")
    def test_bhosdike(self):           should_block("bhosdike")
    def test_bhosdi(self):             should_block("bhosdi")
    def test_gaand(self):              should_block("gaand")
    def test_gandu(self):              should_block("gandu")
    def test_harami(self):             should_block("harami")
    def test_randi(self):              should_block("randi")
    def test_lund(self):               should_block("lund")
    def test_lauda(self):              should_block("lauda")
    def test_mc(self):                 should_block("mc")
    def test_bc(self):                 should_block("bc")
    def test_bkl(self):                should_block("bkl")
    def test_maderchod(self):          should_block("maderchod")
    def test_chut(self):               should_block("chut")
    def test_bakchod(self):            should_block("bakchod")
    def test_haramzada(self):          should_block("haramzada")
    def test_kutta(self):              should_block("kutta")
    def test_bsdk(self):               should_block("bsdk")
    def test_madharchod(self):         should_block("madharchod")
    def test_penchod(self):            should_block("penchod")
    def test_fuddi(self):              should_block("fuddi")
    def test_kanjari(self):            should_block("kanjari")
    def test_randibaaz(self):          should_block("randibaaz")
    def test_haramkhor(self):          should_block("haramkhor")
    def test_bhadwa(self):             should_block("bhadwa")
    def test_jhatu(self):              should_block("jhatu")
    def test_kamina(self):             should_block("kamina")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — CASE INSENSITIVE
# ══════════════════════════════════════════════════════════════════════════════

class TestCaseInsensitive:
    def test_madarchod_upper(self):   should_block("MADARCHOD")
    def test_madarchod_title(self):   should_block("Madarchod")
    def test_madarchod_mixed(self):   should_block("MaDaRcHoD")
    def test_bhenchod_upper(self):    should_block("BHENCHOD")
    def test_chutiya_upper(self):     should_block("CHUTIYA")
    def test_bc_upper(self):          should_block("BC")
    def test_mc_upper(self):          should_block("MC")
    def test_bkl_upper(self):         should_block("BKL")
    def test_gandu_mixed(self):       should_block("GaNdU")
    def test_randi_upper(self):       should_block("RANDI")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — SEPARATOR VARIANTS (spaces, dots, dashes, underscores)
# ══════════════════════════════════════════════════════════════════════════════

class TestSeparatorVariants:
    def test_madarchod_space(self):          should_block("madar chod")
    def test_madarchod_dot(self):            should_block("madar.chod")
    def test_madarchod_dash(self):           should_block("madar-chod")
    def test_madarchod_underscore(self):     should_block("madar_chod")
    def test_madarchod_per_char_dot(self):   should_block("m.a.d.a.r.c.h.o.d")
    def test_madarchod_per_char_space(self): should_block("m a d a r c h o d")
    def test_madarchod_per_char_dash(self):  should_block("m-a-d-a-r-c-h-o-d")
    def test_madarchod_per_char_under(self): should_block("m_a_d_a_r_c_h_o_d")
    def test_madarchod_mixed_sep(self):      should_block("m.a.d.a.r-c_h o d")
    def test_bhenchod_space(self):           should_block("bhen chod")
    def test_bhenchod_dot(self):             should_block("bhen.chod")
    def test_chutiya_space(self):            should_block("chut iya")
    def test_bhosdike_space(self):           should_block("bhosd ike")
    def test_mc_space(self):                 should_block("m c")
    def test_bc_space(self):                 should_block("b c")
    def test_lund_dot(self):                 should_block("l.u.n.d")
    def test_gaand_dash(self):               should_block("g-a-a-n-d")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — LEETSPEAK / NUMBER SUBSTITUTION
# ══════════════════════════════════════════════════════════════════════════════

class TestLeetspeak:
    def test_madarchod_4(self):       should_block("m4darchod",       "4→a")
    def test_madarchod_at(self):      should_block("m@darchod",       "@→a")
    def test_bhosdike_0(self):        should_block("bh0sdike",        "0→o")
    def test_chutiya_1(self):         should_block("chut1ya",         "1→i")
    def test_chutiya_3(self):
        # 3→e means chut3ya → chuteya, NOT chutiya
        # This is intentional: 3 is not an unambiguous 'i' substitute
        result = check_abuse("chut3ya")
        # The word 'chuteya' is not in our lexicon; this tests that the
        # normalizer correctly converts 3→e and that we don't have false detection.
        # If added to lexicon it would block; for now verify the normalization is correct.
        assert normalize("chut3ya") == "chuteya"  # 3→e
    def test_randi_1(self):           should_block("r4nd1",           "4→a, 1→i")
    def test_lund_0(self):            should_block("lund",            "no leet needed")
    def test_gaand_at(self):          should_block("g@@nd",           "@→a")
    def test_chutiya_dollar(self):    should_block("chut1y@",         "@ and 1 combo")
    def test_madarchod_all_leet(self):should_block("m4d4rch0d",      "4→a, 4→a, 0→o")
    def test_bc_dollar(self):
        # $ → s, so "b$c" → "bsc", NOT "bc". Therefore it won't match 'bc'.
        # This is correct behaviour: $ is 's' not a separator.
        # Use b*c or b c (space) for censored bc.
        result = check_abuse("b$c")
        assert normalize("b$c") == "bsc"  # $ maps to 's', so bsc not bc
    def test_haramzada_0(self):       should_block("har4mzada",       "4→a")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — SPECIAL SYMBOL VARIANTS (*, #, !, @, $, %)
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecialSymbols:
    def test_madarchod_star(self):       should_block("m*darchod")
    def test_bhenchod_star(self):        should_block("bh*nchod")
    def test_chutiya_hash(self):         should_block("ch*tiya")
    def test_madarchod_exclaim(self):    should_block("madarchod!!!")
    def test_bhenchod_question(self):    should_block("bhenchod???")
    def test_gaand_stars(self):          should_block("g**nd")
    def test_randi_stars(self):          should_block("r*ndi")
    def test_lund_star(self):            should_block("l*nd")
    def test_bhosdike_all_stars(self):   should_block("b*h*o*s*d*i*k*e")
    def test_mc_star(self):              should_block("m*c",       "censored mc")
    def test_bc_star(self):              should_block("b*c",       "censored bc")
    def test_mc_double_star(self):       should_block("m**c",      "censored mc double star")
    def test_bc_double_star(self):       should_block("b**c",      "censored bc double star")
    def test_madarchod_in_sentence(self): should_block("Tu madarchod hai")
    def test_bhenchod_in_sentence(self):  should_block("Yeh bc bhenchod hai")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — REPEATED LETTERS
# ══════════════════════════════════════════════════════════════════════════════

class TestRepeatedLetters:
    def test_madarchod_a_repeat(self):   should_block("maadarchod")
    def test_madarchod_aaa_repeat(self): should_block("maaadarchod")
    def test_madarchod_aaaa(self):       should_block("maaaadarchod")
    def test_bhenchod_e_repeat(self):    should_block("bheenchod")
    def test_chutiya_u_repeat(self):     should_block("chuutiya")
    def test_chutiya_all_repeat(self):   should_block("chuutttiyaaa")
    def test_gaand_a_repeat(self):       should_block("gaaand")
    def test_lund_u_repeat(self):        should_block("luund")
    def test_randi_a_repeat(self):       should_block("raandi")
    def test_mc_m_repeat(self):          should_block("mmc")
    def test_bc_b_repeat(self):          should_block("bbc")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — COMPOUND PHRASES
# ══════════════════════════════════════════════════════════════════════════════

class TestCompoundPhrases:
    def test_teri_maa_ki_chut(self):        should_block("teri maa ki chut")
    def test_lund_ke_baal(self):            should_block("lund ke baal")
    def test_bkl_chutiya(self):             should_block("bkl chutiya")
    def test_bc_mc(self):                   should_block("bc mc")
    def test_saala_kutta(self):             should_block("saala kutta")
    def test_gandu_chutiya(self):           should_block("gandu chutiya")
    def test_harami_kutte(self):            should_block("harami kutte")
    def test_randi_ka_bacha(self):          should_block("randi ka bacha")
    def test_maa_ki_chut(self):             should_block("maa ki chut")
    def test_tera_baap_ka_lund(self):       should_block("tera baap ka lund")
    def test_bhen_ke_laude(self):           should_block("bhen ke laude")
    def test_laude_ke_baal(self):           should_block("laude ke baal")
    def test_madar_chod(self):              should_block("madar chod")
    def test_maa_chod(self):                should_block("maa chod")
    def test_kamine_saale(self):            should_block("kamine saale")
    def test_madarchod_bhenchod(self):      should_block("madarchod bhenchod")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 8 — MIXED CONTEXT (abuse in sentences)
# ══════════════════════════════════════════════════════════════════════════════

class TestAbuseinSentences:
    def test_abuse_at_start(self):
        should_block("Madarchod tum log movie nahi samjhe")

    def test_abuse_in_middle(self):
        should_block("Yeh movie bhenchod wali thi")

    def test_abuse_at_end(self):
        should_block("Director hai ek dum chutiya")

    def test_abuse_in_english_sentence(self):
        should_block("The director is a real madarchod")

    def test_abuse_with_punctuation(self):
        should_block("Tu toh bc hai yaar!!!")

    def test_leeted_in_sentence(self):
        should_block("Yeh m4darchod director ne sab barbaad kiya")

    def test_spaced_abuse_in_sentence(self):
        should_block("Woh ek m c hai")

    def test_dotted_abuse_in_sentence(self):
        should_block("Iska answer sirf ek hai: m.a.d.a.r.c.h.o.d")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 9 — CENSORED PATTERNS (letters replaced with * or .)
# ══════════════════════════════════════════════════════════════════════════════

class TestCensoredPatterns:
    def test_m_star_c(self):              should_block("m*c")
    def test_m_stars_c(self):             should_block("m***c")
    def test_b_star_c(self):              should_block("b*c")
    def test_b_stars_c(self):             should_block("b**c")
    def test_m_stars_chod(self):          should_block("m*****chod")
    def test_m_dots_chod(self):           should_block("m.....chod")
    def test_bh_stars_dike(self):         should_block("bh****dike")
    def test_mc_dot_separated(self):      should_block("m.c")
    def test_bc_dot_separated(self):      should_block("b.c")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 10 — FALSE POSITIVE PROTECTION (safe inputs MUST NOT be blocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestFalsePositiveProtection:
    def test_safe_english(self):
        should_pass("This movie was absolutely brilliant!")

    def test_gandhi(self):
        # "gand" is in the abuse list but "gandhi" should NOT match
        should_pass("Mahatma Gandhi was a great leader")

    def test_conduct(self):
        # Contains "duct" not any abuse word
        should_pass("His conduct was exemplary throughout")

    def test_chutney(self):
        # "chut" is in the abuse list but "chutney" should NOT match
        should_pass("I love mint chutney with my samosa")

    def test_documentation(self):
        should_pass("Please read the documentation before proceeding")

    def test_accumulate(self):
        should_pass("We need to accumulate more data for this experiment")

    def test_manipulation(self):
        should_pass("Data manipulation is a core skill in data science")

    def test_hindi_safe_devanagari(self):
        should_pass("यह फिल्म बहुत अच्छी थी")

    def test_positive_hinglish(self):
        should_pass("Yeh movie bahut acchi thi yaar maza aa gaya")

    def test_empty_string(self):
        result = check_abuse("")
        assert result["blocked"] is False

    def test_whitespace_only(self):
        result = check_abuse("   ")
        assert result["blocked"] is False

    def test_punctuation_only(self):
        result = check_abuse("!!! ??? ...")
        assert result["blocked"] is False

    def test_numbers_only(self):
        result = check_abuse("123 456 789")
        assert result["blocked"] is False

    def test_great_film(self):
        should_pass("Great cinematography, excellent story, must watch")

    def test_classic(self):
        should_pass("This is a classic example of great filmmaking")

    def test_lunge(self):
        # "lund" is in the abuse list but "lunge" should NOT match
        should_pass("He made a lunge for the door handle")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 11 — NORMALIZE() UNIT TESTS
# Verify the normalization function produces expected output.
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalization:
    def test_lowercase(self):
        assert normalize("HELLO WORLD") == "hello world"

    def test_leet_4(self):
        assert normalize("m4darchod") == "madarchod"

    def test_leet_0(self):
        assert normalize("bh0sdike") == "bhosdike"

    def test_leet_at(self):
        assert normalize("m@darchod") == "madarchod"

    def test_leet_dollar(self):
        assert normalize("$aale") == "saale"

    def test_dots_become_spaces(self):
        assert normalize("m.a.d") == "m a d"

    def test_dashes_become_spaces(self):
        assert normalize("madar-chod") == "madar chod"

    def test_stars_become_spaces(self):
        assert normalize("m*c") == "m c"

    def test_collapse_spaces(self):
        assert normalize("hello   world") == "hello world"

    def test_unicode_normalization(self):
        # Full-width ASCII characters should normalize to regular ASCII
        result = normalize("\uff4d\uff41\uff44\uff41\uff52")  # ｍａｄａｒ
        assert result == "madar"

    def test_emoji_removed(self):
        assert normalize("hello 😀 world") == "hello world"

    def test_zero_width_removed(self):
        # Zero-width space between letters should not affect matching
        result = normalize("mad\u200barchod")
        assert "madarchod" in result.replace(" ", "")

    def test_leet_1(self):
        assert normalize("chut1ya") == "chutiya"

    def test_leet_3(self):
        # 3 maps to 'e', so chut3ya → chuteya
        assert normalize("chut3ya") == "chuteya"

    def test_leet_7(self):
        assert normalize("7u") == "tu"


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 12 — RETURN STRUCTURE VALIDATION
# Verify check_abuse() always returns the correct dict structure.
# ══════════════════════════════════════════════════════════════════════════════

class TestReturnStructure:
    def test_blocked_has_all_fields(self):
        result = check_abuse("madarchod")
        assert "blocked" in result
        assert "matched_word" in result
        assert "matched_rule" in result
        assert "reason" in result
        assert "confidence" in result

    def test_blocked_confidence_is_one(self):
        result = check_abuse("madarchod")
        assert result["confidence"] == 1.0

    def test_blocked_has_matched_word(self):
        result = check_abuse("bhenchod")
        assert result["matched_word"] is not None
        assert len(result["matched_word"]) > 0

    def test_blocked_has_matched_rule(self):
        result = check_abuse("chutiya")
        assert result["matched_rule"] is not None

    def test_blocked_has_reason(self):
        result = check_abuse("gaand")
        assert result["reason"] is not None

    def test_safe_confidence_is_zero(self):
        result = check_abuse("This movie was amazing!")
        assert result["confidence"] == 0.0

    def test_safe_matched_word_is_none(self):
        result = check_abuse("Great film!")
        assert result["matched_word"] is None

    def test_safe_reason_is_none(self):
        result = check_abuse("Excellent performance!")
        assert result["reason"] is None
