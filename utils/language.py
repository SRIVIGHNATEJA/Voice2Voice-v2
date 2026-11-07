from difflib import get_close_matches


# ─────────────────────────────────────────────
# Canonical Indian language definitions
# ─────────────────────────────────────────────

# Code → human-readable name (used for ASR routing & logs)
INDIAN_LANG_MAP = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

# Code-only set (used for fast membership checks)
INDIAN_LANGS = set(INDIAN_LANG_MAP.keys())

# ─────────────────────────────────────────────
# Canonical Nature Scientific Reports (2026) Evaluation Sets
# Table 6 & Page 8:
#   L_Indic = {hi, te, ta, bn} (4 Principal Indic Languages)
#   L_Non_Indic = {fr, de, ar, ja} (4 Principal Non-Indic Languages)
#   LID & ASR Scope: Exactly 8 Principal Languages
#   Bridge Language: en (Common source/target for 16 NMT directions)
#   TTS Target Scope: Exactly 8 Target Languages (Table 24)
#   E2E S2ST Scope: Exactly 5 Directions into English (CVSS Table 22)
# ─────────────────────────────────────────────
# 1. Research Baseline Scope (Nature Scientific Reports 2026, 16:36511)
PAPER_PRINCIPAL_INDIC_LANGS = frozenset({"hi", "te", "ta", "bn"})
PAPER_PRINCIPAL_NON_INDIC_LANGS = frozenset({"fr", "de", "ar", "ja"})

# 8 Principal Languages for FLEURS LID & ASR evaluation (Eq. 1 candidate set L)
RESEARCH_LID_LANGS = PAPER_PRINCIPAL_INDIC_LANGS | PAPER_PRINCIPAL_NON_INDIC_LANGS
PAPER_LID_ASR_LANGS = RESEARCH_LID_LANGS

# 9 Languages for Runtime Automatic Source LID (8 FLEURS principal + English)
RUNTIME_LID_LANGS = RESEARCH_LID_LANGS | frozenset({"en"})

# 16 English-Centric NMT directions (8 to English, 8 from English)
RESEARCH_NMT_LANGS = RESEARCH_LID_LANGS | frozenset({"en"})
PAPER_NMT_LANGS = RESEARCH_NMT_LANGS

# 8 Target Languages for TTS Evaluation (Table 24)
PAPER_TTS_LANGS = RESEARCH_LID_LANGS

# Operating Routing Threshold (Nature Scientific Reports 2026, Eq. 3, Page 9)
RESEARCH_ROUTING_TAU = 0.70

# 5 CVSS End-to-End Speech-to-Speech Directions into English (Table 22)
PAPER_E2E_S2ST_DIRECTIONS = frozenset({"ta:en", "fr:en", "de:en", "ar:en", "ja:en"})

# Fast membership lookups
PAPER_INDIC_SET = set(PAPER_PRINCIPAL_INDIC_LANGS)
PAPER_NON_INDIC_SET = set(PAPER_PRINCIPAL_NON_INDIC_LANGS)
PAPER_LANG_SET = set(PAPER_NMT_LANGS)

# 2. Empirically Tested Directions with Verified Audio Fixtures in the Test Suite
# Strictly requires an actual spoken input waveform, ASR transcription, NMT translation,
# and synthesized output audio waveform verified in the automated regression suite.
EMPIRICALLY_TESTED_DIRECTIONS = frozenset({
    "te:en",  # Verified on input.wav (Telugu speech) -> English text -> OS System Voice
    "te:hi",  # Verified on input.wav (Telugu speech) -> Hindi text -> IndicParler-TTS (Neural)
})

# Architecture-supported dialogue pairs (28 registered directional combinations forming 14 reciprocal pairs)
# Formed strictly by English + 4 Principal Indic {hi, te, ta, bn} + 4 Principal Non-Indic {fr, de, ar, ja}
# Note: Spanish ('es') is supported as a single-turn target language via NLLB-200 and OS voice,
# but is not a member of the paper's 8 FLEURS languages and is not in RUNTIME_SUPPORTED_PAIRS.
RUNTIME_SUPPORTED_PAIRS = frozenset({
    # English <-> Indic (IndicParler-TTS for Indic, OS System Voice for English) - 8 directions
    "en:hi", "hi:en", "en:te", "te:en", "en:ta", "ta:en", "en:bn", "bn:en",
    # Indic <-> Indic (IndicParler-TTS for both) - 12 directions
    "hi:te", "te:hi", "hi:ta", "ta:hi", "hi:bn", "bn:hi",
    "te:ta", "ta:te", "te:bn", "bn:te", "ta:bn", "bn:ta",
    # English <-> Non-Indic (OS System Voice for both) - 8 directions
    "en:fr", "fr:en", "en:de", "de:en", "en:ar", "ar:en", "en:ja", "ja:en"
})

# Backward compatibility alias (deprecated: prefer RUNTIME_SUPPORTED_PAIRS)
VERIFIED_S2ST_PAIRS = RUNTIME_SUPPORTED_PAIRS


def is_paper_indic_language(lang_code: str) -> bool:
    """Checks if language belongs to the 4 canonical Indic languages evaluated in the paper."""
    return bool(lang_code and lang_code.strip().lower() in PAPER_PRINCIPAL_INDIC_LANGS)

def is_indic_language(lang_code: str) -> bool:
    """General utility: checks if language is in the broader Indian languages set (22 scheduled)."""
    return bool(lang_code and lang_code.strip().lower() in INDIAN_LANGS)

# ─────────────────────────────────────────────
# System Execution Configurations (Research vs. Runtime)
# ─────────────────────────────────────────────
RESEARCH_CONFIG = {
    "routing_threshold": 0.70,
    "fleurs_principal_languages": sorted(PAPER_LID_ASR_LANGS),
    "paper_indic_languages": sorted(PAPER_PRINCIPAL_INDIC_LANGS),
    "nmt_directions_count": 16,
    "cvss_directions": sorted(PAPER_E2E_S2ST_DIRECTIONS),
    "execution_mode": "reference"
}

RUNTIME_CONFIG = {
    "stage_and_flush": True,
    "persistent_session": False,
    "dialogue_pair_mode": True
}

# ─────────────────────────────────────────────
# Full Display Names Map (Indic + International)
# Guarantees full canonical names (e.g. 'en' -> 'English', 'te' -> 'Telugu')
# ─────────────────────────────────────────────
DISPLAY_LANG_MAP = {
    **INDIAN_LANG_MAP,
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ko": "Korean",
    "pt": "Portuguese",
    "it": "Italian",
    "si": "Sinhala",
}


def get_display_name(lang_code: str) -> str:
    """
    Returns the canonical English display name for any supported language code.
    Example: 'en' -> 'English', 'te' -> 'Telugu', 'hi' -> 'Hindi', 'fr' -> 'French'.
    """
    if not lang_code:
        return ""
    code_clean = lang_code.strip().lower()
    return DISPLAY_LANG_MAP.get(code_clean, code_clean.capitalize())


def validate_pair_syntax(pair_str: str) -> tuple[str, str]:
    """
    Validates and parses a dialogue pair string (e.g. 'en:te').
    Ensures:
      1. Correct delimiter syntax (exactly one ':').
      2. Both language codes are distinct (e.g., rejects 'en:en').
      3. Both language codes are known, valid ISO language codes.
      4. The directional pair is registered in RUNTIME_SUPPORTED_PAIRS (28 approved directions).
    Returns: (clean_lang1, clean_lang2)
    Raises: ValueError on invalid input syntax, unknown language codes, or unregistered pair.
    """
    if not pair_str or not isinstance(pair_str, str):
        raise ValueError("Dialogue pair string cannot be empty.")

    parts = [p.strip().lower() for p in pair_str.split(":")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid dialogue pair format '{pair_str}'. Expected format: '<src>:<tgt>' (e.g. 'en:te')."
        )

    lang1, lang2 = parts[0], parts[1]
    if lang1 == lang2:
        raise ValueError(
            f"Invalid dialogue pair '{pair_str}': source and target languages must be distinct."
        )

    valid_all = set(INDIAN_LANG_MAP.keys()) | set(DISPLAY_LANG_MAP.keys()) | set(RESEARCH_NMT_LANGS)
    for code in (lang1, lang2):
        if code not in valid_all:
            raise ValueError(
                f"Unsupported language code '{code}' in pair '{pair_str}'. "
                f"Must be a valid ISO language code (e.g. hi, te, ta, bn, en, fr, de, ar, ja)."
            )

    pair_directional = f"{lang1}:{lang2}"
    if pair_directional not in RUNTIME_SUPPORTED_PAIRS:
        raise ValueError(
            f"Unregistered dialogue pair '{pair_str}'. Must be one of the 28 registered directions in RUNTIME_SUPPORTED_PAIRS "
            f"(e.g. en:te, en:hi, hi:te, en:ta, en:bn, en:fr, en:de, en:ar, en:ja). "
            f"For individual unlisted languages like Spanish ('es'), use single-turn mode: -t es"
        )

    return lang1, lang2


def validate_target_language(target_str: str) -> str:
    """
    Validates a single-turn target language code (e.g. 'te', 'hi', 'en', 'es').
    """
    if not target_str or not isinstance(target_str, str):
        raise ValueError("Target language code cannot be empty.")
    code = target_str.strip().lower()
    valid_all = set(INDIAN_LANG_MAP.keys()) | set(DISPLAY_LANG_MAP.keys()) | set(RESEARCH_NMT_LANGS)
    if code not in valid_all:
        raise ValueError(
            f"Unsupported target language '{target_str}'. "
            f"Must be a recognized Indic or Western language code (e.g. te, hi, en, ta, bn, fr, de, es, ja, ar)."
        )
    return code


def validate_source_language(source_str: str) -> str:
    """
    Validates an explicitly locked source language code (e.g. 'te', 'hi', 'en').
    """
    if not source_str or not isinstance(source_str, str):
        raise ValueError("Source language code cannot be empty.")
    code = source_str.strip().lower()
    valid_all = set(INDIAN_LANG_MAP.keys()) | set(DISPLAY_LANG_MAP.keys()) | set(RESEARCH_NMT_LANGS)
    if code not in valid_all:
        raise ValueError(
            f"Unsupported source language '{source_str}'. "
            f"Must be a recognized Indic or Western language code (e.g. te, hi, en, ta, bn, fr, de, es, ja, ar)."
        )
    return code


def validate_language_arguments(source_lang: str = None, target_lang: str = None, pair_lang: str = None):
    """
    Performs comprehensive pre-execution validation across CLI / API arguments.
    Catches invalid language codes and contradictory flag combinations before
    allocating memory or loading neural models.
    """
    p1, p2 = None, None
    if pair_lang:
        p1, p2 = validate_pair_syntax(pair_lang)

    if source_lang:
        s = validate_source_language(source_lang)
        if pair_lang and s not in (p1, p2):
            raise ValueError(
                f"Contradictory arguments: locked source language '{source_lang}' is not a participant "
                f"in configured dialogue pair '{pair_lang}' (must be '{p1}' or '{p2}')."
            )

    if target_lang:
        t = validate_target_language(target_lang)
        if pair_lang and t not in (p1, p2):
            raise ValueError(
                f"Contradictory arguments: locked target language '{target_lang}' is not a participant "
                f"in configured dialogue pair '{pair_lang}' (must be '{p1}' or '{p2}')."
            )


# ─────────────────────────────────────────────
# Supported spoken language aliases & acoustic mishearings
# ─────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    # English
    "english": "en",
    "inglish": "en",
    "angrezi": "en",
    "en": "en",

    # Telugu
    "telugu": "te",
    "telegu": "te",
    "telgu": "te",
    "tell you": "te",
    "tell go": "te",
    "tilugu": "te",
    "talgu": "te",
    "te": "te",

    # Hindi
    "hindi": "hi",
    "hindee": "hi",
    "hindu": "hi",
    "hi": "hi",

    # Tamil
    "tamil": "ta",
    "tamizh": "ta",
    "thamizh": "ta",
    "tumil": "ta",
    "ta": "ta",

    # Kannada
    "kannada": "kn",
    "kanada": "kn",
    "kannad": "kn",
    "canada": "kn",
    "kn": "kn",

    # Malayalam
    "malayalam": "ml",
    "malyalam": "ml",
    "mallu": "ml",
    "ml": "ml",

    # Bengali
    "bengali": "bn",
    "bangla": "bn",
    "bengla": "bn",
    "bn": "bn",

    # Marathi
    "marathi": "mr",
    "marati": "mr",
    "mr": "mr",

    # Gujarati
    "gujarati": "gu",
    "gujrati": "gu",
    "gu": "gu",

    # Punjabi
    "punjabi": "pa",
    "panjabi": "pa",
    "pa": "pa",

    # International
    "german": "de",
    "deutsch": "de",
    "de": "de",
    "french": "fr",
    "francais": "fr",
    "fr": "fr",
    "spanish": "es",
    "espanol": "es",
    "es": "es",
    "japanese": "ja",
    "nihongo": "ja",
    "ja": "ja",
    "russian": "ru",
    "ru": "ru",
}


# ─────────────────────────────────────────────
# Language normalization utilities
# ─────────────────────────────────────────────

def normalize_language_input(text: str) -> str:
    """
    Normalizes spoken or typed language input by removing punctuation,
    filler words, and extra whitespace.
    """
    if not text:
        return ""
    return (
        text.lower()
        .replace("language", "")
        .replace("lang", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
        .replace("-", " ")
        .strip()
    )


def get_language_code(spoken_input: str):
    """
    Matches spoken/typed input to a supported language.
    1. Direct dictionary match (exact or normalized alias).
    2. Substring matching (e.g. "i want telugu" -> "telugu").
    3. Fuzzy Levenshtein match with difflib.
    Returns (lang_code, canonical_name) or (None, None).
    """
    normalized = normalize_language_input(spoken_input)
    if not normalized:
        return None, None

    # 1. Direct dictionary lookup
    if normalized in SUPPORTED_LANGUAGES:
        code = SUPPORTED_LANGUAGES[normalized]
        canonical = DISPLAY_LANG_MAP.get(code, normalized.capitalize())
        return code, canonical

    # 2. Substring token scan (e.g. "translate to telugu please")
    words = normalized.split()
    for w in words:
        if w in SUPPORTED_LANGUAGES:
            code = SUPPORTED_LANGUAGES[w]
            canonical = DISPLAY_LANG_MAP.get(code, w.capitalize())
            return code, canonical

    # 3. Fuzzy match fallback
    matches = get_close_matches(
        normalized,
        SUPPORTED_LANGUAGES.keys(),
        n=1,
        cutoff=0.45
    )

    if matches:
        lang_alias = matches[0]
        code = SUPPORTED_LANGUAGES[lang_alias]
        canonical = DISPLAY_LANG_MAP.get(code, lang_alias.capitalize())
        return code, canonical

    return None, None


def detect_target_language_manually(max_attempts=3):
    """
    Interactive manual fallback for target language selection with retry and suggestion loop.
    Allows typing 'exit' or 'q' to cancel gracefully.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"🔡 Please type target language name (Attempt {attempt}/{max_attempts}, or 'exit' to cancel):")
        try:
            typed_input = input("Your input: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ Input cancelled by user.")
            return None, None

        if not typed_input or typed_input.lower() in {"exit", "quit", "q", "cancel"}:
            print("🚪 Language selection cancelled.")
            return None, None

        code, name = get_language_code(typed_input)
        if code:
            return code, name

        print("⚠️ Not recognized.")
        suggestions = get_close_matches(
            normalize_language_input(typed_input),
            SUPPORTED_LANGUAGES.keys(),
            n=3
        )
        if suggestions:
            print("🔎 Did you mean:", ", ".join(s.capitalize() for s in suggestions))

    print("❌ Maximum target language selection attempts reached.")
    return None, None


# ─────────────────────────────────────────────
# Verified NLLB language codes
# ─────────────────────────────────────────────

NLLB_LANG_CODE_MAP = {
    # 22 Scheduled Indian Languages
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "gom_Deva",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Beng",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",

    # International Languages & Others
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "si": "sin_Sinh",
}
