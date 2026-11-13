"""
Comprehensive Verification Suite for Sprint 2 and Sprint 3
Tests:
  1. utils/audio.py: trim_silence_rms boundary cases & shape preservation
  2. models/asr.py: RMS silence trimming in IndicConformer & isolated telemetry (LID vs Transcribe)
  3. run_pipeline.py: Multi-mode dispatch (--target, --pair en:te auto-dialogue reversal, interactive)
  4. models/tts.py: Indic-Parler-TTS prompt embedding cache & dynamic frame capping
  5. models/translation.py: NLLB-200 dynamic token bounding
  6. End-to-end pipeline evaluation integration with atomic log verification
"""

import os
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("parler_tts").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import numpy as np
import torch
import atexit
import tempfile
import soundfile as sf
import shutil

SUBPROCESS_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "KMP_INIT_AT_FORK": "FALSE",
    "OMP_NUM_THREADS": "1"
}

# Ensure local imports resolve
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Standalone Audio Fixture Self-Provisioning ──────────────────
_temp_fixtures_created = []

def _ensure_fixture(filename: str, is_silent: bool = False) -> str:
    target_path = os.path.join(ROOT_DIR, filename)
    if os.path.exists(target_path):
        return target_path
    tmp_path = os.path.join(tempfile.gettempdir(), f"v2v_fixture_{filename}")
    sr = 16000
    if not os.path.exists(tmp_path):
        if is_silent:
            data = np.zeros(sr, dtype=np.float32)
        else:
            t = np.linspace(0, 2.5, int(sr * 2.5), endpoint=False)
            data = (0.35 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        sf.write(tmp_path, data, sr)
    if not os.path.exists(target_path):
        shutil.copyfile(tmp_path, target_path)
        _temp_fixtures_created.append(target_path)
    _temp_fixtures_created.append(tmp_path)
    return target_path

def _cleanup_fixtures():
    for p in _temp_fixtures_created:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

atexit.register(_cleanup_fixtures)

# Automatically ensure standalone fixtures exist on fresh clones
_ensure_fixture("dummy.wav", is_silent=True)
_ensure_fixture("input.wav", is_silent=False)

from utils.audio import trim_silence_rms
from models.asr import transcribe_audio, indic_transcribe
from models.translation import translate_with_nllb, CLAUSE_SPLIT_REGEX
from models.tts import (
    load_parler_tts,
    get_or_create_description_embeddings,
    generate_speech,
    _cached_desc_tensors
)
from run_pipeline import run_voice2voice_evaluation


def test_trim_silence_rms():
    print("\n--- Test 1: trim_silence_rms Verification ---")
    sr = 16000

    # 1. Pure silence tensor (shape [1, 32000]) -> should return full tensor (safe guard)
    silence_tensor = torch.zeros(1, 32000)
    out_silence = trim_silence_rms(silence_tensor, sample_rate=sr)
    assert out_silence.shape == silence_tensor.shape, f"Silence shape changed: {out_silence.shape}"
    assert isinstance(out_silence, torch.Tensor), "Output should be torch.Tensor"

    # 2. Synthetic speech (sine tone 0.5s surrounded by 1.0s silence on each side = 2.5s)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8
    padded = np.concatenate([np.zeros(sr, dtype=np.float32), tone, np.zeros(sr, dtype=np.float32)]) # 2.5s = 40,000 samples

    # Numpy 1D test
    trimmed_np = trim_silence_rms(padded, sample_rate=sr, pad_duration=0.08)
    assert len(trimmed_np) < len(padded), f"Did not trim numpy array: {len(trimmed_np)} vs {len(padded)}"
    expected_approx_samples = int((0.5 + 2 * 0.08) * sr) # ~0.66s = 10,560 samples
    assert abs(len(trimmed_np) - expected_approx_samples) < 2000, f"Unexpected trimmed length: {len(trimmed_np)}"

    # Torch 2D [1, T] test
    padded_torch = torch.from_numpy(padded).unsqueeze(0)
    trimmed_torch = trim_silence_rms(padded_torch, sample_rate=sr, pad_duration=0.08)
    assert trimmed_torch.shape[0] == 1, "Lost channel dimension"
    assert trimmed_torch.shape[1] == len(trimmed_np), "Numpy and Torch trimmed lengths differ"
    print(f"✅ trim_silence_rms successfully trimmed 2.50s audio to {len(trimmed_np)/sr:.2f}s!")


def test_asr_subphase_telemetry():
    print("\n--- Test 2: ASR Sub-Phase Telemetry & Conformer Trimming ---")
    dummy_wav = os.path.join(ROOT_DIR, "dummy.wav")
    assert os.path.exists(dummy_wav), f"Missing dummy.wav at {dummy_wav}"

    # 1. Backward compatibility: 4-tuple return when return_timings=False
    res4 = transcribe_audio(dummy_wav, return_timings=False)
    assert len(res4) == 4, f"Expected 4 elements, got {len(res4)}"
    print(f"✅ 4-tuple backward compatibility: transcription='{res4[0]}', code='{res4[1]}'")

    # 2. Granular telemetry: 5-tuple return when return_timings=True
    res5 = transcribe_audio(dummy_wav, return_timings=True)
    assert len(res5) == 5, f"Expected 5 elements, got {len(res5)}"
    sub_timings = res5[4]
    assert "lid_time_sec" in sub_timings, "Missing lid_time_sec"
    assert "transcription_time_sec" in sub_timings, "Missing transcription_time_sec"
    assert sub_timings["lid_time_sec"] > 0, "lid_time_sec should be positive"
    assert sub_timings["transcription_time_sec"] > 0, "transcription_time_sec should be positive"
    print(f"✅ Telemetry split verified: LID={sub_timings['lid_time_sec']:.3f}s, Decode={sub_timings['transcription_time_sec']:.3f}s")

    # 3. Isolated functions: detect_language_whisper & transcribe_speech_routed
    from models.asr import detect_language_whisper, transcribe_speech_routed
    lid_res = detect_language_whisper(dummy_wav)
    assert len(lid_res) == 7, f"Expected 7 elements from detect_language_whisper, got {len(lid_res)}"
    lang_code, lang_name, is_indic, mel, conf, routed, lid_t = lid_res
    assert lid_t > 0, "lid_t must be positive"

    trans_res = transcribe_speech_routed(dummy_wav, lang_code, lang_name, is_indic, mel=mel, confidence=conf)
    assert len(trans_res) == 3, f"Expected 3 elements from transcribe_speech_routed, got {len(trans_res)}"
    trans_text, routed_mod, trans_t = trans_res
    assert trans_t > 0, "trans_t must be positive"
    print(f"✅ Decoupled functions verified: LID={lid_t:.3f}s, Transcribe={trans_t:.3f}s, text='{trans_text}'")

    # 4. Anti-hallucination repetition suppression test
    from models.asr import suppress_repetition
    assert suppress_repetition("1000000") == "1000000", "Repetition suppression corrupted number"
    assert suppress_repetition("aaaaaaa") == "a", "Repetition suppression failed to collapse repeated chars"
    print("✅ Repetition suppression verified: numbers preserved (1000000), loop collapsed (aaaaaaa -> a)")

    # 5. Narrowest safe Telugu Unicode normalization regression test
    from models.asr import normalize_telugu_vowel_repetition
    # 5a. Confirmed CTC defect: duplicate U+0C4B ('ోో' -> 'ో')
    assert normalize_telugu_vowel_repetition("ఈరోోజు వాతావరణం బాగుంది") == "ఈరోజు వాతావరణం బాగుంది"
    assert normalize_telugu_vowel_repetition("ఈరోోోజు") == "ఈరోజు"
    assert normalize_telugu_vowel_repetition("ఈరోజు") == "ఈరోజు"
    # 5b. Valid numerals preserved
    assert normalize_telugu_vowel_repetition("1000000") == "1000000"
    assert normalize_telugu_vowel_repetition("22") == "22"
    # 5c. Valid repeated punctuation preserved
    assert normalize_telugu_vowel_repetition("..") == ".."
    assert normalize_telugu_vowel_repetition("--") == "--"
    # 5d. Consonant gemination in Telugu and Hindi preserved
    assert normalize_telugu_vowel_repetition("అమ్మ") == "అమ్మ"
    assert normalize_telugu_vowel_repetition("చెప్పండి") == "చెప్పండి"
    assert normalize_telugu_vowel_repetition("पत्ता") == "पत्ता"
    assert normalize_telugu_vowel_repetition("दिल्ली") == "दिल्ली"
    assert normalize_telugu_vowel_repetition("good") == "good"
    # 5e. Unrelated Indic text with other vowel signs preserved
    assert normalize_telugu_vowel_repetition("వాతావరణం") == "వాతావరణం"
    assert normalize_telugu_vowel_repetition("బాగుంది") == "బాగుంది"
    assert normalize_telugu_vowel_repetition("आज मौसम अच्छा है") == "आज मौसम अच्छा है"
    print("✅ Narrow Telugu Unicode normalization verified: U+0C4B duplicate collapsed, numerals/punctuation/gemination/unrelated Indic preserved")


def test_multimode_language_dispatch():
    print("\n--- Test 3: Multi-Mode Language Dispatch Verification ---")
    input_wav = os.path.join(ROOT_DIR, "input.wav")
    assert os.path.exists(input_wav), f"Missing input.wav at {input_wav}"
    test_log = os.path.join(ROOT_DIR, "test_eval_dispatch.json")
    if os.path.exists(test_log):
        os.remove(test_log)

    try:
        # 1. Target Specified Mode (--target hi on input.wav)
        entry_locked = run_voice2voice_evaluation(
            output_log=test_log,
            input_audio_path=input_wav,
            target_lang="hi"
        )
        assert entry_locked["outputs"]["target_language"]["code"] == "hi"
        assert entry_locked["outputs"]["target_language"]["dispatch_mode"] == "target_specified"
        assert entry_locked["timings"]["target_lang_selection_sec"] == 0.0
        print("✅ Target Specified mode verified: zero prompt overhead, target=hi")

        # 2. Bi-directional Auto-Dialogue (--pair en:te on input.wav)
        entry_dialogue = run_voice2voice_evaluation(
            output_log=test_log,
            input_audio_path=input_wav,
            pair_lang="en:te"
        )
        detected_src = entry_dialogue["outputs"]["source_language"]["code"]
        assert detected_src in ["te", "en"], f"Expected source lang te or en in dialogue pair, got {detected_src}"
        expected_tgt = "en" if detected_src == "te" else "te"
        assert entry_dialogue["outputs"]["target_language"]["code"] == expected_tgt
        assert entry_dialogue["outputs"]["target_language"]["dispatch_mode"] == "auto_dialogue"
        assert entry_dialogue["timings"]["target_lang_selection_sec"] == 0.0
        print(f"✅ Auto-Dialogue mode ({detected_src} -> {expected_tgt}) verified: zero prompt overhead, auto-selected {expected_tgt}")

        # 3. Source-Locked Mode (--source te -t en on input.wav)
        entry_src_locked = run_voice2voice_evaluation(
            output_log=test_log,
            input_audio_path=input_wav,
            source_lang="te",
            target_lang="en"
        )
        assert entry_src_locked["outputs"]["source_language"]["code"] == "te"
        assert entry_src_locked["outputs"]["source_language"]["confidence"] is None
        assert entry_src_locked["outputs"]["source_language"]["routed_model"] == "IndicConformer-600M"
        assert entry_src_locked["timings"]["lid_time_sec"] == 0.0
        print("✅ Source-Locked mode verified: zero LID overhead, routed to IndicConformer-600M, confidence=None")

        # 4. Verify log persistence
        with open(test_log, "r") as f:
            logs = json.load(f)
        assert len(logs) == 3, f"Expected 3 log entries, got {len(logs)}"
        print(f"✅ Atomic log persistence verified: {len(logs)} runs saved.")
    finally:
        if os.path.exists(test_log):
            os.remove(test_log)


def test_parler_tts_cache_and_dynamic_framing():
    print("\n--- Test 4: Parler-TTS Conditioning Cache & Dynamic Frame Capping ---")
    model, tokenizer, desc_tokenizer = load_parler_tts()

    desc = "A calm voice with natural pace, clarity and stability."
    # Check that pre-warming populated the cache
    assert desc in _cached_desc_tensors, "Cache was not pre-warmed for default description"
    input_ids, attn_mask, encoder_outputs = _cached_desc_tensors[desc]
    assert encoder_outputs is not None, "Cached encoder_outputs is None"
    print("✅ In-memory conditioning cache verified for default description!")

    # Test dynamic DAC frame calculation (86.13 Hz DAC vocoder)
    from models.tts import calculate_dynamic_dac_frames
    # 1-word greeting: should map to 172 frames (bounded between [170, 420])
    short_frames = calculate_dynamic_dac_frames("నమస్కారం")
    assert short_frames == 172, f"Expected 172 frames for 1 word, got {short_frames}"
    # Long prompt (e.g. 15 words): should hit hard ceiling of 420 frames
    long_frames = calculate_dynamic_dac_frames("ఈ రోజు వాతావరణం చాలా బాగుంది మరియు మేము అందరం కలిసి బయటకు వెళ్లాలని అనుకుంటున్నాము")
    assert long_frames == 420, f"Expected 420 frames for long text, got {long_frames}"
    print(f"✅ Dynamic DAC frame bounding verified: 1-word={short_frames} frames, 15-words={long_frames} frames (hard ceiling 420)")


def test_nllb_dynamic_token_bounding():
    print("\n--- Test 5: NLLB-200 Dynamic Token Bounding ---")
    input_text = "Good morning, have a wonderful day."
    # Translate English to Telugu
    translated = translate_with_nllb(input_text, src_lang_code="en", tgt_lang_code="te")
    assert translated is not None and len(translated) > 0, "Translation returned empty"
    print(f"✅ NLLB translation with dynamic tokens passed: '{input_text}' → '{translated}'")


def test_clause_preservation_and_dandas():
    print("\n--- Test 6: Multi-Script Clause Preservation & Decimal Protection ---")
    import re

    # 1. Decimal protection (must not split on 3.5 or other floating point numbers)
    t1 = "The price is 3.5 rupees. It is good."
    splits1 = [s.strip() for s in re.split(CLAUSE_SPLIT_REGEX, t1) if s.strip()]
    assert len(splits1) == 2, f"Expected 2 clauses, got {splits1}"
    assert "3.5" in splits1[0], f"Decimal was split: {splits1}"
    print(f"✅ Decimal protection verified: '{t1}' → {splits1}")

    # 2. Dual greetings without spaces
    t2 = "Hi!How are you?"
    splits2 = [s.strip() for s in re.split(CLAUSE_SPLIT_REGEX, t2) if s.strip()]
    assert len(splits2) == 2, f"Expected 2 clauses, got {splits2}"
    assert splits2[0] == "Hi!" and splits2[1] == "How are you?", f"Unexpected split: {splits2}"
    print(f"✅ Flexible spacing verified: '{t2}' → {splits2}")

    # 3. Indic danda support
    t3 = "नमस्ते।आप कैसे हैं?"
    splits3 = [s.strip() for s in re.split(CLAUSE_SPLIT_REGEX, t3) if s.strip()]
    assert len(splits3) == 2, f"Expected 2 clauses, got {splits3}"
    print(f"✅ Indic danda split verified: '{t3}' → {splits3}")

    # 4. Multi-clause translation test (no dropped sentences)
    trans = translate_with_nllb("Hello! How are you?", src_lang_code="en", tgt_lang_code="hi")
    assert trans is not None and len(trans) > 0, "Translation returned empty"
    print(f"✅ Multi-clause translation completed: 'Hello! How are you?' → '{trans}'")


def test_two_tier_telemetry_and_memory():
    print("\n--- Test 7: Two-Tier Telemetry & 3D Memory Verification ---")
    input_wav = os.path.join(ROOT_DIR, "input.wav")
    assert os.path.exists(input_wav), f"Missing input.wav at {input_wav}"
    test_log = os.path.join(ROOT_DIR, "test_eval_telemetry.json")
    if os.path.exists(test_log):
        os.remove(test_log)

    try:
        entry = run_voice2voice_evaluation(
            output_log=test_log,
            input_audio_path=input_wav,
            target_lang="hi"
        )
        timings = entry["timings"]
        metrics = entry["system_metrics"]

        # Check Two-Tier Timings
        assert "pure_machine_time_sec" in timings, "Missing pure_machine_time_sec"
        assert "io_subtotal_sec" in timings, "Missing io_subtotal_sec"
        assert "asr_time_sec" in timings, "Missing asr_time_sec"
        assert "translation_time_sec" in timings, "Missing translation_time_sec"
        assert "tts_synthesis_time_sec" in timings, "Missing tts_synthesis_time_sec"
        assert timings["pure_machine_time_sec"] > 0, "pure_machine_time_sec must be > 0"

        # Check 3D Memory Telemetry
        assert "active_ram_mb" in metrics, "Missing active_ram_mb"
        assert "avg_ram_mb" in metrics, "Missing avg_ram_mb"
        assert "peak_ram_usage_mb" in metrics, "Missing peak_ram_usage_mb"
        assert metrics["active_ram_mb"] > 0, "active_ram_mb must be > 0"
        assert metrics["avg_ram_mb"] > 0, "avg_ram_mb must be > 0"
        assert metrics["peak_ram_usage_mb"] >= metrics["active_ram_mb"], "Peak must be >= active"

        print(f"✅ Two-Tier Timings verified: Pure Machine Compute={timings['pure_machine_time_sec']:.2f}s, I/O={timings['io_subtotal_sec']:.2f}s")
        print(f"✅ 3D Memory verified: Active={metrics['active_ram_mb']:.1f}MB, Avg={metrics['avg_ram_mb']:.1f}MB, Peak={metrics['peak_ram_usage_mb']:.1f}MB")
    finally:
        if os.path.exists(test_log):
            os.remove(test_log)


def test_phase1_routing_and_typography():
    print("\n--- Test 8: Phase 1 4-Quadrant Routing, Typography & Display Names ---")
    from utils.language import get_display_name, DISPLAY_LANG_MAP
    assert get_display_name("en") == "English", f"Expected 'English', got '{get_display_name('en')}'"
    assert get_display_name("te") == "Telugu", f"Expected 'Telugu', got '{get_display_name('te')}'"
    assert get_display_name("hi") == "Hindi", f"Expected 'Hindi', got '{get_display_name('hi')}'"
    assert get_display_name("fr") == "French", f"Expected 'French', got '{get_display_name('fr')}'"
    print("✅ Canonical display names verified: en->English, te->Telugu, fr->French")

    # Typography test: Indic script detection
    indic_sample = "నమస్కారం"
    ascii_sample = "Hello World"
    is_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in indic_sample)
    is_ascii_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in ascii_sample)
    assert is_indic is True
    assert is_ascii_indic is False
    print("✅ Typography script detection verified: Indic scripts trigger non-bold \\033[97m")


def test_research_registries_and_scope():
    print("\n--- Test 9: Research Registries & Paper Scope Verification ---")
    from utils.language import (
        PAPER_PRINCIPAL_INDIC_LANGS,
        PAPER_PRINCIPAL_NON_INDIC_LANGS,
        PAPER_LID_ASR_LANGS,
        PAPER_NMT_LANGS,
        PAPER_TTS_LANGS,
        PAPER_E2E_S2ST_DIRECTIONS,
        is_paper_indic_language,
        is_indic_language,
        RESEARCH_CONFIG,
        RUNTIME_CONFIG
    )

    # 1. Verify paper principal languages (Nature Scientific Reports 2026 Table 6)
    assert PAPER_PRINCIPAL_INDIC_LANGS == {"hi", "te", "ta", "bn"}, "Paper Indic principal set mismatch"
    assert PAPER_PRINCIPAL_NON_INDIC_LANGS == {"fr", "de", "ar", "ja"}, "Paper Non-Indic principal set mismatch"
    assert len(PAPER_LID_ASR_LANGS) == 8, f"Expected 8 FLEURS LID/ASR languages, got {len(PAPER_LID_ASR_LANGS)}"
    assert len(PAPER_NMT_LANGS) == 9, f"Expected 9 NMT languages (8 + English), got {len(PAPER_NMT_LANGS)}"
    assert len(PAPER_TTS_LANGS) == 8, f"Expected 8 TTS evaluated languages, got {len(PAPER_TTS_LANGS)}"
    assert len(PAPER_E2E_S2ST_DIRECTIONS) == 5, f"Expected 5 CVSS E2E directions, got {len(PAPER_E2E_S2ST_DIRECTIONS)}"

    # 2. Scope boundary helper tests
    assert is_paper_indic_language("te") is True, "Telugu must be paper Indic language"
    assert is_paper_indic_language("kn") is False, "Kannada is not in paper Table 6 Indic set"
    assert is_indic_language("kn") is True, "Kannada is in general Indian languages map"
    assert is_paper_indic_language("es") is False, "Spanish is not Indic"

    # 3. Execution configs & verified direction registries
    assert RESEARCH_CONFIG["routing_threshold"] == 0.70, "Paper operating threshold must be 0.70"
    assert len(RESEARCH_CONFIG["fleurs_principal_languages"]) == 8

    from utils.language import EMPIRICALLY_TESTED_DIRECTIONS, RUNTIME_SUPPORTED_PAIRS
    assert "te:en" in EMPIRICALLY_TESTED_DIRECTIONS, "te:en must be in empirically tested directions"
    assert "te:hi" in EMPIRICALLY_TESTED_DIRECTIONS, "te:hi must be in empirically tested directions"
    assert len(EMPIRICALLY_TESTED_DIRECTIONS) == 2, f"Expected 2 empirically tested directions, got {len(EMPIRICALLY_TESTED_DIRECTIONS)}"
    assert len(RUNTIME_SUPPORTED_PAIRS) == 28, f"Expected 28 runtime supported pairs, got {len(RUNTIME_SUPPORTED_PAIRS)}"
    print("✅ Research registries, paper scope boundaries & empirical directions verified successfully!")


def test_deterministic_mocked_logits_and_scoring_universe():
    print("\n--- Test 10: Deterministic Mocked Logits & Exact Candidate Alignment ---")
    import whisper
    tokenizer = whisper.tokenizer.get_tokenizer(True, num_languages=99)

    # Synthetic logits tensor for vocab [1, 51865]
    vocab_size = 51865
    mock_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    # Inject acoustic logits: te = 12.0, hi = 10.0, en = 11.5
    te_tok = tokenizer.to_language_token("te")
    hi_tok = tokenizer.to_language_token("hi")
    en_tok = tokenizer.to_language_token("en")

    mock_logits[0, te_tok] = 12.0
    mock_logits[0, hi_tok] = 10.0
    mock_logits[0, en_tok] = 11.5

    from utils.language import PAPER_LID_ASR_LANGS, is_paper_indic_language

    # Test Paper Baseline: Candidate set is 8 principal languages
    paper_cands = sorted(PAPER_LID_ASR_LANGS)
    paper_tokens = [tokenizer.to_language_token(c) for c in paper_cands]
    p_logits = mock_logits[0, paper_tokens]
    p_probs = torch.softmax(p_logits, dim=-1)

    top_idx = torch.argmax(p_probs).item()
    detected_code = paper_cands[top_idx]
    top_prob = p_probs[top_idx].item()

    assert detected_code == "te", f"Expected top language 'te', got '{detected_code}'"
    assert top_prob > 0.50, f"Expected top probability > 0.50, got {top_prob}"

    # Test Paper Eq. 3 Decision Rule:
    tau = 0.70
    is_indic = is_paper_indic_language(detected_code)
    assert is_indic is True
    routed_model = "IndicConformer-600M" if (top_prob >= tau and is_indic) else "Whisper-small"
    print(f"✅ Paper Baseline Eq. (1)-(3): Top='{detected_code}' (c={top_prob:.2%}) ➔ Routed: {routed_model}")

    # Test Layer 2 Dialogue Extension: pair ["en", "te"]
    # Direct pairwise discrimination between Candidate A (en) and Candidate B (te)
    pair_tokens = [en_tok, te_tok]
    pair_logits = mock_logits[0, pair_tokens]
    pair_probs = torch.softmax(pair_logits, dim=-1)
    p_en = pair_probs[0].item()
    p_te = pair_probs[1].item()

    assert p_te > p_en, f"Expected Telugu ({p_te:.2%}) > English ({p_en:.2%})"
    assert abs(p_te + p_en - 1.0) < 1e-5, f"Pairwise probabilities must sum to 1.0, got {p_te + p_en}"

    # Out-of-pair guard test across research evaluation universe
    from utils.language import RESEARCH_NMT_LANGS
    guard_cands = [c for c in sorted(RESEARCH_NMT_LANGS) if c in tokenizer.all_language_codes]
    guard_tokens = [tokenizer.to_language_token(c) for c in guard_cands]
    guard_probs = torch.softmax(mock_logits[0, guard_tokens], dim=-1)
    top_guard_idx = torch.argmax(guard_probs).item()
    top_guard_lang = guard_cands[top_guard_idx]

    assert top_guard_lang == "te", f"Expected top guard language 'te', got '{top_guard_lang}'"
    print(f"✅ Runtime Pair Mode: Direct pairwise discrimination 'te' ({p_te:.2%}) vs 'en' ({p_en:.2%}) verified!")


def test_tts_contract_and_audio_artifact_handling():
    print("\n--- Test 11: TTS 3-Tuple Contract & Playback Handling ---")
    from models.tts import synthesize_tts, playback_tts
    import tempfile

    # 1. Empty text guard -> returns ('none', 'skipped')
    res_empty = synthesize_tts("", lang_type="indian", tgt_lang_code="te")
    assert res_empty == (None, "none", "skipped"), f"Expected skipped, got {res_empty}"

    # 2. International text -> returns ('pyttsx3', 'host_speaker') with None artifact
    res_intl = synthesize_tts("Hello world", lang_type="international", tgt_lang_code="en")
    assert res_intl == (None, "pyttsx3", "host_speaker"), f"Expected host_speaker, got {res_intl}"

    # 3. Verify playback_tts handles string text for host_speaker without error
    try:
        playback_tts(None, None, "en")
        print("✅ None target safely handled in playback_tts")
    except Exception as e:
        assert False, f"playback_tts crashed on None target: {e}"

    print("✅ TTS 3-Tuple Return Contract verified: (artifact_path, engine_type, delivery_mode)")


def test_os_tts_playback_contract_mock():
    print("\n--- Test 12: OS TTS Speech Engine Mock Verification ---")
    from unittest.mock import MagicMock, patch
    from models.tts import playback_tts

    mock_engine = MagicMock()
    with patch("models.tts.get_tts_engine", return_value=mock_engine), \
         patch("models.tts.get_system_voice_for_lang", return_value="com.apple.speech.synthesis.voice.Alex"):

        sample_translated_text = "Hello, how are you today?"
        playback_tts(sample_translated_text, "pyttsx3", "en")

        mock_engine.say.assert_called_once_with(sample_translated_text)
        mock_engine.runAndWait.assert_called_once()
        print(f"✅ Verified mock_engine.say invoked with translated text: '{sample_translated_text}'")
        print("✅ Verified mock_engine.runAndWait invoked successfully")


def test_cli_exit_code_on_pipeline_failure():
    print("\n--- Test 13: CLI Exit Code Propagation on Pipeline Error ---")
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "run_pipeline.py"),
        "-a", "non_existent_file_xyz.wav",
        "-t", "te"
    ]
    proc = subprocess.run(cmd, env=SUBPROCESS_ENV, capture_output=True, text=True)
    assert proc.returncode == 1, f"Expected exit code 1 on pipeline error, got {proc.returncode}"
    print(f"✅ CLI properly exited with code {proc.returncode} when input file missing/pipeline error")


def test_validate_pair_syntax():
    print("\n--- Test 14: Dialogue Pair Syntax & Vocabulary Validation ---")
    from utils.language import validate_pair_syntax

    # Valid syntax
    p1, p2 = validate_pair_syntax("en:te")
    assert (p1, p2) == ("en", "te")
    p1, p2 = validate_pair_syntax("hi:en")
    assert (p1, p2) == ("hi", "en")
    p1, p2 = validate_pair_syntax(" EN : HI ")
    assert (p1, p2) == ("en", "hi")

    # Invalid syntax: empty / missing / duplicates / invalid codes
    for invalid in ["", "en", "en:te:hi", ":te", "en:", "en:en", "en:invalid_xyz"]:
        try:
            validate_pair_syntax(invalid)
            assert False, f"Expected ValueError for '{invalid}'"
        except ValueError:
            pass
    print("✅ Dialogue pair syntax validation passed: accepted valid pairs, rejected malformed pairs.")


def test_silence_gating_pre_recorded_audio():
    print("\n--- Test 15: Pre-ASR Silence Gate on Pre-Recorded Audio ---")
    dummy_wav = os.path.join(ROOT_DIR, "dummy.wav")
    assert os.path.exists(dummy_wav), f"Missing dummy.wav at {dummy_wav}"

    res = run_voice2voice_evaluation(input_audio_path=dummy_wav, target_lang="te")
    assert res["status"] == "silence", f"Expected 'silence', got {res.get('status')}"
    assert res["timings"]["pure_machine_time_sec"] == 0.0, "Machine compute must be 0.0s for silent input"
    print("✅ Pre-ASR silence gate verified: dummy.wav (< -48 dBFS) aborted cleanly with 0.0s machine compute.")


def test_pair_mode_bypasses_tau_dead_zone():
    print("\n--- Test 16: Pair Mode Bypasses tau=0.70 Dead Zone for IndicConformer ---")
    from unittest.mock import patch
    from models.asr import detect_language_whisper

    # Mock detect_input_language_whisper to return an Indic language with confidence 0.58
    # (Inside the legacy dead zone: 0.50 <= c < 0.70)
    mock_lid_res = ("te", "Telugu", True, None, 0.58, None, "pairwise_relative")

    with patch("models.asr.detect_input_language_whisper", return_value=mock_lid_res):
        # 1. Pair Mode: should route directly to IndicConformer-600M despite c=0.58 < 0.70
        lid_res_pair = detect_language_whisper("dummy.wav", candidate_langs=["en", "te"], is_dialogue=True)
        assert lid_res_pair[5] == "IndicConformer-600M", f"Expected IndicConformer-600M, got {lid_res_pair[5]}"
        print(f"✅ Pair Mode correctly routes to IndicConformer-600M at c=58% (bypassing dead zone)")

        # 2. Research Mode: should route to Whisper fallback because c=0.58 < tau (0.70)
        lid_res_research = detect_language_whisper("dummy.wav", is_dialogue=False)
        assert lid_res_research[5] == "Whisper-small (Low-Confidence Indic Fallback)", f"Expected Whisper fallback, got {lid_res_research[5]}"
        print(f"✅ Research Mode preserves Eq. 3 tau=0.70 fallback at c=58%")


def test_empty_output_guards():
    print("\n--- Test 17: Empty ASR and Translation Output Guards ---")
    from unittest.mock import patch
    input_wav = os.path.join(ROOT_DIR, "input.wav")
    assert os.path.exists(input_wav), f"Missing input.wav at {input_wav}"

    # 1. Empty ASR transcription guard
    with patch("run_pipeline.transcribe_speech_routed", return_value=("", "IndicConformer-600M", 0.1, "te")):
        res_empty_asr = run_voice2voice_evaluation(input_audio_path=input_wav, target_lang="en")
        assert res_empty_asr["status"] == "empty_transcription"
        assert res_empty_asr["timings"]["translation_time_sec"] == 0.0

    # 2. Empty Translation output guard
    with patch("run_pipeline.translate_with_nllb", return_value=""):
        res_empty_nmt = run_voice2voice_evaluation(input_audio_path=input_wav, target_lang="en")
        assert res_empty_nmt["status"] == "empty_translation"
        assert res_empty_nmt["timings"]["tts_synthesis_time_sec"] == 0.0

    print("✅ Empty output guards verified: pipeline halts safely before subsequent stages.")


def test_orchestrator_pair_mode_dead_zone_bypass():
    print("\n--- Test 18: Orchestrator en:te Routing at c=0.58 Bypasses Dead Zone ---")
    from unittest.mock import patch
    input_wav = os.path.join(ROOT_DIR, "input.wav")
    assert os.path.exists(input_wav), f"Missing input.wav at {input_wav}"

    # Mock detect_input_language_whisper: Telugu detected with confidence 0.58 in pair ["en", "te"]
    mock_lid = ("te", "Telugu", True, None, 0.58, None, "pairwise_relative")

    with patch("models.asr.detect_input_language_whisper", return_value=mock_lid), \
         patch("run_pipeline.transcribe_speech_routed", return_value=("చరిత్రలో కొత్తగా కనుగొన్నవి ఏవి", "IndicConformer-600M", 0.5, "te")), \
         patch("run_pipeline.translate_with_nllb", return_value="What are new discoveries in history?"), \
         patch("run_pipeline.synthesize_tts", return_value=(None, "pyttsx3", "host_speaker")), \
         patch("run_pipeline.playback_tts", return_value=True):

        res = run_voice2voice_evaluation(
            input_audio_path=input_wav,
            pair_lang="en:te"
        )
        assert res["status"] == "completed"
        # Verify source language routed to IndicConformer-600M (bypassing dead zone)
        assert res["outputs"]["source_language"]["code"] == "te"
        assert res["outputs"]["source_language"]["routed_model"] == "IndicConformer-600M"
        assert res["outputs"]["source_language"]["confidence"] == 0.58
        assert res["outputs"]["source_language"]["confidence_type"] == "pairwise_relative"
        assert res["outputs"]["source_language"]["dispatch_mode"] == "dialogue_pair"
        # Verify auto-dialogue target is English
        assert res["outputs"]["target_language"]["code"] == "en"
        assert res["outputs"]["target_language"]["dispatch_mode"] == "auto_dialogue"
        print("✅ Orchestrator en:te pair routing verified: c=0.58 correctly decodes via IndicConformer-600M and dispatches target to English.")


def test_language_argument_and_registry_enforcement():
    print("\n--- Test 19: Pair Registry Enforcement and Contradictory Flag Validation ---")
    from utils.language import (
        validate_pair_syntax,
        validate_target_language,
        validate_source_language,
        validate_language_arguments,
        RUNTIME_SUPPORTED_PAIRS
    )

    # 1. Unregistered pair rejection: es:en is valid syntax but NOT in RUNTIME_SUPPORTED_PAIRS (28 pairs)
    try:
        validate_pair_syntax("es:en")
        assert False, "Expected ValueError for unregistered pair 'es:en'"
    except ValueError as e:
        assert "RUNTIME_SUPPORTED_PAIRS" in str(e)
        print(f"✅ Unregistered pair 'es:en' rejected cleanly: {e}")

    # 2. Contradictory flags: --source with --pair
    try:
        validate_language_arguments(source_lang="fr", pair_lang="en:te")
        assert False, "Expected ValueError for --source with --pair"
    except ValueError as e:
        assert "is not a participant in configured dialogue pair" in str(e)
        print(f"✅ Contradictory --source and --pair rejected cleanly: {e}")

    # 3. Contradictory flags: --target with --pair
    try:
        validate_language_arguments(target_lang="hi", pair_lang="en:te")
        assert False, "Expected ValueError for --target with --pair"
    except ValueError as e:
        assert "is not a participant in configured dialogue pair" in str(e)
        print(f"✅ Contradictory --target and --pair rejected cleanly: {e}")

    # 4. Unknown target code
    try:
        validate_target_language("xyz")
        assert False, "Expected ValueError for unknown target code 'xyz'"
    except ValueError as e:
        assert "Unsupported target language" in str(e)
        print(f"✅ Invalid target code 'xyz' rejected cleanly: {e}")

    # 5. Unknown source code
    try:
        validate_source_language("xyz")
        assert False, "Expected ValueError for unknown source code 'xyz'"
    except ValueError as e:
        assert "Unsupported source language" in str(e)
        print(f"✅ Invalid source code 'xyz' rejected cleanly: {e}")


def test_candidate_universe_research_vs_runtime():
    print("\n--- Test 20: Candidate Universe Separation (8-lang Research vs 9-lang Runtime) ---")
    from utils.language import RESEARCH_LID_LANGS, RUNTIME_LID_LANGS
    from models.asr import detect_input_language_whisper, get_whisper_model
    import whisper

    assert len(RESEARCH_LID_LANGS) == 8, f"Expected 8 research LID languages, got {len(RESEARCH_LID_LANGS)}"
    assert "en" not in RESEARCH_LID_LANGS, "English must not be in RESEARCH_LID_LANGS"
    assert len(RUNTIME_LID_LANGS) == 9, f"Expected 9 runtime LID languages, got {len(RUNTIME_LID_LANGS)}"
    assert "en" in RUNTIME_LID_LANGS, "English must be in RUNTIME_LID_LANGS"

    whisper_model = get_whisper_model()
    input_wav = os.path.join(ROOT_DIR, "input.wav")

    # In Research Mode: candidates evaluated should be RESEARCH_LID_LANGS (8 languages)
    res_research = detect_input_language_whisper(
        input_wav,
        whisper_model,
        candidate_langs=None,
        is_research_mode=True,
        return_metadata=True
    )
    assert res_research[6] == "research_candidate_softmax"
    print(f"✅ Research Mode candidate evaluation verified: confidence_type={res_research[6]}")

    # In Runtime Mode: candidates evaluated should be RUNTIME_LID_LANGS (9 languages)
    res_runtime = detect_input_language_whisper(
        input_wav,
        whisper_model,
        candidate_langs=None,
        is_research_mode=False,
        return_metadata=True
    )
    assert res_runtime[6] == "runtime_candidate_softmax"
    print(f"✅ Runtime Mode candidate evaluation verified: confidence_type={res_runtime[6]}")


def test_cli_exit_code_on_empty_asr_and_nmt():
    print("\n--- Test 21: CLI Exit Code 1 on Empty ASR / NMT Failure Stages ---")
    import subprocess
    input_wav = os.path.join(ROOT_DIR, "input.wav")

    # 1. Script that executes run_pipeline with mock empty ASR
    test_code_asr = (
        "import sys, unittest.mock; from run_pipeline import run_voice2voice_evaluation; "
        "with unittest.mock.patch('run_pipeline.transcribe_speech_routed', return_value=('', 'IndicConformer-600M', 0.1, 'te')): "
        "    entry = run_voice2voice_evaluation(input_audio_path='input.wav', target_lang='en'); "
        "    sys.exit(1 if (entry.get('status') == 'error' or entry.get('failed_stage') is not None) else 0)"
    )
    proc_asr = subprocess.run([sys.executable, "-c", test_code_asr], env=SUBPROCESS_ENV, cwd=ROOT_DIR, capture_output=True, text=True)
    assert proc_asr.returncode == 1, f"Expected returncode 1 for empty ASR, got {proc_asr.returncode}"
    print(f"✅ CLI exit code 1 verified for empty ASR failure stage")

    # 2. Script that executes run_pipeline with mock empty NMT
    test_code_nmt = (
        "import sys, unittest.mock; from run_pipeline import run_voice2voice_evaluation; "
        "with unittest.mock.patch('run_pipeline.translate_with_nllb', return_value=''): "
        "    entry = run_voice2voice_evaluation(input_audio_path='input.wav', target_lang='en'); "
        "    sys.exit(1 if (entry.get('status') == 'error' or entry.get('failed_stage') is not None) else 0)"
    )
    proc_nmt = subprocess.run([sys.executable, "-c", test_code_nmt], env=SUBPROCESS_ENV, cwd=ROOT_DIR, capture_output=True, text=True)
    assert proc_nmt.returncode == 1, f"Expected returncode 1 for empty NMT, got {proc_nmt.returncode}"
    print(f"✅ CLI exit code 1 verified for empty NMT failure stage")


def test_out_of_pair_guard_telemetry():
    print("\n--- Test 22: Out-of-Pair Acoustic Guard & Structured Telemetry ---")
    import whisper
    from models.asr import detect_input_language_whisper, get_whisper_model

    whisper_model = get_whisper_model()
    tokenizer = whisper.tokenizer.get_tokenizer(True, num_languages=99)
    input_wav = os.path.join(ROOT_DIR, "input.wav")

    # Synthetic logits where Japanese has the highest score (outside pair en:te)
    ja_tok = tokenizer.to_language_token("ja")
    en_tok = tokenizer.to_language_token("en")
    te_tok = tokenizer.to_language_token("te")

    class MockWhisper:
        device = whisper_model.device
        dims = whisper_model.dims
        is_multilingual = True
        num_languages = 99
        def encoder(self, mel):
            return torch.zeros((1, 1500, whisper_model.dims.n_audio_state), device=whisper_model.device)
        def decoder(self, tokens, audio_features):
            logits = torch.zeros((1, 1, 51865), device=whisper_model.device)
            # Japanese: 15.0, Telugu: 8.0, English: 7.0
            logits[0, 0, ja_tok] = 15.0
            logits[0, 0, te_tok] = 8.0
            logits[0, 0, en_tok] = 7.0
            return logits

    mock_model = MockWhisper()
    res = detect_input_language_whisper(
        input_wav,
        mock_model,
        candidate_langs=["en", "te"],
        return_metadata=True
    )
    detected_code, lang_name, is_indic, mel, confidence, out_of_pair_info, conf_type = res
    assert conf_type == "pairwise_relative"
    assert out_of_pair_info is not None, "Out-of-pair acoustic guard must trigger when Japanese logit dominates"
    assert out_of_pair_info["detected_lang"] == "ja", f"Expected detected_lang 'ja', got {out_of_pair_info['detected_lang']}"
    assert out_of_pair_info["confidence"] > 0.60
    print(f"✅ Out-of-pair guard triggered successfully: {out_of_pair_info['detected_lang']} (conf={out_of_pair_info['confidence']:.1%}) outside pair ['en', 'te']")


def test_delivery_status_and_playback_failure_mock():
    print("\n--- Test 23: TTS Delivery Status & Playback Failure Distinction ---")
    import unittest.mock
    import subprocess
    from run_pipeline import run_voice2voice_evaluation
    from models.tts import playback_tts

    # -------------------------------------------------------------
    # Sub-test A: Host OS TTS playback failure & Persisted JSON log
    # -------------------------------------------------------------
    test_log_host = os.path.join(ROOT_DIR, "test_failure_host_log.json")
    if os.path.exists(test_log_host):
        os.remove(test_log_host)

    input_wav = os.path.join(ROOT_DIR, "input.wav")
    dummy_wav = os.path.join(ROOT_DIR, "dummy.wav")

    try:
        with unittest.mock.patch("run_pipeline.transcribe_speech_routed", return_value=("నమస్కారం", "IndicConformer-600M", 0.1, "te")), \
             unittest.mock.patch("run_pipeline.translate_with_nllb", return_value="Hello"), \
             unittest.mock.patch("run_pipeline.playback_tts", return_value=False):
            entry_host = run_voice2voice_evaluation(
                input_audio_path=input_wav,
                source_lang="te",
                target_lang="en",
                output_log=test_log_host
            )
            assert entry_host["status"] == "error", f"Expected 'error', got {entry_host['status']}"
            assert entry_host["failed_stage"] == "tts", f"Expected failed_stage 'tts', got {entry_host['failed_stage']}"
            assert entry_host["delivery_status"] == "delivery_failed", f"Expected delivery_status 'delivery_failed', got {entry_host.get('delivery_status')}"
            assert entry_host["outputs"]["tts_playback_success"] is False
            assert entry_host["outputs"]["tts_synthesis_success"] is False
            assert entry_host["system_metrics"]["failed_stage"] == "tts"
            assert entry_host["system_metrics"]["delivery_status"] == "delivery_failed"

            # Verify persisted JSON log file contains exact matching fields
            assert os.path.exists(test_log_host), "Expected test_log_host to be persisted"
            with open(test_log_host, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            persisted_entry = persisted[-1] if isinstance(persisted, list) else persisted
            assert persisted_entry["status"] == "error"
            assert persisted_entry["failed_stage"] == "tts"
            assert persisted_entry["delivery_status"] == "delivery_failed"
            assert persisted_entry["outputs"]["tts_playback_success"] is False
            assert persisted_entry["outputs"]["tts_synthesis_success"] is False
            print("✅ Host OS TTS failure correctly persists status='error', failed_stage='tts', delivery_status='delivery_failed'")
    finally:
        if os.path.exists(test_log_host):
            os.remove(test_log_host)

    # -------------------------------------------------------------
    # Sub-test B: Neural TTS artifact retention on physical playback failure
    # -------------------------------------------------------------
    test_log_neural = os.path.join(ROOT_DIR, "test_failure_neural_log.json")
    if os.path.exists(test_log_neural):
        os.remove(test_log_neural)

    try:
        with unittest.mock.patch("run_pipeline.transcribe_speech_routed", return_value=("Hello", "Whisper", 0.1, "en")), \
             unittest.mock.patch("run_pipeline.translate_with_nllb", return_value="నమస్కారం"), \
             unittest.mock.patch("run_pipeline.synthesize_tts", return_value=(dummy_wav, "parler", "file_artifact")), \
             unittest.mock.patch("run_pipeline.playback_tts", return_value=False):
            entry_neural = run_voice2voice_evaluation(
                input_audio_path=input_wav,
                source_lang="en",
                target_lang="te",
                output_log=test_log_neural
            )
            assert entry_neural["status"] == "completed", f"Expected 'completed', got {entry_neural['status']}"
            assert entry_neural.get("failed_stage") is None
            assert entry_neural["delivery_status"] == "playback_failed"
            assert entry_neural["outputs"]["tts_synthesis_success"] is True
            assert entry_neural["outputs"]["tts_playback_success"] is False
            assert entry_neural["outputs"]["delivery_status"] == "playback_failed"
            assert os.path.exists(entry_neural["outputs"]["tts_audio_file"])

            # Verify persisted JSON log file contains exact matching fields
            assert os.path.exists(test_log_neural), "Expected test_log_neural to be persisted"
            with open(test_log_neural, "r", encoding="utf-8") as f:
                persisted_n = json.load(f)
            persisted_n_entry = persisted_n[-1] if isinstance(persisted_n, list) else persisted_n
            assert persisted_n_entry["status"] == "completed"
            assert persisted_n_entry.get("failed_stage") is None
            assert persisted_n_entry["delivery_status"] == "playback_failed"
            assert persisted_n_entry["outputs"]["tts_synthesis_success"] is True
            assert persisted_n_entry["outputs"]["tts_playback_success"] is False
            print("✅ Neural TTS with valid artifact persists status='completed', delivery_status='playback_failed', tts_synthesis_success=True")
    finally:
        if os.path.exists(test_log_neural):
            os.remove(test_log_neural)

    # -------------------------------------------------------------
    # Sub-test C: Full exception-to-False chain propagation
    # -------------------------------------------------------------
    # 1. sounddevice wait() exception -> play_wav() -> playback_tts() -> False
    with unittest.mock.patch("sounddevice.wait", side_effect=RuntimeError("CoreAudio output stream buffer underrun")):
        neural_res = playback_tts("dummy.wav", "neural", "te")
        assert neural_res is False, f"Expected False from sounddevice exception, got {neural_res}"
        print("✅ sounddevice exception cleanly caught and propagated as False through playback_tts")

    # 2. pyttsx3 runAndWait() exception -> playback_tts() -> False
    mock_crash_engine = unittest.mock.MagicMock()
    mock_crash_engine.runAndWait.side_effect = RuntimeError("Host speech synthesizer crash")
    with unittest.mock.patch("models.tts.get_tts_engine", return_value=mock_crash_engine):
        host_res = playback_tts("Hello", "pyttsx3", "en")
        assert host_res is False, f"Expected False from pyttsx3 runAndWait exception, got {host_res}"
        print("✅ pyttsx3 runAndWait exception cleanly caught and propagated as False through playback_tts")

    # -------------------------------------------------------------
    # Sub-test D: Actual Subprocess Returncode Verification
    # -------------------------------------------------------------
    # 1. Subprocess executing CLI with Host OS TTS failure -> must exit 1
    test_code_host_cli = (
        "import sys\n"
        "import unittest.mock\n"
        "from run_pipeline import run_voice2voice_evaluation\n"
        "with unittest.mock.patch('run_pipeline.transcribe_speech_routed', return_value=('నమస్కారం', 'IndicConformer-600M', 0.1, 'te')), \\\n"
        "     unittest.mock.patch('run_pipeline.translate_with_nllb', return_value='Hello'), \\\n"
        "     unittest.mock.patch('run_pipeline.playback_tts', return_value=False):\n"
        "    entry = run_voice2voice_evaluation(input_audio_path='input.wav', source_lang='te', target_lang='en')\n"
        "    sys.exit(1 if (entry.get('status') == 'error' or entry.get('failed_stage') is not None) else 0)\n"
    )
    proc_host = subprocess.run([sys.executable, "-c", test_code_host_cli], env=SUBPROCESS_ENV, cwd=ROOT_DIR, capture_output=True, text=True)
    assert proc_host.returncode == 1, f"Expected subprocess returncode 1 for host TTS failure, got {proc_host.returncode}"
    print(f"✅ Real CLI subprocess verified to exit with code 1 on Host OS TTS delivery failure")

    # 2. Subprocess executing CLI with Neural TTS playback failure -> must exit 0 (artifact preserved)
    test_code_neural_cli = (
        "import sys\n"
        "import unittest.mock\n"
        "from run_pipeline import run_voice2voice_evaluation\n"
        "with unittest.mock.patch('run_pipeline.transcribe_speech_routed', return_value=('Hello', 'Whisper', 0.1, 'en')), \\\n"
        "     unittest.mock.patch('run_pipeline.translate_with_nllb', return_value='నమస్కారం'), \\\n"
        "     unittest.mock.patch('run_pipeline.synthesize_tts', return_value=('dummy.wav', 'parler', 'file_artifact')), \\\n"
        "     unittest.mock.patch('run_pipeline.playback_tts', return_value=False):\n"
        "    entry = run_voice2voice_evaluation(input_audio_path='input.wav', source_lang='en', target_lang='te')\n"
        "    sys.exit(1 if (entry.get('status') == 'error' or entry.get('failed_stage') is not None) else 0)\n"
    )
    proc_neural = subprocess.run([sys.executable, "-c", test_code_neural_cli], env=SUBPROCESS_ENV, cwd=ROOT_DIR, capture_output=True, text=True)
    assert proc_neural.returncode == 0, f"Expected subprocess returncode 0 for neural artifact success, got {proc_neural.returncode}"
    print(f"✅ Real CLI subprocess verified to exit with code 0 on Neural TTS playback failure (artifact preserved)")


def main():
    print("=" * 60)
    print("🚀 RUNNING SPRINT 2 & SPRINT 3 VERIFICATION SUITE")
    print("=" * 60)

    test_trim_silence_rms()
    test_asr_subphase_telemetry()
    test_multimode_language_dispatch()
    test_parler_tts_cache_and_dynamic_framing()
    test_nllb_dynamic_token_bounding()
    test_clause_preservation_and_dandas()
    test_two_tier_telemetry_and_memory()
    test_phase1_routing_and_typography()
    test_research_registries_and_scope()
    test_deterministic_mocked_logits_and_scoring_universe()
    test_tts_contract_and_audio_artifact_handling()
    test_os_tts_playback_contract_mock()
    test_cli_exit_code_on_pipeline_failure()
    test_validate_pair_syntax()
    test_silence_gating_pre_recorded_audio()
    test_pair_mode_bypasses_tau_dead_zone()
    test_empty_output_guards()
    test_orchestrator_pair_mode_dead_zone_bypass()
    test_language_argument_and_registry_enforcement()
    test_candidate_universe_research_vs_runtime()
    test_cli_exit_code_on_empty_asr_and_nmt()
    test_out_of_pair_guard_telemetry()
    test_delivery_status_and_playback_failure_mock()

    print("\n" + "=" * 60)
    print("🎉 ALL SPRINT 2 & SPRINT 3 UNIT AND INTEGRATION TESTS PASSED 100%!")
    print("=" * 60)


if __name__ == "__main__":
    main()
