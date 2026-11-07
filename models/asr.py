import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore")

import re
import contextlib
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import whisper
import torch
import torchaudio
from time import perf_counter
from transformers import AutoModel

from utils.system import device
from utils.language import (
    INDIAN_LANG_MAP,
    DISPLAY_LANG_MAP,
    PAPER_LANG_SET,
    PAPER_INDIC_SET,
    PAPER_NON_INDIC_SET,
    PAPER_LID_ASR_LANGS,
    PAPER_PRINCIPAL_INDIC_LANGS,
    is_paper_indic_language,
    is_indic_language,
    get_language_code,
    get_display_name,
    detect_target_language_manually
)
from utils.audio import record_audio, play_beep
from utils.ui import Spinner, countdown_alert


# ─────────────────────────────────────────────
# 🔒 Global cached models (lazy-loaded)
# ─────────────────────────────────────────────
_whisper_model = None
_indic_model = None


def unload_asr():
    """
    Explicitly dereferences Whisper and IndicConformer models and triggers
    cyclic garbage collection to free memory on Apple Silicon / CPU.
    """
    global _whisper_model, _indic_model
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
    if _indic_model is not None:
        del _indic_model
        _indic_model = None
    import gc
    gc.collect()



# ─────────────────────────────────────────────
# 🎤 Whisper ASR (Primary / Authoritative)
# ─────────────────────────────────────────────
def get_whisper_model(verbose: bool = False):
    global _whisper_model
    if _whisper_model is None:
        if verbose:
            print("🔁 Loading Whisper model...")
        _whisper_model = whisper.load_model("small").to(device)
        _whisper_model.eval()
    return _whisper_model


# ─────────────────────────────────────────────
# 🪷 IndicConformer (Detection / Experimental)
# ─────────────────────────────────────────────
def get_indic_model(verbose: bool = False):
    global _indic_model
    if _indic_model is None:
        if verbose:
            print("🔁 Loading IndicConformer model...")

        # Configure ONNX Runtime intra-op thread pool to 4 compute threads to prevent CPU oversubscription
        orig_ort_init = None
        try:
            import onnxruntime as ort
            orig_ort_init = ort.InferenceSession.__init__
            def _threaded_ort_init(self, path_or_bytes, *args, **kwargs):
                sess_opts = kwargs.get("sess_options", None)
                if sess_opts is None:
                    if len(args) > 0 and isinstance(args[0], ort.SessionOptions):
                        sess_opts = args[0]
                    else:
                        sess_opts = ort.SessionOptions()
                        kwargs["sess_options"] = sess_opts
                sess_opts.intra_op_num_threads = 4
                sess_opts.inter_op_num_threads = 1
                return orig_ort_init(self, path_or_bytes, *args, **kwargs)
            ort.InferenceSession.__init__ = _threaded_ort_init
        except Exception:
            orig_ort_init = None

        try:
            with open(os.devnull, "w") as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    try:
                        _indic_model = AutoModel.from_pretrained(
                            "ai4bharat/indic-conformer-600m-multilingual",
                            trust_remote_code=True,
                            local_files_only=True
                        ).to(device)
                    except Exception:
                        _indic_model = AutoModel.from_pretrained(
                            "ai4bharat/indic-conformer-600m-multilingual",
                            trust_remote_code=True
                        ).to(device)
                    _indic_model.eval()
        finally:
            if orig_ort_init is not None:
                import onnxruntime as ort
                ort.InferenceSession.__init__ = orig_ort_init

    return _indic_model


# ─────────────────────────────────────────────
# 🌐 Language Detection (Whisper-based)
# ─────────────────────────────────────────────
def detect_input_language_whisper(
    audio_input,
    model,
    candidate_langs=None,
    verbose: bool = False,
    is_research_mode: bool = False,
    return_metadata: bool = False
):
    """
    Detects language using Whisper with confidence distribution tracking.
    Supports three distinct routing contracts:
      1. Runtime Pair Mode (when candidate_langs contains exactly 2 codes):
         Performs direct pairwise discrimination between Candidate A and Candidate B,
         with out-of-pair acoustic anomaly detection (advisory warning/telemetry) over the 9-language research universe.
      2. Explicit Research Mode (--research):
         Constrains softmax strictly to the 8 published FLEURS languages:
         L = {hi, te, ta, bn, fr, de, ar, ja} per Nature Scientific Reports (2026) 16:36511 Eq. 1 & 2.
      3. Normal Automatic Runtime Mode (Single-Turn):
         Evaluates the 9-language research universe (8 FLEURS + English), allowing English
         input speech to be accurately recognized as English without misclassification.
    Accepts either an audio file path (str) or a pre-computed Mel spectrogram tensor.
    Returns (detected_code, lang_name, is_indic, mel, routing_confidence) by default,
    or (detected_code, lang_name, is_indic, mel, routing_confidence, out_of_pair_info, confidence_type)
    when return_metadata=True.
    """
    if isinstance(audio_input, str):
        from utils.audio import trim_silence_rms
        audio = whisper.load_audio(audio_input)
        audio = trim_silence_rms(audio, sample_rate=16000)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
    else:
        mel = audio_input.to(model.device)

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    with torch.inference_mode():
        mel_input = mel.unsqueeze(0) if mel.ndim == 2 else mel
        audio_features = model.encoder(mel_input)
        sot = torch.tensor([[tokenizer.sot]], device=model.device)
        logits = model.decoder(sot, audio_features)[:, 0]

    # Resolve candidate set L
    if candidate_langs:
        raw_candidates = list(candidate_langs)
        valid_cands = []
        cand_tokens = []
        for c in raw_candidates:
            c_clean = c.lower().strip()
            if c_clean not in tokenizer.all_language_codes:
                raise ValueError(f"Candidate language code '{c}' is not supported by Whisper.")
            if c_clean not in valid_cands:
                valid_cands.append(c_clean)
                cand_tokens.append(tokenizer.to_language_token(c_clean))
    else:
        from utils.language import RESEARCH_LID_LANGS, RUNTIME_LID_LANGS
        target_universe = RESEARCH_LID_LANGS if is_research_mode else RUNTIME_LID_LANGS
        missing = [c for c in target_universe if c not in tokenizer.all_language_codes]
        if missing:
            raise RuntimeError(
                f"Candidate universe integrity violation: languages {missing} "
                f"from {'RESEARCH_LID_LANGS' if is_research_mode else 'RUNTIME_LID_LANGS'} "
                f"are missing from Whisper tokenizer vocabulary."
            )
        valid_cands = sorted(target_universe)
        assert len(valid_cands) == (8 if is_research_mode else 9), (
            f"Expected exactly {8 if is_research_mode else 9} candidates, got {len(valid_cands)}: {valid_cands}"
        )
        cand_tokens = [tokenizer.to_language_token(c) for c in valid_cands]

    # ─────────────────────────────────────────────────────────────
    # Contract 2: Runtime Pair Mode (Exactly 2 configured languages)
    # ─────────────────────────────────────────────────────────────
    is_dialogue_pair = bool(candidate_langs and len(valid_cands) == 2)
    if is_dialogue_pair:
        cand_A, cand_B = valid_cands[0], valid_cands[1]
        tok_A = tokenizer.to_language_token(cand_A)
        tok_B = tokenizer.to_language_token(cand_B)

        # 1. Direct pairwise normalization between the two configured candidates
        pair_logits = logits[0, [tok_A, tok_B]]
        pair_probs = torch.softmax(pair_logits, dim=-1)
        p_A = pair_probs[0].item()
        p_B = pair_probs[1].item()

        # 2. Out-of-pair acoustic anomaly detection across the 9-language research NMT candidate universe (RESEARCH_NMT_LANGS)
        from utils.language import RESEARCH_NMT_LANGS
        guard_cands = [c for c in sorted(RESEARCH_NMT_LANGS) if c in tokenizer.all_language_codes]
        guard_tokens = [tokenizer.to_language_token(c) for c in guard_cands]
        guard_probs = torch.softmax(logits[0, guard_tokens], dim=-1)
        top_guard_idx = torch.argmax(guard_probs).item()
        top_guard_lang = guard_cands[top_guard_idx]
        top_guard_prob = guard_probs[top_guard_idx].item()

        out_of_pair_info = None
        if top_guard_lang not in (cand_A, cand_B) and top_guard_prob > 0.60:
            out_of_pair_info = {
                "detected_lang": top_guard_lang,
                "confidence": top_guard_prob
            }
            if verbose:
                print(
                    f"⚠️ Research-Universe Out-of-Pair Alert: Input audio matches unconfigured language '{top_guard_lang}' "
                    f"(conf={top_guard_prob:.1%}) within the 9-language research candidate universe, "
                    f"outside configured dialogue pair ['{cand_A}', '{cand_B}']."
                )

        # 3. Closed-world binary decision between paired participants
        if p_A >= p_B:
            detected_code = cand_A
            routing_confidence = p_A
        else:
            detected_code = cand_B
            routing_confidence = p_B

        is_indian = is_paper_indic_language(detected_code) or (detected_code in INDIAN_LANG_MAP)
        lang_name = get_display_name(detected_code)
        if verbose:
            print(
                f"🌐 Dialogue Pair LID: '{cand_A}' ({p_A:.1%}) vs '{cand_B}' ({p_B:.1%}) "
                f"➔ Selected: {detected_code} ({lang_name})"
            )
        confidence_type = "pairwise_relative"
        if return_metadata:
            return detected_code, lang_name, is_indian, mel, routing_confidence, out_of_pair_info, confidence_type
        return detected_code, lang_name, is_indian, mel, routing_confidence

    # ─────────────────────────────────────────────────────────────
    # Contract 1 & 3: Research Baseline Mode & Normal Runtime Mode
    # ─────────────────────────────────────────────────────────────
    cand_logits = logits[0, cand_tokens]
    # Eq. (1): p(l|x) = exp(z_l) / sum_{j in L} exp(z_j)
    cand_probs = torch.softmax(cand_logits, dim=-1)
    sorted_indices = torch.argsort(cand_probs, descending=True)
    top_idx = sorted_indices[0].item()

    # Eq. (2): l_hat = argmax_{l in L} p(l|x),  c(x) = max_{l in L} p(l|x)
    detected_code = valid_cands[top_idx]
    routing_confidence = cand_probs[top_idx].item()
    is_indian = is_paper_indic_language(detected_code) or (detected_code in INDIAN_LANG_MAP)
    lang_name = get_display_name(detected_code)

    if verbose:
        runner_up_code = valid_cands[sorted_indices[1].item()] if len(valid_cands) > 1 else None
        runner_up_prob = cand_probs[sorted_indices[1].item()].item() if len(valid_cands) > 1 else 0.0
        margin = routing_confidence - runner_up_prob
        mode_label = "Research Baseline (8-lang)" if is_research_mode else "Normal Runtime (9-lang)"
        print(f"🌐 Detected Language [{mode_label}]: {lang_name} ({detected_code}) (c={routing_confidence:.1%}, margin={margin:.1%})")

    confidence_type = "research_candidate_softmax" if is_research_mode else "runtime_candidate_softmax"
    if return_metadata:
        return detected_code, lang_name, is_indian, mel, routing_confidence, None, confidence_type
    return detected_code, lang_name, is_indian, mel, routing_confidence


# ─────────────────────────────────────────────
# 🛡️ Anti-Hallucination & Repetition Suppression
# ─────────────────────────────────────────────
def suppress_repetition(text: str) -> str:
    """
    Collapses repetitive token loops, repeating n-gram phrases,
    and runaway grapheme repetitions caused by Whisper hallucinations.
    """
    if not text or not text.strip():
        return ""

    # Strip replacement chars
    cleaned = text.replace("\ufffd", "").strip()
    # Collapse runaway non-digit grapheme/character repetitions (4+ repeated units)
    # Excludes digits to protect numbers like 1000000 from corruption
    cleaned = re.sub(r'([^\d\s]{1,3}?)\1{3,}', r'\1', cleaned)

    words = cleaned.split()
    if not words:
        return ""

    # 1. Collapse consecutive identical words: e.g. ['a', 'a', 'a'] -> ['a']
    deduped = []
    prev_word = None
    for w in words:
        if w != prev_word:
            deduped.append(w)
            prev_word = w

    # 2. Collapse repeating multi-word phrases (e.g. 2-gram to 8-gram cycles)
    n = len(deduped)
    for k in range(2, min(n // 2 + 1, 9)):
        if n >= 2 * k:
            blocks = [deduped[i:i+k] for i in range(0, n - (n % k), k)]
            if len(blocks) >= 2 and all(b == blocks[0] for b in blocks):
                deduped = blocks[0]
                break

    return " ".join(deduped)


def normalize_telugu_vowel_repetition(text: str) -> str:
    """
    Collapses consecutive duplicate occurrences of combining vowel signs across
    Telugu ([\u0c3e-\u0c4d]) and Devanagari ([\u093e-\u094d]).
    Leaves base consonants, geminated consonants (e.g. 'అమ్మ', 'पत्ता'),
    numerals ('1000'), punctuation, and unrelated scripts untouched.
    """
    if not text:
        return text
    # Telugu combining vowel signs U+0C3E to U+0C4D
    text = re.sub(r"([\u0c3e-\u0c4d])\1+", r"\1", text)
    # Devanagari combining vowel signs U+093E to U+094D
    text = re.sub(r"([\u093e-\u094d])\1+", r"\1", text)
    return text


# ─────────────────────────────────────────────
# 📝 Whisper Transcription
# ─────────────────────────────────────────────
def whisper_transcribe(audio_path: str, model, lang_code: str = "en", mel=None):
    """
    Transcribes audio using Whisper with explicit language decoding.
    Reuses pre-computed Mel spectrogram if provided to avoid redundant I/O and FFTs.
    Follows Table 4 decoding settings: beam_size=5, batch_size=1.
    Guards against hallucination loops via compression_ratio threshold and repetition suppression.
    """
    if mel is None:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(
        language=lang_code,
        fp16=False,
        suppress_blank=True,
        beam_size=5,
        sample_len=96
    )
    result = whisper.decode(model, mel, options)

    text = result.text.strip()
    compression_ratio = getattr(result, "compression_ratio", 1.0)
    no_speech_prob = getattr(result, "no_speech_prob", 0.0)
    avg_logprob = getattr(result, "avg_logprob", 0.0)

    # Check for repetitive hallucination loop (Whisper standard threshold is 2.4)
    if compression_ratio > 2.4:
        print(f"⚠️ Whisper loop detected (compression_ratio={compression_ratio:.2f} > 2.4). Suppressing repetition...")
        text = suppress_repetition(text)

    # Silence hallucination guard (Whisper official standard: no_speech_prob > 0.6 AND avg_logprob < -1.0)
    # Also catches degenerate repetitive loops on background noise
    if (no_speech_prob > 0.60 and avg_logprob < -1.0) or (compression_ratio > 2.4 and no_speech_prob > 0.70):
        text = ""

    return text


# ─────────────────────────────────────────────
# 🪷 IndicConformer Transcription
# ─────────────────────────────────────────────
def indic_transcribe(
    audio_path: str,
    lang_code: str = "hi",
    verbose: bool = False,
    candidate_langs: list = None,
    return_lang: bool = False
):
    """
    Transcribe using IndicConformer for Indian languages on raw 16kHz audio.
    Matches published Nature Scientific Reports paper baseline (Eq. 4).
    If candidate_langs is provided (or if lang_code is in the 4 Indic languages),
    uses IndicConformer CTC decoder likelihood to ensure the exact matching language mask.
    """
    indic_model = get_indic_model(verbose=verbose)

    # Load audio at 16kHz (IndicConformer requirement)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Move to device
    waveform = waveform.to(device)

    # 🎯 Disambiguation safeguard (Nature Scientific Reports Eq. 3 & Eq. 4):
    # Only run multi-candidate CTC scoring when candidate_langs is explicitly provided with > 1 candidates
    # (e.g. Indic ↔ Indic dialogue pairs like ["hi", "te"] to prevent Whisper inter-Indic acoustic leakage).
    # When running in --pair en:te (1 Indic, 1 Non-Indic) or --source te (Source Locked), candidate_langs
    # is None or single-item: skip the multi-candidate loop and decode lang_code directly in a single pass!
    cands = [c for c in candidate_langs if c in indic_model.language_masks] if (candidate_langs and len(candidate_langs) > 1) else None
    if cands and len(cands) > 1 and hasattr(indic_model, "models") and "ctc_decoder" in indic_model.models:
        try:
            with torch.no_grad():
                enc_out, enc_len = indic_model.encode(waveform)
                raw_lp = indic_model.models['ctc_decoder'].run(['logprobs'], {'encoder_output': enc_out})[0]
                scores = {}
                for c in cands:
                    if c in indic_model.language_masks:
                        lp = torch.from_numpy(raw_lp[:, :, indic_model.language_masks[c]]).log_softmax(dim=-1)
                        max_lp, idx = torch.max(lp[0], dim=-1)
                        mask = idx != indic_model.config.BLANK_ID
                        scores[c] = max_lp[mask].mean().item() if mask.sum() > 0 else -999.0
                best_lang = max(scores, key=scores.get)
                lp_best = torch.from_numpy(raw_lp[:, :, indic_model.language_masks[best_lang]]).log_softmax(dim=-1)
                indices = torch.argmax(lp_best[0], dim=-1)
                collapsed = torch.unique_consecutive(indices, dim=-1)
                transcription = ''.join([indic_model.vocab[best_lang][x] for x in collapsed if x != indic_model.config.BLANK_ID]).replace('▁', ' ').strip()
                # 🪷 Narrowest safe normalization for confirmed CTC vowel sign OO repetition artifact
                transcription = normalize_telugu_vowel_repetition(transcription)
                if verbose and best_lang != lang_code:
                    print(f"🪷 IndicConformer refined language token: '{lang_code}' ➔ '{best_lang}' (mean nonblank framewise CTC heuristic score: {scores[best_lang]:.3f})")
                if return_lang:
                    return transcription, best_lang
                return transcription
        except Exception as e:
            if verbose:
                print(f"⚠️ Multi-candidate CTC decode fallback ({e})")

    # Direct CTC call (Paper Eq. 4)
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            with torch.no_grad():
                transcription = indic_model(waveform, lang_code, "ctc")

    res_text = transcription[0] if isinstance(transcription, list) and len(transcription) > 0 else (transcription or "")
    # 🪷 Narrowest safe normalization for confirmed CTC vowel sign OO repetition artifact
    res_text = normalize_telugu_vowel_repetition(res_text)
    if return_lang:
        return res_text, lang_code
    return res_text


# ─────────────────────────────────────────────
# 🌐 Isolated Whisper Language Identification (LID)
# ─────────────────────────────────────────────
def detect_language_whisper(
    audio_path: str,
    candidate_langs=None,
    verbose: bool = False,
    is_dialogue: bool = False,
    is_research_mode: bool = False,
    return_metadata: bool = False
):
    """
    Isolated Whisper Language Identification (LID).
    Measures pure LID compute latency and determines confidence routing destination.
    Implements:
      1. Runtime Pair Mode (is_dialogue or 2 candidates): Direct backend routing
         (IndicConformer for Indic, Whisper for Non-Indic) without dead-zone fallback.
      2. Explicit Research Mode (--research): Nature Scientific Reports (2026) 16:36511 Eq. (1), (2), (3) with tau=0.70
         evaluated strictly across the 8 published FLEURS languages.
      3. Normal Automatic Runtime Mode (Single-Turn): Evaluated across the 9 research languages (8 FLEURS + English).
    Returns 7-tuple: (lang_code, lang_name, is_indic, mel, confidence, routed_backend, lid_time_sec) by default,
    or 9-tuple: (..., out_of_pair_info, confidence_type) if return_metadata=True.
    """
    whisper_model = get_whisper_model(verbose=verbose)
    t0 = perf_counter()
    lang_code, lang_name, is_indic, mel, confidence, out_of_pair_info, conf_type = detect_input_language_whisper(
        audio_path,
        whisper_model,
        candidate_langs=candidate_langs,
        verbose=verbose,
        is_research_mode=is_research_mode,
        return_metadata=True
    )
    lid_time_sec = perf_counter() - t0

    is_pair_mode = is_dialogue or bool(candidate_langs and len(candidate_langs) == 2)
    if is_pair_mode:
        if is_indic:
            routed_backend = "IndicConformer-600M"
        else:
            routed_backend = "Whisper-small"
    else:
        # 🎯 Confidence-Aware ASR Routing (Nature Scientific Reports 2026, Eq. 3)
        # Operating threshold tau=0.70 belongs to the 8-language research baseline.
        tau = 0.70
        if is_indic and confidence >= tau:
            routed_backend = "IndicConformer-600M"
        elif is_indic and confidence < tau:
            routed_backend = "Whisper-small (Low-Confidence Indic Fallback)"
        else:
            routed_backend = "Whisper-small"

    if return_metadata:
        return lang_code, lang_name, is_indic, mel, confidence, routed_backend, lid_time_sec, out_of_pair_info, conf_type
    return lang_code, lang_name, is_indic, mel, confidence, routed_backend, lid_time_sec


# ─────────────────────────────────────────────
# 📝 Isolated Speech Transcription Stage
# ─────────────────────────────────────────────
def transcribe_speech_routed(
    audio_path: str,
    lang_code: str,
    lang_name: str,
    is_indic: bool,
    mel=None,
    confidence: float = 1.0,
    verbose: bool = False,
    return_refined_lang: bool = False,
    candidate_langs: list = None,
    is_dialogue: bool = False
):
    """
    Isolated Speech Transcription Stage.
    Executes routed model:
      - In Pair Mode (is_dialogue or 2 candidates): uses IndicConformer directly for Indic
        speech without applying the research tau=0.70 threshold.
      - In Research Mode: enforces tau=0.70 (Nature Sci Rep 2026 Eq. 3) where c < 0.70 falls back to Whisper.
    Returns: (transcription, routed_model, transcription_time_sec)
    Or (transcription, routed_model, transcription_time_sec, actual_lang) if return_refined_lang=True.
    """
    tau = 0.70
    is_pair_mode = is_dialogue or bool(candidate_langs and len(candidate_langs) == 2)
    use_indic = is_indic and (is_pair_mode or confidence >= tau)

    if use_indic:
        _ = get_indic_model(verbose=verbose)

    t0 = perf_counter()
    actual_lang = lang_code

    if use_indic:
        routed_model = "IndicConformer-600M"
        if verbose:
            if is_pair_mode:
                print(f"🛤️ Dialogue Pair Indic speech detected ({lang_name}, direct pairwise confidence {confidence:.1%}).")
            else:
                print(f"🛤️ High-confidence Indic speech ({lang_name}, c={confidence:.1%} ≥ τ={tau}).")
            print("🪷 Using IndicConformer for transcription...")
        try:
            trans_res = indic_transcribe(
                audio_path,
                lang_code,
                verbose=verbose,
                candidate_langs=candidate_langs,
                return_lang=True
            )
            if isinstance(trans_res, tuple):
                transcription, actual_lang = trans_res
            else:
                transcription, actual_lang = trans_res, lang_code

            if not transcription or not transcription.strip():
                if verbose:
                    print("⚠️ IndicConformer returned empty transcription. Attempting Whisper fallback...")
                whisper_model = get_whisper_model(verbose=verbose)
                transcription = whisper_transcribe(
                    audio_path, whisper_model, lang_code=lang_code, mel=mel
                )
                actual_lang = lang_code
                routed_model = "Whisper-small (Conformer Empty Fallback)"
        except Exception as e:
            if verbose:
                print(f"⚠️ IndicConformer failed ({e}). Falling back to Whisper...")
            whisper_model = get_whisper_model(verbose=verbose)
            transcription = whisper_transcribe(
                audio_path, whisper_model, lang_code=lang_code, mel=mel
            )
            actual_lang = lang_code
            routed_model = "Whisper-small (Conformer Error Fallback)"
    elif is_indic and confidence < tau:
        # Research Mode low-confidence fallback
        routed_model = "Whisper-small (Low-Confidence Indic Fallback)"
        if verbose:
            print(f"⚠️ Indic speech below confidence threshold ({lang_name}, c={confidence:.1%} < τ={tau}). Falling back to Whisper...")
        whisper_model = get_whisper_model(verbose=verbose)
        transcription = whisper_transcribe(
            audio_path, whisper_model, lang_code=lang_code, mel=mel
        )
    else:
        routed_model = "Whisper-small"
        if verbose:
            print(f"🌍 Non-Indic language detected ({lang_code}, c={confidence:.1%}). Using Whisper-small...")
        whisper_model = get_whisper_model(verbose=verbose)
        transcription = whisper_transcribe(
            audio_path, whisper_model, lang_code=lang_code, mel=mel
        )

    transcription_time_sec = perf_counter() - t0

    if verbose:
        print(f"📝 Final Transcription: {transcription}")

    if return_refined_lang:
        return transcription, routed_model, transcription_time_sec, actual_lang
    return transcription, routed_model, transcription_time_sec


# ─────────────────────────────────────────────
# 🔀 Unified Transcription Router (Backward-Compatible)
# ─────────────────────────────────────────────
def transcribe_audio(audio_path: str, return_timings: bool = False, verbose: bool = False, on_lid_done=None):
    """
    Unified Transcription Router (backward-compatible).
    Calls isolated detect_language_whisper followed by transcribe_speech_routed.
    """
    lang_code, lang_name, is_indic, mel, confidence, routed_backend, lid_time_sec = detect_language_whisper(
        audio_path, verbose=verbose
    )

    if on_lid_done is not None:
        try:
            on_lid_done(lang_code, lang_name, is_indic, confidence, routed_backend, lid_time_sec)
        except Exception:
            pass

    transcription, routed_model, transcription_time_sec = transcribe_speech_routed(
        audio_path, lang_code, lang_name, is_indic, mel=mel, confidence=confidence, verbose=verbose
    )

    if return_timings:
        sub_timings = {
            "lid_time_sec": lid_time_sec,
            "transcription_time_sec": transcription_time_sec,
            "confidence": float(confidence),
            "threshold": 0.70,
            "routed_model": routed_model
        }
        return transcription, lang_code, lang_name, is_indic, sub_timings

    return transcription, lang_code, lang_name, is_indic


# ─────────────────────────────────────────────
# 🎯 Target Language Detection (Voice-based)
# ─────────────────────────────────────────────
def detect_target_language_by_voice(attempts=2, verbose: bool = False):
    whisper_model = get_whisper_model(verbose=verbose)

    # Contextual vocabulary conditioning prompt to constrain Whisper decoding
    target_prompt = "Languages: English, Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, German, French, Spanish, Japanese, Russian."

    for attempt in range(attempts):
        if verbose:
            print(
                f"🗣️ Attempt {attempt + 1}/{attempts}: "
                "Speak target language name (e.g., Telugu, Hindi, Tamil, German)"
            )

        if not verbose:
            countdown_alert(seconds=2, label="Target language cue in")
        play_beep()
        file_path = record_audio(
            duration=1.4,
            filename=f"target_attempt_{attempt + 1}.wav",
            show_progress=(not verbose),
            verbose=verbose
        )

        with Spinner("Interpreting target language cue...", enabled=not verbose) as sp_dec:
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

            options = whisper.DecodingOptions(fp16=False, prompt=target_prompt, sample_len=32)
            result = whisper.decode(whisper_model, mel, options)
            spoken = result.text.strip()
            sp_dec.stop()

        if verbose:
            print(f"📝 You said: \"{spoken}\"")

        lang_code, lang_name = get_language_code(spoken)
        if lang_code:
            if verbose:
                print(f"✅ Interpreted as: {lang_name.capitalize()} ({lang_code})")
            return lang_code, lang_name

        if verbose:
            print("⚠️ Could not resolve target language. Try again...\n")

    return None, None


# ─────────────────────────────────────────────
# 🧭 Unified Target Language Selector
# ─────────────────────────────────────────────
def get_target_language(verbose: bool = False):
    if verbose:
        print("🎯 Target Language Selection Started")

    code, name = detect_target_language_by_voice(verbose=verbose)

    if not code:
        if verbose:
            print("🔁 Switching to manual input...")
        code, name = detect_target_language_manually()

    if code:
        if verbose:
            print(f"✅ Final Target Language: {name.capitalize()} ({code})")
    else:
        if verbose:
            print("❌ Language not supported or selection cancelled.")

    return code, name
