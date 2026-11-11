import os
import sys

# 🛡️ Force UTF-8 stream encoding across standard I/O (macOS / Linux terminal safe)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore")

import json
import datetime
import gc
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("parler_tts").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import shutil
import platform
import resource
import psutil
import threading
import time
from time import perf_counter

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.system import device, physical_cores, configured_compute_threads, get_system_info
from utils.language import (
    INDIAN_LANG_MAP,
    DISPLAY_LANG_MAP,
    PAPER_LANG_SET,
    PAPER_INDIC_SET,
    PAPER_NON_INDIC_SET,
    PAPER_LID_ASR_LANGS,
    PAPER_NMT_LANGS,
    PAPER_TTS_LANGS,
    PAPER_PRINCIPAL_INDIC_LANGS,
    RESEARCH_LID_LANGS,
    RUNTIME_LID_LANGS,
    is_paper_indic_language,
    is_indic_language,
    get_display_name,
    detect_target_language_manually,
    validate_pair_syntax,
    validate_target_language,
    validate_source_language,
    validate_language_arguments
)
from utils.audio import record_audio, play_beep, check_audio_silence
from utils.ui import StudioCanvas, Spinner, typewrite_text, countdown_alert

from models.asr import (
    transcribe_audio,
    detect_language_whisper,
    transcribe_speech_routed,
    get_target_language,
    unload_asr
)
from models.translation import translate_with_nllb, get_nllb_pipeline, unload_nllb
from models.tts import synthesize_tts, playback_tts, play_tts_output, load_parler_tts, unload_tts, get_tts_engine, get_system_voice_for_lang



# ─────────────────────────────────────────────
# 🔄 Single-Turn Execution Worker (Function-Scoped)
# ─────────────────────────────────────────────
def process_single_turn(
    turn_index: int = 1,
    duration: float = 2.5,
    output_log: str = "evaluation_log.json",
    input_audio_path: str = None,
    source_lang: str = None,
    target_lang: str = None,
    pair_langs: tuple = None,
    use_vad: bool = False,
    is_warm: bool = False,
    cold_init_time_sec: float = 0.0,
    stage_and_flush: bool = True,
    on_stage_update: callable = None,
    verbose: bool = False,
    is_research_mode: bool = False
) -> dict:
    """
    Executes a single conversational turn with functional scoping.
    When stage_and_flush=True, each model is explicitly unloaded and its memory
    reclaimed after its processing stage, minimizing persistent memory overlap;
    peak transient memory remains workload-dependent.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    timings = {}
    outputs = {}
    start_total = perf_counter()

    # PID-isolated process & memory tracking
    process = psutil.Process()
    cpu_times_start = process.cpu_times()
    rss_samples = [process.memory_info().rss]

    canvas = StudioCanvas(stage_and_flush=stage_and_flush, enabled=not verbose)
    if not verbose:
        canvas.print_header()

    # ─── Phase 1: Audio Capture (I/O) ─────────────────────────
    if on_stage_update:
        on_stage_update("audio_capture")

    if input_audio_path is None:
        if not verbose:
            countdown_alert(seconds=3)
        play_beep()
        t0 = perf_counter()
        audio_path = record_audio(
            duration=duration,
            filename="input.wav",
            vad=use_vad,
            show_progress=(not verbose),
            verbose=verbose
        )
        timings["recording_time_sec"] = perf_counter() - t0
        if not verbose:
            canvas.print_step("Input Audio", f"Recorded {duration:.2f}s speech (16 kHz mono)")

        # 🔇 Pre-ASR Silence Gate: reject live recordings with zero speech energy (< -48 dBFS)
        is_silent, peak_db = check_audio_silence(audio_path, threshold_db=-48.0)
        if is_silent:
            if not verbose:
                canvas.print_step("Input Audio", f"⚠️ No speech detected (peak: {peak_db:.1f} dBFS < -48 dBFS). Turn aborted cleanly.")
            else:
                print(f"\n⚠️ No speech detected in recorded audio (peak: {peak_db:.1f} dBFS < -48.0 dBFS). Turn aborted cleanly.")

            timings["asr_time_sec"] = 0.0
            timings["translation_time_sec"] = 0.0
            timings["tts_synthesis_time_sec"] = 0.0
            timings["playback_time_sec"] = 0.0
            timings["pure_machine_time_sec"] = 0.0
            timings["total_pipeline_time_sec"] = timings["recording_time_sec"]
            outputs["status"] = "silence"
            outputs["reason"] = f"No speech detected (peak: {peak_db:.1f} dBFS < -48 dBFS)"
            if on_stage_update:
                on_stage_update("done")
            return {
                "status": "silence",
                "turn_index": turn_index,
                "timings": timings,
                "outputs": outputs,
                "peak_db": peak_db
            }
    else:
        audio_path = input_audio_path
        timings["recording_time_sec"] = 0.0
        if not os.path.exists(audio_path):
            if not verbose:
                canvas.print_step("Input Audio", f"❌ Audio file not found: {audio_path}")
            else:
                print(f"\n❌ Error: Input audio file not found: {audio_path}")
            outputs["status"] = "error"
            outputs["reason"] = f"Audio file not found: {audio_path}"
            total_runtime = perf_counter() - start_total
            timings["total_pipeline_time_sec"] = total_runtime
            evaluation_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "turn_index": turn_index,
                "status": "error",
                "is_warm": is_warm,
                "device": device,
                "system_environment": get_system_info(),
                "timings": timings,
                "outputs": outputs,
                "system_metrics": {
                    "status": "error",
                    "total_pipeline_time_sec": total_runtime
                }
            }
            _save_evaluation_log(output_log, evaluation_entry)
            if on_stage_update:
                on_stage_update("done")
            return evaluation_entry

        if verbose:
            print(f"📁 Using pre-recorded audio input: {audio_path}")
        if not verbose:
            canvas.print_step("Input Audio", f"Loaded audio ({os.path.basename(audio_path)})")

        # 🔇 Pre-ASR Silence Gate: reject pre-recorded input files with zero speech energy (< -48 dBFS)
        is_silent, peak_db = check_audio_silence(audio_path, threshold_db=-48.0)
        if is_silent:
            if not verbose:
                canvas.print_step("Input Audio", f"⚠️ Audio file contains only silence (peak: {peak_db:.1f} dBFS < -48 dBFS). Turn aborted cleanly.")
            else:
                print(f"\n⚠️ Audio file contains only silence (peak: {peak_db:.1f} dBFS < -48.0 dBFS). Turn aborted cleanly.")

            timings["asr_time_sec"] = 0.0
            timings["translation_time_sec"] = 0.0
            timings["tts_synthesis_time_sec"] = 0.0
            timings["playback_time_sec"] = 0.0
            timings["pure_machine_time_sec"] = 0.0
            timings["total_pipeline_time_sec"] = 0.0
            outputs["status"] = "silence"
            outputs["reason"] = f"No speech detected in audio file (peak: {peak_db:.1f} dBFS < -48 dBFS)"
            if on_stage_update:
                on_stage_update("done")
            return {
                "status": "silence",
                "turn_index": turn_index,
                "timings": timings,
                "outputs": outputs,
                "peak_db": peak_db
            }

    rss_samples.append(process.memory_info().rss)

    # ─── Phase 2: ASR (LID + Decode Compute) ───────────────────
    neural_cpu_sec = 0.0

    if source_lang:
        # 🎯 Source-Locked Mode (Bypasses Whisper LID entirely)
        src_lang_code = source_lang.strip().lower()
        src_lang_name = get_display_name(src_lang_code)
        is_indic = (src_lang_code in PAPER_INDIC_SET) or (src_lang_code in INDIAN_LANG_MAP)
        confidence = None
        confidence_type = "source_locked_unspecified"
        out_of_pair_info = None
        mel = None
        routed_backend = "IndicConformer-600M" if is_indic else "Whisper-small"
        lid_time_sec = 0.0
        if not verbose:
            canvas.render_stage_sync(f"🎯 Source Locked [{src_lang_name}] ➔ ⠋ [1/3] ASR ({routed_backend})... ➔ ○ NMT ➔ ○ Voice")
        elif verbose:
            print(f"🎯 Source-Locked Language: {src_lang_name} ({src_lang_code}) ➔ {routed_backend}")
        lid_lang_code = src_lang_code
    else:
        # Step 2A: Isolated Whisper Language Identification (LID)
        # Constrained candidate set L: pair candidates, research 8-languages (--research), or runtime 9-languages
        cand_set = list(pair_langs) if pair_langs else None

        c0 = process.cpu_times()
        with Spinner("Identifying language (Whisper LID)...", enabled=not verbose) as sp_lid:
            lid_res = detect_language_whisper(
                audio_path,
                candidate_langs=cand_set,
                verbose=verbose,
                is_dialogue=bool(pair_langs),
                is_research_mode=is_research_mode,
                return_metadata=True
            )
            sp_lid.stop()
        c1 = process.cpu_times()
        neural_cpu_sec += (c1.user + c1.system) - (c0.user + c0.system)

        src_lang_code, src_lang_name, is_indic, mel, confidence, routed_backend, lid_time_sec, out_of_pair_info, conf_type = lid_res
        confidence_type = conf_type
        conf_pct = (confidence * 100) if confidence is not None else 100.0

        lid_lang_code = src_lang_code
        if not verbose:
            canvas.render_stage_sync(f"✔ LID ({src_lang_name}) ➔ ⠋ [1/3] ASR ({routed_backend})... ➔ ○ NMT ➔ ○ Voice")

    # Step 2B: Isolated Transcription Decode (IndicConformer or Whisper)
    # 🎯 Safeguard: For Indic ↔ Indic pairs (e.g. hi:te), pass candidate_langs to indic_transcribe
    # to evaluate CTC log-probability between those 2 paired Indic languages.
    # For single-Indic contexts (--pair en:te or -S te), pass None to decode directly in 1 pass.
    pair_indic_cands = None
    if pair_langs and all(p in INDIAN_LANG_MAP or p in PAPER_INDIC_SET for p in pair_langs):
        pair_indic_cands = list(pair_langs)

    # Wrap IndicConformer cold SSD weight read in a visible spinner if not yet resident
    tau = 0.70
    is_pair_mode = bool(pair_langs)
    if is_indic and (is_pair_mode or (confidence is not None and confidence >= tau) or source_lang):
        from models import asr
        if getattr(asr, "_indic_model", None) is None:
            with Spinner("Loading IndicConformer-600M into RAM...", enabled=not verbose) as sp_load_indic:
                _ = asr.get_indic_model(verbose=verbose)
                sp_load_indic.stop()

    c0 = process.cpu_times()
    with Spinner(f"Transcribing {src_lang_name} speech...", enabled=not verbose) as sp_trans:
        trans_res = transcribe_speech_routed(
            audio_path,
            src_lang_code,
            src_lang_name,
            is_indic,
            mel,
            confidence if confidence is not None else 1.0,
            verbose=verbose,
            return_refined_lang=True,
            candidate_langs=pair_indic_cands,
            is_dialogue=is_pair_mode
        )
        sp_trans.stop()
    c1 = process.cpu_times()
    neural_cpu_sec += (c1.user + c1.system) - (c0.user + c0.system)

    transcription, final_routed_model, transcription_time_sec, refined_lang = trans_res
    if refined_lang and refined_lang != src_lang_code and not source_lang:
        src_lang_code = refined_lang
        src_lang_name = get_display_name(src_lang_code)
        is_indic = (src_lang_code in PAPER_INDIC_SET) or (src_lang_code in INDIAN_LANG_MAP)

    # 🛑 Guard: Reject empty or whitespace-only transcriptions
    if not transcription or not transcription.strip():
        if stage_and_flush:
            unload_asr()
        if not verbose:
            canvas.print_step("Spoken Input", "⚠️ No intelligible speech detected (empty transcription). Turn aborted cleanly.")
        else:
            print("\n⚠️ No intelligible speech detected (empty transcription). Turn aborted cleanly.")
        timings["asr_time_sec"] = lid_time_sec + transcription_time_sec
        timings["lid_time_sec"] = lid_time_sec
        timings["transcription_time_sec"] = transcription_time_sec
        timings["translation_time_sec"] = 0.0
        timings["tts_synthesis_time_sec"] = 0.0
        timings["playback_time_sec"] = 0.0
        timings["pure_machine_time_sec"] = timings["asr_time_sec"]
        timings["total_pipeline_time_sec"] = perf_counter() - start_total
        outputs["status"] = "empty_transcription"
        outputs["failed_stage"] = "asr"
        outputs["reason"] = "ASR returned empty or whitespace-only transcription."
        outputs["transcription"] = ""
        outputs["source_language"] = {
            "code": src_lang_code,
            "name": src_lang_name,
            "is_indic": is_indic,
            "lid_predicted_code": lid_lang_code if not source_lang else None,
            "ctc_refined_code": refined_lang if (refined_lang and refined_lang != lid_lang_code and not source_lang) else None,
            "confidence": float(confidence) if confidence is not None else None,
            "confidence_type": confidence_type,
            "out_of_pair_detected": out_of_pair_info["detected_lang"] if out_of_pair_info else None,
            "out_of_pair_confidence": float(out_of_pair_info["confidence"]) if out_of_pair_info else None,
            "routed_model": final_routed_model,
            "dispatch_mode": "source_locked" if source_lang else ("dialogue_pair" if pair_langs else ("research_lid" if is_research_mode else "runtime_lid"))
        }
        evaluation_entry = {
            "status": "empty_transcription",
            "failed_stage": "asr",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "turn_index": turn_index,
            "is_warm": is_warm,
            "device": device,
            "system_environment": get_system_info(),
            "timings": timings,
            "outputs": outputs,
            "system_metrics": {
                "status": "empty_transcription",
                "failed_stage": "asr",
                "total_pipeline_time_sec": timings["total_pipeline_time_sec"]
            }
        }
        _save_evaluation_log(output_log, evaluation_entry)
        if on_stage_update:
            on_stage_update("done")
        return evaluation_entry

    # 🚀 IMMEDIATELY print You Said Row the millisecond decoding completes!
    if not verbose:
        canvas.clear_stage_sync()
        canvas.stream_step("You Said", transcription, prefix_quote=True, suffix_tag=f"[{src_lang_name}]")

    asr_elapsed = lid_time_sec + transcription_time_sec
    timings["asr_time_sec"] = asr_elapsed
    timings["lid_time_sec"] = lid_time_sec
    timings["transcription_time_sec"] = transcription_time_sec

    outputs["transcription"] = transcription
    outputs["source_language"] = {
        "code": src_lang_code,
        "name": src_lang_name,
        "is_indic": is_indic,
        "lid_predicted_code": lid_lang_code if not source_lang else None,
        "ctc_refined_code": refined_lang if (refined_lang and refined_lang != lid_lang_code and not source_lang) else None,
        "confidence": float(confidence) if confidence is not None else None,
        "confidence_type": confidence_type,
        "out_of_pair_detected": out_of_pair_info["detected_lang"] if out_of_pair_info else None,
        "out_of_pair_confidence": float(out_of_pair_info["confidence"]) if out_of_pair_info else None,
        "routed_model": final_routed_model,
        "dispatch_mode": "source_locked" if source_lang else ("dialogue_pair" if pair_langs else ("research_lid" if is_research_mode else "runtime_lid"))
    }

    rss_samples.append(process.memory_info().rss)

    # ─── Phase 3: Target Language Selection (I/O or Dispatch) ─
    # Note: ASR models remain in RAM across Phase 2 and Phase 3 so interactive voice target
    # selection reuses the resident Whisper instance without a redundant SSD cold reload!
    if pair_langs:
        # 💬 Bi-directional Auto-Dialogue Mode
        timings["target_lang_selection_sec"] = 0.0
        src_norm = (src_lang_code or "").lower()
        if src_norm == pair_langs[0]:
            tgt_lang_code = pair_langs[1]
        elif src_norm == pair_langs[1]:
            tgt_lang_code = pair_langs[0]
        else:
            tgt_lang_code = pair_langs[1] if pair_langs[0] in ["en", "eng_latn"] else pair_langs[0]
            if verbose:
                print(f"⚠️ Detected [{src_lang_code}] not in pair '{pair_langs[0]}:{pair_langs[1]}'. Defaulting target to [{tgt_lang_code}].")

        tgt_lang_name = get_display_name(tgt_lang_code)
        dispatch_mode = "auto_dialogue"
        if verbose:
            print(f"💬 Auto-Dialogue: [{src_lang_code}] → [{tgt_lang_code}] ({tgt_lang_name})")

    elif target_lang is not None:
        # 🎯 Target Specified Mode
        tgt_lang_code = target_lang.strip().lower()
        tgt_lang_name = get_display_name(tgt_lang_code)
        timings["target_lang_selection_sec"] = 0.0
        dispatch_mode = "target_specified"
        if verbose:
            print(f"🎯 Target Specified: {tgt_lang_name} ({tgt_lang_code})")

    else:
        # 🗣️ Interactive Voice Fallback (re-uses resident Whisper with zero SSD reload)
        t0 = perf_counter()
        tgt_lang_code, tgt_lang_name = get_target_language(verbose=verbose)
        tgt_lang_name = get_display_name(tgt_lang_code)
        timings["target_lang_selection_sec"] = perf_counter() - t0
        dispatch_mode = "interactive_fallback"

    outputs["target_language"] = {
        "code": tgt_lang_code,
        "name": tgt_lang_name,
        "dispatch_mode": dispatch_mode
    }

    if not verbose and tgt_lang_code:
        # 🌐 Compact Adaptive Pipeline Routing Proof (Paper Eq. 1-4)
        is_target_indic = (tgt_lang_code in INDIAN_LANG_MAP) or (tgt_lang_code in PAPER_INDIC_SET)
        tts_engine_name = "IndicParler-TTS" if is_target_indic else "OS Voice (pyttsx3)"
        conf_str = f" ({conf_pct:.1f}%)" if (confidence is not None and not source_lang) else ""
        compact_route = f"{src_lang_name}{conf_str} ➔ {tgt_lang_name} · {final_routed_model} ➔ NLLB-INT8 ➔ {tts_engine_name}"
        canvas.print_step("Route", f"\033[1;37m{compact_route}\033[0m")

    rss_samples.append(process.memory_info().rss)

    # 🛑 Guard: Handle aborted/cancelled target language selection
    if not tgt_lang_code:
        if stage_and_flush:
            unload_asr()
        print("\n🛑 Target language selection was cancelled or failed. Aborting pipeline execution gracefully.")
        outputs["status"] = "aborted"
        outputs["reason"] = "Target language selection failed or was cancelled by user."
        total_runtime = perf_counter() - start_total
        timings["total_pipeline_time_sec"] = total_runtime
        timings["pure_machine_time_sec"] = timings.get("asr_time_sec", 0.0)
        timings["io_subtotal_sec"] = timings.get("recording_time_sec", 0.0) + timings.get("target_lang_selection_sec", 0.0)

        evaluation_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "turn_index": turn_index,
            "is_warm": is_warm,
            "device": device,
            "system_environment": get_system_info(),
            "timings": timings,
            "outputs": outputs,
            "system_metrics": {
                "status": "aborted",
                "total_pipeline_time_sec": total_runtime
            }
        }
        _save_evaluation_log(output_log, evaluation_entry)
        return evaluation_entry

    # 🧹 Stage-and-Flush: Wipe ASR models (Whisper + IndicConformer) from RAM before NMT loads
    if stage_and_flush:
        unload_asr()
        rss_samples.append(process.memory_info().rss)

    # ─── Phase 4: Translation (NMT Compute) ───────────────────
    if on_stage_update:
        on_stage_update("translate")

    # Wrap NLLB cold SSD reload in visible spinner before starting inference stopwatch
    from models import translation
    if getattr(translation, "_nllb_pipeline", None) is None:
        with Spinner("Loading NLLB-200 (INT8) into RAM...", enabled=not verbose) as sp_load_nmt:
            _ = get_nllb_pipeline()
            sp_load_nmt.stop()

    t0 = perf_counter()
    c0 = process.cpu_times()
    with Spinner(f"Translating [{src_lang_name} ➔ {tgt_lang_name}]...", enabled=not verbose) as sp_nmt:
        translated_text = translate_with_nllb(
            transcription,
            src_lang_code=src_lang_code,
            tgt_lang_code=tgt_lang_code,
            stream=False,
            verbose=verbose
        )
        sp_nmt.stop()
    timings["translation_time_sec"] = perf_counter() - t0
    c1 = process.cpu_times()
    neural_cpu_sec += (c1.user + c1.system) - (c0.user + c0.system)
    outputs["translation"] = translated_text

    # 🛑 Guard: Reject empty or whitespace-only translation
    if not translated_text or not translated_text.strip():
        if stage_and_flush:
            unload_nllb()
        if not verbose:
            canvas.print_step("Translation", "⚠️ NMT returned empty translation. Turn aborted cleanly.")
        else:
            print("\n⚠️ NMT returned empty translation. Turn aborted cleanly.")
        timings["tts_synthesis_time_sec"] = 0.0
        timings["playback_time_sec"] = 0.0
        timings["pure_machine_time_sec"] = timings["asr_time_sec"] + timings["translation_time_sec"]
        timings["total_pipeline_time_sec"] = perf_counter() - start_total
        outputs["status"] = "empty_translation"
        outputs["failed_stage"] = "nmt"
        outputs["reason"] = "NMT returned empty or whitespace-only translation."
        outputs["translation"] = ""
        evaluation_entry = {
            "status": "empty_translation",
            "failed_stage": "nmt",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "turn_index": turn_index,
            "is_warm": is_warm,
            "device": device,
            "system_environment": get_system_info(),
            "timings": timings,
            "outputs": outputs,
            "system_metrics": {
                "status": "empty_translation",
                "failed_stage": "nmt",
                "total_pipeline_time_sec": timings["total_pipeline_time_sec"]
            }
        }
        _save_evaluation_log(output_log, evaluation_entry)
        if on_stage_update:
            on_stage_update("done")
        return evaluation_entry

    # 📄 Save clean UTF-8 text for direct inspection (100% immune to terminal font rasterization bugs)
    try:
        latest_path = os.path.join(ROOT_DIR, "latest_translation.txt")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"Source [{src_lang_name}]: {transcription}\nTarget [{tgt_lang_name}]: {translated_text}\n")
    except Exception:
        pass

    if not verbose:
        canvas.clear_stage_sync()
        canvas.stream_step("Translation", translated_text, prefix_quote=True, suffix_tag=f"[{tgt_lang_name}]")
        canvas.render_stage_sync(f"✔ ASR ({timings['asr_time_sec']:.2f}s) ➔ ✔ NMT ({timings['translation_time_sec']:.2f}s) ➔ ⠋ [3/3] Synthesizing Voice...")

    rss_samples.append(process.memory_info().rss)

    # 🧹 Stage-and-Flush: Wipe NMT model from RAM
    if stage_and_flush:
        unload_nllb()
        rss_samples.append(process.memory_info().rss)

    # ─── Phase 5: TTS Synthesis (Compute Latency) ─────────────
    if on_stage_update:
        on_stage_update("synthesize")

    lang_type = (
        "indian"
        if (is_paper_indic_language(tgt_lang_code) or tgt_lang_code in PAPER_INDIC_SET or tgt_lang_code in INDIAN_LANG_MAP)
        else "international"
    )
    if lang_type == "indian":
        from models import tts
        if getattr(tts, "_parler_model", None) is None:
            with Spinner("Loading Indic-Parler-TTS into RAM...", enabled=not verbose) as sp_load_tts:
                _ = load_parler_tts()
                sp_load_tts.stop()

    t0 = perf_counter()
    c0 = process.cpu_times()
    with Spinner(f"Synthesizing [{tgt_lang_name}] speech...", enabled=not verbose) as sp_tts:
        audio_target, engine_type, delivery_mode = synthesize_tts(
            translated_text,
            lang_type,
            tgt_lang_code,
            turn_index=turn_index,
            prefer_fp32=True
        )
        # 🛑 Cut off the TTS synthesis timer the exact millisecond the audio file is generated
        sp_tts.stop()
    timings["tts_synthesis_time_sec"] = perf_counter() - t0
    c1 = process.cpu_times()
    neural_cpu_sec += (c1.user + c1.system) - (c0.user + c0.system)

    has_audio = bool(delivery_mode == "file_artifact" and audio_target and os.path.exists(str(audio_target)))
    outputs["tts_audio_file"] = audio_target if has_audio else None
    outputs["tts_engine"] = engine_type
    outputs["tts_delivery"] = delivery_mode
    outputs["tts_synthesis_success"] = bool(has_audio) if delivery_mode == "file_artifact" else False

    # Check if target voice fell back to default host voice
    target_voice_fallback = False
    if engine_type == "pyttsx3":
        try:
            tts_engine = get_tts_engine()
            _, target_voice_fallback = get_system_voice_for_lang(tts_engine, tgt_lang_code, return_status=True)
        except Exception:
            target_voice_fallback = False
    outputs["tts_voice_fallback"] = target_voice_fallback

    rss_samples.append(process.memory_info().rss)

    # ─── Phase 6: Audio Playback (Speaker Output I/O) ─────────
    if on_stage_update:
        on_stage_update("playback")

    playback_ok = False
    if has_audio:
        if not verbose:
            canvas.render_stage_sync(f"✔ ASR ({timings['asr_time_sec']:.2f}s) ➔ ✔ NMT ({timings['translation_time_sec']:.2f}s) ➔ 🔊 Speaking [{tgt_lang_name.capitalize()}]...")
        t0 = perf_counter()
        playback_ok = playback_tts(audio_target, engine_type, tgt_lang_code)
        timings["playback_time_sec"] = perf_counter() - t0
        timings["tts_total_time_sec"] = timings["tts_synthesis_time_sec"] + timings["playback_time_sec"]
    elif delivery_mode == "host_speaker":
        if not verbose:
            canvas.render_stage_sync(f"✔ ASR ({timings['asr_time_sec']:.2f}s) ➔ ✔ NMT ({timings['translation_time_sec']:.2f}s) ➔ 🔊 Speaking [{tgt_lang_name.capitalize()}]...")
        t0 = perf_counter()
        playback_ok = playback_tts(translated_text, engine_type, tgt_lang_code)
        timings["playback_time_sec"] = perf_counter() - t0
        timings["tts_total_time_sec"] = timings["playback_time_sec"]
    else:
        timings["playback_time_sec"] = 0.0
        timings["tts_total_time_sec"] = timings["tts_synthesis_time_sec"]

    if not verbose:
        canvas.clear_stage_sync()
        tts_name = "IndicParler-TTS" if is_target_indic else "OS Voice"
        if playback_ok:
            canvas.print_step("Audio Output", f"🔊 Spoken [{tgt_lang_name.capitalize()}] ({tts_name} · {timings['playback_time_sec']:.2f}s)")
        elif has_audio:
            canvas.print_step("Audio Output", f"🔊 Neural audio ready ({timings['tts_synthesis_time_sec']:.2f}s, playback unconfirmed)")
        else:
            canvas.print_step("Audio Output", f"⚠️ Playback unavailable ({timings['playback_time_sec']:.2f}s)", icon="\033[31m✖\033[0m")

    outputs["tts_playback_success"] = bool(playback_ok)

    rss_samples.append(process.memory_info().rss)

    # 🧹 Stage-and-Flush: Wipe TTS model from RAM
    if stage_and_flush:
        unload_tts()
        rss_samples.append(process.memory_info().rss)

    # ─── Two-Tier Timing & Subtotals ──────────────────────────
    total_runtime = perf_counter() - start_total
    timings["total_pipeline_time_sec"] = total_runtime

    io_subtotal_sec = (
        timings["recording_time_sec"]
        + timings["target_lang_selection_sec"]
        + timings["playback_time_sec"]
    )
    pure_machine_time_sec = (
        timings["asr_time_sec"]
        + timings["translation_time_sec"]
        + timings["tts_synthesis_time_sec"]
    )

    timings["io_subtotal_sec"] = io_subtotal_sec
    timings["pure_machine_time_sec"] = pure_machine_time_sec
    timings["active_compute_time_sec"] = pure_machine_time_sec
    timings["cold_init_time_sec"] = cold_init_time_sec

    # ─── 3D Memory Telemetry (PID-Isolated) ───────────────────
    current_rss_mb = process.memory_info().rss / (1024 * 1024)
    avg_rss_mb = (sum(rss_samples) / len(rss_samples)) / (1024 * 1024) if rss_samples else current_rss_mb

    rusage_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kernel_peak_ram_mb = (
        rusage_max / (1024 * 1024)
        if platform.system() == "Darwin"
        else rusage_max / 1024
    )
    peak_ram_mb = max(max(rss_samples) / (1024 * 1024), kernel_peak_ram_mb)

    # ─── CPU Metrics ──────────────────────────────────────────
    cpu_times_end = process.cpu_times()
    total_cpu_time_used = (
        (cpu_times_end.user + cpu_times_end.system)
        - (cpu_times_start.user + cpu_times_start.system)
    )

    cores_count = physical_cores or psutil.cpu_count(logical=False) or os.cpu_count() or 1
    thread_count = configured_compute_threads or 4
    overall_pipeline_cores = total_cpu_time_used / total_runtime if total_runtime > 0 else 0.0
    active_compute_cores = min(float(thread_count), neural_cpu_sec / max(pure_machine_time_sec, 1e-4))

    normalized_compute_percent = (active_compute_cores / thread_count) * 100 if thread_count else 0.0
    normalized_pipeline_percent = (overall_pipeline_cores / cores_count) * 100 if cores_count else 0.0

    # Distinguish neural audio artifact generation from actual speaker playback
    if delivery_mode == "file_artifact":
        outputs["tts_synthesis_success"] = bool(has_audio)
        outputs["tts_playback_success"] = bool(playback_ok)
        if has_audio:
            outputs["status"] = "completed"
            outputs["delivery_status"] = "delivered" if playback_ok else "playback_failed"
        else:
            outputs["status"] = "error"
            outputs["failed_stage"] = "tts"
            outputs["delivery_status"] = "synthesis_failed"
            outputs["reason"] = "Neural TTS synthesis failed to generate audio artifact."
            outputs["error_message"] = "Neural TTS synthesis failed to generate audio artifact."
    elif delivery_mode == "host_speaker":
        outputs["tts_synthesis_success"] = bool(playback_ok)
        outputs["tts_playback_success"] = bool(playback_ok)
        if playback_ok:
            outputs["status"] = "completed"
            outputs["delivery_status"] = "delivered"
        else:
            outputs["status"] = "error"
            outputs["failed_stage"] = "tts"
            outputs["delivery_status"] = "delivery_failed"
            outputs["reason"] = "Host OS TTS speech delivery failed."
            outputs["error_message"] = "Host OS TTS speech delivery failed."
    else:
        outputs["tts_synthesis_success"] = False
        outputs["tts_playback_success"] = False
        if not tgt_lang_code:
            outputs["status"] = "aborted"
            outputs["delivery_status"] = "aborted"
            outputs["reason"] = "Target language selection cancelled or missing."
        else:
            outputs["status"] = "error"
            outputs["failed_stage"] = "tts"
            outputs["delivery_status"] = "unavailable"
            outputs["reason"] = "TTS synthesis was unavailable or failed."
            outputs["error_message"] = "TTS synthesis was unavailable or failed."

    final_status = outputs["status"]
    failed_stage = outputs.get("failed_stage", None)
    delivery_status = outputs.get("delivery_status", None)

    evaluation_entry = {
        "status": final_status,
        "failed_stage": failed_stage,
        "delivery_status": delivery_status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "turn_index": turn_index,
        "is_warm": is_warm,
        "device": device,
        "stage_and_flush": stage_and_flush,
        "system_environment": get_system_info(),
        "timings": timings,
        "outputs": outputs,
        "system_metrics": {
            "status": final_status,
            "failed_stage": failed_stage,
            "delivery_status": delivery_status,
            "active_ram_mb": round(current_rss_mb, 2),
            "avg_ram_mb": round(avg_rss_mb, 2),
            "peak_ram_usage_mb": round(peak_ram_mb, 2),
            "turn_sampled_peak_ram_mb": round(max(rss_samples) / (1024 * 1024), 2),
            "process_lifetime_peak_ram_mb": round(kernel_peak_ram_mb, 2),
            "kernel_ru_maxrss_raw": rusage_max,
            "active_compute_cores": active_compute_cores,
            "overall_pipeline_cores": overall_pipeline_cores,
            "normalized_compute_percent": normalized_compute_percent,
            "normalized_pipeline_percent": normalized_pipeline_percent,
            "physical_cores": cores_count,
            "configured_compute_threads": thread_count
        }
    }

    # ─── Terminal Display (Clean UX vs Verbose Telemetry) ──────
    disp_mode = outputs.get("target_language", {}).get("dispatch_mode", "standard")
    routed_model = outputs.get("source_language", {}).get("routed_model", "ASR")

    if not verbose:
        canvas.print_footer(
            timings,
            active_ram_mb=current_rss_mb,
            avg_ram_mb=avg_rss_mb,
            peak_ram_mb=peak_ram_mb,
            cores=thread_count
        )
    else:
        # 🔬 Full Academic Diagnostic Tree (--verbose)
        warm_label = " (WARM)" if is_warm else " (COLD BOOT)"
        flush_label = " [STAGE & FLUSH ACTIVE]" if stage_and_flush else " [RESIDENT IN RAM]"
        print("\n" + "=" * 66)
        print(f"⏱️  PERFORMANCE & RESOURCE TELEMETRY (TURN {turn_index}{warm_label}){flush_label}")
        print("=" * 66)
        print("  [A] INTERACTIVE & AUDIO I/O:")
        print(f"      • Audio Recording Window      :  {timings['recording_time_sec']:>6.2f} s")
        print(f"      • Language Confirmation       :  {timings['target_lang_selection_sec']:>6.2f} s [{disp_mode}]")
        print(f"      • Output Speech Playback      :  {timings['playback_time_sec']:>6.2f} s")
        print("      " + "-" * 50)
        print(f"      Subtotal (User & Audio I/O)   :  {io_subtotal_sec:>6.2f} s")
        print()
        print("  [B] PURE MACHINE PROCESSING LATENCY:")
        print(f"      • ASR [LID + Transcription]   :  {timings['asr_time_sec']:>6.2f} s")
        print(f"        ├── LID (Whisper)           :  {timings.get('lid_time_sec', 0.0):>6.2f} s")
        print(f"        └── Decode ({routed_model:<14}):  {timings.get('transcription_time_sec', 0.0):>6.2f} s")
        print(f"      • NMT [NLLB-200 INT8]         :  {timings['translation_time_sec']:>6.2f} s")
        print(f"      • TTS [Synthesis Compute]     :  {timings['tts_synthesis_time_sec']:>6.2f} s")
        print("      " + "-" * 50)
        print(f"      Subtotal (Pure Compute)       :  {pure_machine_time_sec:>6.2f} s")
        print("  " + "-" * 62)
        print(f"  ⚡ PURE INFERENCE TURNAROUND     :  {pure_machine_time_sec:>6.2f} s")
        print(f"  🚀 WALL-CLOCK TOTAL ELAPSED      :  {total_runtime:>6.2f} s")
        if cold_init_time_sec > 0:
            print(f"  ❄️ Cold Model Boot Time          :  {cold_init_time_sec:>6.2f} s (Excluded from inference)")
        print("  " + "-" * 62)
        print("  MEMORY & SYSTEM FOOTPRINT (PID ISOLATED):")
        print(f"  💾 Active Working RAM (RSS)      :  {current_rss_mb:>6.1f} MB (Physical memory currently in use)")
        print(f"  📊 Average Memory Duty           :  {avg_rss_mb:>6.1f} MB (Mean footprint across checkpoints)")
        print(f"  📈 Peak Lifetime Watermark       :  {peak_ram_mb:>6.1f} MB (OS kernel high-water mark)")
        print(f"  🖥️ Active Compute Utilization    :  {active_compute_cores:>6.2f} cores ({normalized_compute_percent:.1f}%)")
        print(f"  ⚙️ CPU Thread Configuration      :  {cores_count} Worker Threads")
        print("=" * 66 + "\n")

    if on_stage_update:
        on_stage_update("done")

    _save_evaluation_log(output_log, evaluation_entry)
    return evaluation_entry


# ─────────────────────────────────────────────
# 🚀 Single-Shot Evaluation Entrypoint
# ─────────────────────────────────────────────
def run_voice2voice_evaluation(
    duration: float = 2.5,
    output_log: str = "evaluation_log.json",
    input_audio_path: str = None,
    source_lang: str = None,
    target_lang: str = None,
    pair_lang: str = None,
    use_vad: bool = False,
    is_warm: bool = False,
    verbose: bool = False,
    is_research_mode: bool = False
) -> dict:
    """
    Journal-grade execution + evaluation harness for single runs.
    Enforces sequential Stage-and-Flush memory lifecycle management
    to prevent concurrent multi-model RAM accumulation on memory-constrained systems.
    """
    validate_language_arguments(
        source_lang=source_lang,
        target_lang=target_lang,
        pair_lang=pair_lang
    )

    pair_langs = None
    if pair_lang:
        if isinstance(pair_lang, str):
            p1, p2 = validate_pair_syntax(pair_lang)
            pair_langs = (p1, p2)
        elif isinstance(pair_lang, (list, tuple)) and len(pair_lang) == 2:
            p1, p2 = validate_pair_syntax(f"{pair_lang[0]}:{pair_lang[1]}")
            pair_langs = (p1, p2)

    cold_load_sec = 0.0
    return process_single_turn(
        turn_index=1,
        duration=duration,
        output_log=output_log,
        input_audio_path=input_audio_path,
        source_lang=source_lang,
        target_lang=target_lang,
        pair_langs=pair_langs,
        use_vad=use_vad,
        is_warm=is_warm,
        cold_init_time_sec=cold_load_sec,
        stage_and_flush=True,
        verbose=verbose,
        is_research_mode=is_research_mode
    )


# ─────────────────────────────────────────────
# 💾 Atomic Log Persistence
# ─────────────────────────────────────────────
def _save_evaluation_log(output_log: str, evaluation_entry: dict):
    """
    Atomic read-modify-write log persistence with timestamped corruption backup.
    Guarantees crash resilience using POSIX atomic os.replace.
    """
    try:
        existing = []
        if os.path.exists(output_log):
            if os.path.getsize(output_log) > 0:
                with open(output_log, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            existing = data
                        elif isinstance(data, dict):
                            existing = [data]
                    except json.JSONDecodeError as err:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{output_log}.corrupted_{ts}.bak"
                        shutil.copyfile(output_log, backup_name)
                        print(f"⚠️ Corrupted JSON detected ({err}). Backed up to {backup_name}")
                        existing = []

        existing.append(evaluation_entry)

        # Atomic write pattern: write to tmp file then atomic replace
        output_dir = os.path.dirname(os.path.abspath(output_log))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = f"{output_log}.{os.getpid()}_{int(time.time() * 1000)}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, output_log)

    except Exception as e:
        print(f"⚠️ Failed to update log file: {e}")


# ─────────────────────────────────────────────
# 💻 CLI Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice2Voice Offline Speech-to-Speech Translation Pipeline"
    )
    parser.add_argument("audio_pos", nargs="?", default=None, help="Optional positional audio file path")
    parser.add_argument("target_pos", nargs="?", default=None, help="Optional positional target language code")
    parser.add_argument("-a", "--audio", default=None, help="Input WAV audio file path")
    parser.add_argument("-S", "--source", "--src", default=None, help="Locked source language code (e.g., te, hi, en)")
    parser.add_argument("-t", "--target", default=None, help="Target language code (e.g., hi, te, en)")
    parser.add_argument("-p", "--pair", default=None, help="Bi-directional auto-dialogue pair (e.g., en:te, en:hi)")
    parser.add_argument("-d", "--duration", type=float, default=2.5, help="Recording duration in seconds (default: 2.5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display full diagnostic academic telemetry trees")
    parser.add_argument("--vad", action="store_true", help="Enable experimental streaming VAD (default: False)")
    parser.add_argument("--research", action="store_true", help="Enforce published 8-language FLEURS candidate universe and tau=0.70 routing")
    parser.add_argument("-o", "--output-log", default="evaluation_log.json", help="Path to evaluation log file")

    args = parser.parse_args()
    cli_audio = args.audio or args.audio_pos
    cli_target = args.target or args.target_pos
    cli_source = args.source

    try:
        validate_language_arguments(
            source_lang=cli_source,
            target_lang=cli_target,
            pair_lang=args.pair
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    entry = run_voice2voice_evaluation(
        duration=args.duration,
        output_log=args.output_log,
        input_audio_path=cli_audio,
        source_lang=cli_source,
        target_lang=cli_target,
        pair_lang=args.pair,
        use_vad=args.vad,
        verbose=args.verbose,
        is_research_mode=args.research
    )
    if isinstance(entry, dict) and (entry.get("status") == "error" or entry.get("failed_stage") is not None):
        sys.exit(1)
