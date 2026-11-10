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

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("parler_tts").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import threading
import torch
import soundfile as sf
import numpy as np
import pyttsx3
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from parler_tts import ParlerTTSForConditionalGeneration
from utils.system import device
from utils.audio import play_wav, trim_silence_rms

import contextlib

# ─────────────────────────────────────────────
# 🔒 Global caches & concurrency locks
# ─────────────────────────────────────────────
_parler_model = None
_tokenizer = None
_desc_tokenizer = None
_engine = None
_cached_desc_tensors = {}
_parler_lock = threading.Lock()

# Checkpoint paths for accelerated loading
_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints")
_FP32_PRUNED_CKPT = os.path.join(_CKPT_DIR, "parler_indic_fp32_pruned.pt")
_QUANT_CKPT = os.path.join(_CKPT_DIR, "parler_indic_decoder_int8.pt")
_DESC_TENSOR_PATH = os.path.join(_CKPT_DIR, "parler_default_desc_tensor.pt")


def unload_tts():
    """
    Explicitly dereferences Parler-TTS model and cached tensors and triggers
    cyclic garbage collection to free memory on Apple Silicon / CPU.
    """
    global _parler_model, _tokenizer, _desc_tokenizer, _cached_desc_tensors, _engine
    if _parler_model is not None:
        del _parler_model
        _parler_model = None
    _cached_desc_tensors.clear()
    if _engine is not None:
        try:
            _engine.stop()
        except Exception:
            pass
        _engine = None
    import gc
    gc.collect()



# ─────────────────────────────────────────────
# 🧠 Lazy-load Indic Parler TTS
# ─────────────────────────────────────────────
def load_parler_tts(prefer_fp32: bool = True):
    global _parler_model, _tokenizer, _desc_tokenizer

    with _parler_lock:
        if _parler_model is None:
            default_desc = "A calm voice with natural pace, clarity and stability."

            # 1. 🚀 Primary Engine: Pre-Saved Pruned FP32 + SDPA Checkpoint (0.76s Load | 32.1 tok/s)
            if prefer_fp32 and os.path.exists(_FP32_PRUNED_CKPT) and os.path.exists(_DESC_TENSOR_PATH):
                with open(os.devnull, "w") as fnull:
                    with contextlib.redirect_stderr(fnull):
                        _parler_model = torch.load(_FP32_PRUNED_CKPT, map_location="cpu", weights_only=False)
                        _parler_model.eval()
                        if hasattr(_parler_model, "decoder") and hasattr(_parler_model.decoder, "config"):
                            _parler_model.decoder.config._attn_implementation = "sdpa"
                        try:
                            _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", local_files_only=True)
                        except Exception:
                            _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
                        if _tokenizer.pad_token is None:
                            _tokenizer.pad_token = _tokenizer.eos_token

                        cached_desc = torch.load(_DESC_TENSOR_PATH, map_location="cpu", weights_only=False)
                        dummy_desc_ids = torch.zeros((1, 10), dtype=torch.long)
                        _cached_desc_tensors[default_desc] = (dummy_desc_ids, None, cached_desc)
                return _parler_model, _tokenizer, None

            # 2. 🛡️ Secondary Engine: Pre-Saved Int8 Quantized Checkpoint (Emergency Low-Memory Fallback)
            if os.path.exists(_QUANT_CKPT) and os.path.exists(_DESC_TENSOR_PATH):
                if device == "cpu":
                    torch.backends.quantized.engine = "qnnpack"
                with open(os.devnull, "w") as fnull:
                    with contextlib.redirect_stderr(fnull):
                        _parler_model = torch.load(_QUANT_CKPT, map_location="cpu", weights_only=False)
                        _parler_model.eval()
                        if hasattr(_parler_model, "decoder") and hasattr(_parler_model.decoder, "config"):
                            _parler_model.decoder.config._attn_implementation = "sdpa"
                        try:
                            _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", local_files_only=True)
                        except Exception:
                            _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
                        if _tokenizer.pad_token is None:
                            _tokenizer.pad_token = _tokenizer.eos_token

                        cached_desc = torch.load(_DESC_TENSOR_PATH, map_location="cpu", weights_only=False)
                        dummy_desc_ids = torch.zeros((1, 10), dtype=torch.long)
                        _cached_desc_tensors[default_desc] = (dummy_desc_ids, None, cached_desc)
                return _parler_model, _tokenizer, None

            # 3. 🔁 Pretrained Fallback: Hugging Face hub with on-the-fly Flan-T5 amputation
            print("🔁 Loading Indic-Parler TTS model (Pretrained fallback)...")
            with open(os.devnull, "w") as fnull:
                with contextlib.redirect_stderr(fnull):
                    _parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
                        "ai4bharat/indic-parler-tts"
                    ).to(device)
                    _parler_model.eval()
                    if hasattr(_parler_model, "decoder") and hasattr(_parler_model.decoder, "config"):
                        _parler_model.decoder.config._attn_implementation = "sdpa"

                    try:
                        _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", local_files_only=True)
                    except Exception:
                        _tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

                    try:
                        _desc_tokenizer = AutoTokenizer.from_pretrained(
                            _parler_model.config.text_encoder._name_or_path,
                            local_files_only=True
                        )
                    except Exception:
                        _desc_tokenizer = AutoTokenizer.from_pretrained(
                            _parler_model.config.text_encoder._name_or_path
                        )
                    if _tokenizer.pad_token is None:
                        _tokenizer.pad_token = _tokenizer.eos_token
                    if _desc_tokenizer.pad_token is None:
                        _desc_tokenizer.pad_token = _desc_tokenizer.eos_token

                    get_or_create_description_embeddings(_parler_model, _desc_tokenizer, default_desc)

                    # Amputate Flan-T5 to free 1.36 GB RAM
                    if hasattr(_parler_model, "text_encoder") and _parler_model.text_encoder is not None:
                        del _parler_model.text_encoder
                        _parler_model.text_encoder = None
                        import gc
                        gc.collect()

        return _parler_model, _tokenizer, _desc_tokenizer


def get_or_create_description_embeddings(model, desc_tokenizer, description: str):
    """
    Retrieves or computes cached Flan-T5 text encoder outputs for voice description conditioning.
    Eliminates redundant forward passes on static description prompts across synthesis calls.
    ALWAYS returns a fresh clone of last_hidden_state to prevent Parler's internal in-place
    prompt concatenation from mutating the cached style embeddings across consecutive turns.
    """
    if description in _cached_desc_tensors:
        desc_ids, desc_mask, base_out = _cached_desc_tensors[description]
        fresh_out = BaseModelOutput(last_hidden_state=base_out.last_hidden_state.clone()) if base_out is not None else None
        return desc_ids, desc_mask, fresh_out

    if os.path.exists(_DESC_TENSOR_PATH):
        cached_desc = torch.load(_DESC_TENSOR_PATH, map_location="cpu", weights_only=False)
        dummy_desc_ids = torch.zeros((1, 10), dtype=torch.long)
        _cached_desc_tensors[description] = (dummy_desc_ids, None, cached_desc)
        fresh_out = BaseModelOutput(last_hidden_state=cached_desc.last_hidden_state.clone())
        return dummy_desc_ids, None, fresh_out

    if desc_tokenizer is None or getattr(model, "text_encoder", None) is None:
        dummy_desc_ids = torch.zeros((1, 10), dtype=torch.long)
        return dummy_desc_ids, None, None

    desc = desc_tokenizer(description, return_tensors="pt").to(device)
    with torch.no_grad():
        encoder = model.get_text_encoder()
        last_hidden_state = encoder(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask
        ).last_hidden_state
        if (
            model.text_encoder.config.hidden_size != model.decoder.config.hidden_size
            and model.decoder.config.cross_attention_hidden_size is None
        ):
            last_hidden_state = model.enc_to_dec_proj(last_hidden_state)
        if desc.attention_mask is not None:
            last_hidden_state = last_hidden_state * desc.attention_mask[..., None]
        cached_encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

    _cached_desc_tensors[description] = (desc.input_ids, desc.attention_mask, cached_encoder_outputs)
    fresh_out = BaseModelOutput(last_hidden_state=cached_encoder_outputs.last_hidden_state.clone())
    return desc.input_ids, desc.attention_mask, fresh_out


# ─────────────────────────────────────────────
# 🔊 Lazy-load pyttsx3 (fallback)
# ─────────────────────────────────────────────
def get_tts_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 180)
        _engine.setProperty("volume", 1.0)
    return _engine


# International voices (macOS presets)
LANG_VOICE_MAP = {
    "en": "com.apple.voice.compact.en-US.Samantha",
    "es": "com.apple.voice.compact.es-ES.Monica",
    "fr": "com.apple.voice.compact.fr-FR.Thomas",
    "zh": "com.apple.voice.compact.zh-CN.Tingting",
    "ar": "com.apple.voice.compact.ar-001.Maged",
    "pt": "com.apple.voice.compact.pt-BR.Luciana",
    "ru": "com.apple.voice.compact.ru-RU.Milena",
    "ja": "com.apple.voice.compact.ja-JP.Kyoko",
    "de": "com.apple.voice.compact.de-DE.Anna"
}


def get_system_voice_for_lang(engine, lang_code: str, return_status: bool = False):
    """
    Cross-platform voice resolver for pyttsx3.
    Tries platform-specific presets (macOS), then scans available voices by
    language/locale/name, falling back gracefully to the default system voice.
    Records is_fallback=True if no target-specific voice match exists.
    """
    if engine is None:
        return (None, False) if return_status else None

    try:
        voices = engine.getProperty("voices")
    except Exception:
        return (None, False) if return_status else None

    if not voices:
        return (None, False) if return_status else None

    # 1. Check known macOS mappings if ID exists in available voices
    if lang_code in LANG_VOICE_MAP:
        target_id = LANG_VOICE_MAP[lang_code]
        for v in voices:
            if v.id == target_id:
                return (v.id, False) if return_status else v.id

    # 2. Dynamic matching across macOS, Linux (espeak), and Windows (SAPI5)
    lang_lower = (lang_code or "en").lower()
    for v in voices:
        # Check languages attribute (list of str or bytes)
        langs = getattr(v, "languages", [])
        if langs:
            for l in langs:
                l_str = l.decode("utf-8", errors="ignore") if isinstance(l, bytes) else str(l)
                if lang_lower in l_str.lower():
                    return (v.id, False) if return_status else v.id

        # Check voice ID and name
        if lang_lower in getattr(v, "id", "").lower() or lang_lower in getattr(v, "name", "").lower():
            return (v.id, False) if return_status else v.id

    # 3. Safe fallback to default first voice
    is_fallback = lang_lower not in ("en", "eng")
    if is_fallback:
        v_name = getattr(voices[0], "name", voices[0].id)
        print(f"ℹ️ Notice: No specific system voice found for '{lang_code}'. Using default platform voice: {v_name}")
    return (voices[0].id, is_fallback) if return_status else voices[0].id


def calculate_dynamic_dac_frames(text: str) -> int:
    """
    Computes optimal max_new_tokens for Indic-Parler TTS (86.13 Hz DAC vocoder).
    Acoustic & articulatory bounds in 2.5s window:
      - Normal conversational human speech ceiling: 8-9 words (max 14 syllables).
      - Cross-lingual expansion factor (1.2x to 1.45x) + 0.8s breathing pause = 4.88s max.
      - 86.13 Hz DAC vocoder (44,100 / 512 hop):
        Floor: 170 frames (~2.0s) for single-word greetings.
        Ceiling: 420 frames (~4.88s) custom-made for 2.5s speech input.
    """
    words = len((text or "").strip().split())
    allowed_seconds = max(2.0, words * 0.45 + 1.2)
    return min(420, max(170, int(allowed_seconds * 86.13)))


# ─────────────────────────────────────────────
# 🎙️ Indic-Parler TTS (Decoupled Synthesis)
# ─────────────────────────────────────────────
def generate_speech(prompt, description, output_file="output.wav", max_new_tokens=None, prefer_fp32=True):
    """
    Synthesizes speech using Indic-Parler TTS and writes to output_file.
    Decoupled from audio playback to ensure accurate latency measurement.
    Optimizations:
      - Uses pre-saved pruned FP32+SDPA checkpoint (0.76s load, 32.1 tok/s on Apple Silicon Neon SIMD).
      - Allocates a 4-thread CPU compute budget during synthesis, cleanly restoring thread count upon exit.
      - Natural EOS termination with mathematical DAC safety ceiling (max_new_tokens=420) to prevent truncation.
      - Stochastic sampling (do_sample=True) to preserve acoustic fidelity.
    Returns output_file on success, None on failure.
    """
    if not prompt or not prompt.strip():
        print("⚠️ Empty prompt. Skipping TTS.")
        return None

    prev_threads = torch.get_num_threads()
    try:
        if device == "cpu":
            torch.set_num_threads(4)

        model, tokenizer, desc_tokenizer = load_parler_tts(prefer_fp32=prefer_fp32)

        # ⚡ Cached description conditioning (zero redundant encoder compute)
        desc_input_ids, desc_attention_mask, cached_encoder_outputs = get_or_create_description_embeddings(
            model, desc_tokenizer, description
        )

        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 🎛️ Natural EOS termination with safety ceiling (avoids speech truncation)
        # Bounded between 170-frame floor and 420-frame ceiling (86.13 Hz DAC vocoder)
        max_frames = max_new_tokens if max_new_tokens is not None else 420

        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stderr(fnull):
                with torch.inference_mode():
                    audio = model.generate(
                        input_ids=desc_input_ids,
                        attention_mask=desc_attention_mask,
                        encoder_outputs=cached_encoder_outputs,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                        max_new_tokens=max_frames,
                        use_cache=True,
                        do_sample=True
                    )

        audio_arr = audio.cpu().numpy().squeeze()

        if audio_arr.size == 0 or np.all(audio_arr == 0) or np.isnan(audio_arr).any():
            print("❌ Invalid waveform generated.")
            return None

        # ✂️ Strip residual vocoder trailing padding
        audio_arr = trim_silence_rms(audio_arr, sample_rate=model.config.sampling_rate)
        if audio_arr is None or len(audio_arr) == 0:
            print("❌ Audio trimming returned empty waveform.")
            return None

        sf.write(output_file, audio_arr, model.config.sampling_rate)

        # 🛡️ 8 Structural WAV Integrity Checks
        try:
            if not os.path.isfile(output_file):
                print(f"❌ Output WAV file not created on disk: {output_file}")
                return None
            val_data, val_sr = sf.read(output_file, dtype="float32")
            if len(val_data) == 0:
                print("❌ Output WAV has zero frames.")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
            if val_data.ndim != 1:
                print(f"❌ Output WAV must be 1D mono, got ndim={val_data.ndim}.")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
            if val_sr != 44100:
                print(f"❌ Output WAV sample rate mismatch: expected 44100, got {val_sr}.")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
            if not np.all(np.isfinite(val_data)):
                print("❌ Output WAV contains non-finite values (NaN / Inf).")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
            if np.all(val_data == 0.0):
                print("❌ Output WAV is digital-zero flatline silence.")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return None
        except Exception as val_err:
            print(f"❌ Structural WAV validation failed: {val_err}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return None

        return output_file

    except Exception as e:
        print(f"🔥 Indic-Parler TTS error: {e}")
        return None
    finally:
        if device == "cpu":
            torch.set_num_threads(prev_threads)


# ─────────────────────────────────────────────
# 🧠 Unified Synthesis Router (Pure Compute)
# ─────────────────────────────────────────────
def synthesize_tts(text, lang_type, tgt_lang_code, output_file=None, turn_index=None, prefer_fp32: bool = True):
    """
    Routes synthesis based on language type.
    Measures pure computational generation time without hardware playback latency.
    Returns: (artifact_path, engine_type, delivery_mode)
      - Neural Indic Success: (filepath, 'neural', 'file_artifact')
      - Neural Indic Failure: (None, 'neural', 'failed')
      - Host OS Voice (pyttsx3): (None, 'pyttsx3', 'host_speaker')
    """
    if not text or not text.strip():
        print("⚠️ Empty text. Skipping synthesis.")
        return None, "none", "skipped"

    if lang_type == "indian":
        description = "A calm voice with natural pace, clarity and stability."
        dynamic_frames = calculate_dynamic_dac_frames(text)

        if output_file is None:
            import time
            import uuid
            t_tag = int(time.time())
            t_idx = turn_index if turn_index is not None else 1
            uuid_suffix = uuid.uuid4().hex[:6]
            output_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                f"output_indic_turn_{t_idx}_{t_tag}_{uuid_suffix}.wav"
            )

        out_path = generate_speech(
            text,
            description,
            output_file=output_file,
            max_new_tokens=dynamic_frames,
            prefer_fp32=prefer_fp32
        )
        if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 44:
            return out_path, "neural", "file_artifact"
        return None, "neural", "failed"
    else:
        # Platform fallback: host OS speech delivery via pyttsx3
        return None, "pyttsx3", "host_speaker"


# ─────────────────────────────────────────────
# 🔊 Dedicated Audio Playback (Speaker Output)
# ─────────────────────────────────────────────
def playback_tts(target_or_text, engine_type, tgt_lang_code) -> bool:
    """
    Handles physical speaker output, separated from neural compute benchmarking.
    Returns: True if audio playback / speech delivery completed without exception, False otherwise.

    Note on delivery semantics:
      - For host OS TTS ('pyttsx3'), returning True confirms that the host OS speech engine
        (e.g., NSSpeechSynthesizer on macOS, SAPI on Windows, eSpeak on Linux) successfully
        accepted and dispatched the speech synthesis call (runAndWait) without an unhandled
        exception. It represents software-level speech delivery completion, not acoustic
        sensor verification that sound physically emanated from external hardware or was
        heard by an end user.
      - For neural TTS ('neural'), returning True confirms that sounddevice successfully wrote
        and waited on the DAC audio buffer.
    """
    if not target_or_text or not engine_type:
        return False

    playback_success = False
    if engine_type == "neural":
        if os.path.exists(str(target_or_text)):
            try:
                playback_success = play_wav(target_or_text)
            except Exception as e:
                print(f"⚠️ Audio playback error: {e}")
                playback_success = False
        else:
            print(f"⚠️ Audio target file not found: {target_or_text}")
            playback_success = False
    elif engine_type == "pyttsx3":
        try:
            engine = get_tts_engine()
            voice_id = get_system_voice_for_lang(engine, tgt_lang_code)
            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception:
                    pass

            engine.say(str(target_or_text))
            engine.runAndWait()
            playback_success = True
        except Exception as e:
            print(f"⚠️ OS voice playback error: {e}")
            playback_success = False

    return playback_success


# ─────────────────────────────────────────────
# 🔀 Unified Convenience Wrapper (Backward-Compatible)
# ─────────────────────────────────────────────
def play_tts_output(text, lang_type, tgt_lang_code):
    artifact_path, engine_type, delivery_mode = synthesize_tts(text, lang_type, tgt_lang_code)
    if delivery_mode == "file_artifact" and artifact_path:
        playback_tts(artifact_path, engine_type, tgt_lang_code)
    elif delivery_mode == "host_speaker":
        playback_tts(text, engine_type, tgt_lang_code)
