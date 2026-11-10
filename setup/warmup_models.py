"""
Optional model warm-up utility.

This script performs dummy inference passes to:
- eliminate first-run overhead
- stabilize runtime performance

It does NOT:
- affect evaluation metrics
- modify model logic
- alter pipeline execution

Warm-up should be run separately from benchmarking.
"""

import os
import sys

# Ensure root directory is in sys.path so utils and models can be imported from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper

from utils.system import device

# Silence tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 Warming up models (optional)…")

# ------------------------------------------------------------------
# Create dummy audio once
# ------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DUMMY_WAV = os.path.join(ROOT_DIR, "dummy.wav")
if not os.path.exists(DUMMY_WAV):
    sf.write(DUMMY_WAV, np.zeros(16000, dtype="float32"), 16000)

# ------------------------------------------------------------------
# 1️⃣ Whisper warm-up (LOCAL load)
# ------------------------------------------------------------------
try:
    whisper_model = whisper.load_model("small").to(device)
    whisper_model.eval()

    for i in range(2):
        t0 = time.time()
        audio = whisper.load_audio(DUMMY_WAV)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(device)
        _ = whisper.decode(whisper_model, mel, fp16=False)
        print(f"✅ Whisper warm-up {i+1}/2 in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"⚠️ Whisper warm-up failed: {e}")

# ------------------------------------------------------------------
# 2️⃣ IndicConformer warm-up (LOCAL load)
# ------------------------------------------------------------------
try:
    from transformers import AutoModel
    try:
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
    except Exception:
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        ).to(device)
    indic_model.eval()

    for i in range(2):
        t0 = time.time()
        at, sr = torchaudio.load(DUMMY_WAV)
        if sr != 16000:
            at = torchaudio.transforms.Resample(sr, 16000)(at)
        inp = at.mean(dim=0, keepdim=True).to(device)
        _ = indic_model(inp, "hi", "ctc")
        print(f"✅ IndicConformer warm-up {i+1}/2 in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"⚠️ IndicConformer warm-up failed: {e}")

# ------------------------------------------------------------------
# 3️⃣ NLLB warm-up (LOCAL load)
# ------------------------------------------------------------------
try:
    from transformers import pipeline
    nllb_pipeline = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        device=0 if device == "cuda" else -1
    )

    for i in range(2):
        t0 = time.time()
        _ = nllb_pipeline("Hello world", src_lang="eng_Latn", tgt_lang="fra_Latn")
        print(f"✅ NLLB warm-up {i+1}/2 in {time.time()-t0:.2f}s")
except Exception as e:
    print(f"⚠️ NLLB warm-up failed: {e}")

# ------------------------------------------------------------------
# 4️⃣ Indic-Parler TTS warm-up (LOCAL load)
# ------------------------------------------------------------------
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    desc_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if desc_tokenizer.pad_token is None:
        desc_tokenizer.pad_token = desc_tokenizer.eos_token

    for i in range(2):
        t0 = time.time()
        desc = desc_tokenizer(
            "Neutral Indian voice.",
            return_tensors="pt"
        ).to(device)
        prompt = tokenizer(
            "नमस्ते दुनिया",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            _ = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask
            )

        print(f"✅ Indic-Parler warm-up {i+1}/2 in {time.time()-t0:.2f}s")

except Exception as e:
    print(f"⚠️ Indic-Parler warm-up failed: {e}")

print("🎯 Warm-up completed. Safe to run pipeline.")
