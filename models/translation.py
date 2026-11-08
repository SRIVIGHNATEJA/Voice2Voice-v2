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
logging.getLogger("torch").setLevel(logging.ERROR)

import platform
import re
import time
import unicodedata
from transformers import pipeline, AutoTokenizer
import torch
from utils.system import device
from utils.language import NLLB_LANG_CODE_MAP

# ─────────────────────────────────────────────
# 📁 Checkpoint & Engine Configuration
# ─────────────────────────────────────────────
MODEL_ID = "facebook/nllb-200-distilled-600M"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANTIZED_CHECKPOINT = os.path.join(ROOT_DIR, "models", "checkpoints", "nllb_int8_600m.pt")

def _setup_quantization_engine():
    supported_engines = getattr(torch.backends.quantized, "supported_engines", [])
    is_arm = any(x in platform.machine().lower() for x in ["arm", "aarch64", "darwin"])
    if is_arm and "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"

# ─────────────────────────────────────────────
# 🔒 Global cached pipeline
# ─────────────────────────────────────────────
_nllb_pipeline = None


# ─────────────────────────────────────────────
# 🌐 NLLB Translation Pipeline (Fast / Lazy)
# ─────────────────────────────────────────────
def get_nllb_pipeline():
    global _nllb_pipeline
    if _nllb_pipeline is None:
        _setup_quantization_engine()

        if os.path.exists(QUANTIZED_CHECKPOINT) and device == "cpu":
            t0 = time.time()
            try:
                # ⚡ Fast path: Direct load of offline pre-quantized INT8 checkpoint into CPU RAM
                try:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                loaded_model = torch.load(
                    QUANTIZED_CHECKPOINT,
                    map_location="cpu",
                    weights_only=False
                )
                loaded_model.eval()
                _nllb_pipeline = pipeline(
                    "translation",
                    model=loaded_model,
                    tokenizer=tokenizer,
                    device=-1
                )
                t_load = time.time() - t0
                # Fast load complete
                return _nllb_pipeline
            except Exception as e:
                print(f"⚠️ Failed to load serialized checkpoint ({e}). Falling back to standard loader.")

        # Standard loader fallback
        print("🔁 Loading NLLB-200 translation model...")
        _nllb_pipeline = pipeline(
            "translation",
            model=MODEL_ID,
            device=0 if device == "cuda" else -1
        )
        
        if device == "cpu":
            print("⚡ Applying dynamic quantization to NLLB (Int8)...")
            try:
                _nllb_pipeline.model = torch.quantization.quantize_dynamic(
                    _nllb_pipeline.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"⚠️ Quantization failed: {e}. Proceeding with float32.")

    return _nllb_pipeline


def unload_nllb():
    """
    Explicitly dereferences NLLB pipeline and triggers garbage collection to reclaim memory.
    """
    global _nllb_pipeline
    if _nllb_pipeline is not None:
        del _nllb_pipeline
        _nllb_pipeline = None
        import gc
        gc.collect()


# ─────────────────────────────────────────────
# 🛡️ Safe Multi-Script Clause Splitting Regex
# Preserves floating point decimals (e.g., 3.5) while splitting on:
# 1. Any punctuation followed by whitespace ([.?!।॥]\s+)
# 2. Never-decimal terminators without whitespace ([?!।॥]\S)
# 3. Periods followed directly by uppercase / Indic characters (e.g. "Hello.How")
# ─────────────────────────────────────────────
CLAUSE_SPLIT_REGEX = r'(?<=[.?!।॥])\s+|(?<=[?!।॥])(?=\S)|(?<=\.)(?=[A-Z\u0900-\u0D7F])'


# ─────────────────────────────────────────────
# 🔁 Core Translation Function
# ─────────────────────────────────────────────
def translate_with_nllb(
    text,
    src_lang_code,
    tgt_lang_code,
    num_beams: int = 5,
    max_new_tokens: int = 256,
    stream: bool = False,
    stream_prefix: str = "",
    verbose: bool = False
):
    """
    Translates text using NLLB-200 following Nature Scientific Reports (2026) Table 4:
    Beam size = 5, max_new_tokens = 256, batch_size = 1.
    For single-clause utterances, performs direct full-context generation.
    For compound multi-sentence utterances, splits across clause boundaries and translates
    in batches to prevent token overflow, trading off long-range discourse context for
    generation stability on CPU hardware.
    """
    if not text or not text.strip():
        print("⚠️ Empty input text. Skipping translation.")
        return None

    try:
        src_nllb = NLLB_LANG_CODE_MAP.get(src_lang_code, src_lang_code or "eng_Latn")
        tgt_nllb = NLLB_LANG_CODE_MAP.get(tgt_lang_code, tgt_lang_code or "eng_Latn")

        # ⚡ Instant bypass if source and target resolve to the same language
        if src_nllb and tgt_nllb and src_nllb == tgt_nllb:
            clean_same_text = unicodedata.normalize("NFC", text.strip())
            if stream and stream_prefix:
                from utils.ui import typewrite_text
                typewrite_text(clean_same_text, delay=0.012, prefix=stream_prefix, suffix='"')
            else:
                if verbose:
                    print(f"⚡ Source and Target are identical ({src_nllb}). Skipping translation.")
            return clean_same_text

        nllb = get_nllb_pipeline()

        # Only pass streamer directly to model.generate if num_beams == 1 (Transformers requirement)
        streamer = None
        if stream and num_beams == 1:
            try:
                from utils.ui import LiveTokenStreamer
                streamer = LiveTokenStreamer(nllb.tokenizer, prefix=stream_prefix, suffix='"')
            except Exception:
                streamer = None

        # Split compound or multi-sentence utterances into clauses to prevent NLLB single-sentence premature EOS cutoff
        clauses = [c.strip() for c in re.split(CLAUSE_SPLIT_REGEX, text.strip()) if c.strip()]
        if not clauses:
            clauses = [text.strip()]

        if len(clauses) == 1:
            # 🎯 Nature Scientific Reports (2026) Standard: Full-sentence translation with 256 token budget
            translated = nllb(
                clauses[0],
                src_lang=src_nllb,
                tgt_lang=tgt_nllb,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                streamer=streamer if num_beams == 1 else None
            )
            raw_result = translated[0]["translation_text"].strip()
        else:
            if verbose:
                print(f"🧩 Multi-sentence utterance detected ({len(clauses)} clauses). Translating clause-by-clause...")
            translated = nllb(
                clauses,
                src_lang=src_nllb,
                tgt_lang=tgt_nllb,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )
            raw_result = " ".join(t["translation_text"].strip() for t in translated if t.get("translation_text"))

        final_text = unicodedata.normalize("NFC", raw_result)

        # If streaming was requested with num_beams > 1, stream the final text cleanly
        if stream and num_beams > 1:
            from utils.ui import typewrite_text
            typewrite_text(final_text, delay=0.012, prefix=stream_prefix, suffix='"')

        return final_text

    except Exception as e:
        print(f"[ERROR] NLLB Translation failed: {e}")
        return None
