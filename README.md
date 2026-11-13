# Voice2Voice-v2: Offline Multilingual Speech-to-Speech Translation with Adaptive ASR Routing and Cross-Lingual TTS

[![Paper](https://img.shields.io/badge/Nature%20Scientific%20Reports-10.1038%2Fs41598--026--67788--0-blue)](https://doi.org/10.1038/s41598-026-67788-0)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20Linux%20%7C%20Windows-lightgrey)](#system-prerequisites)

**Voice2Voice-v2** is the edge-hardened, reproducible reference implementation of the unified offline speech-to-speech translation framework published in *Nature Scientific Reports* (2026, Vol. 16, Article 36511).

Designed specifically for cross-lingual communication between Indian regional languages and global languages on resource-constrained consumer hardware (such as Apple Silicon M-series with 8 GB unified memory), the architecture achieves real-time conversational usability through adaptive model routing, mathematical token bounding, and stage-by-stage memory management.

---

## Key Systems Architecture & Edge Optimizations

```
[ Input Audio ] (Live Mic Capture or 16 kHz Mono WAV)
       │
       ▼
[ Stage 1: Adaptive Dual-ASR Routing (tau = 0.70) ]
   • Constrained Softmax Language Identification (LID)
   • Telugu / Hindi  ──>  ai4bharat/indic-conformer-600m-multilingual (ONNX, 4 P-cores)
   • English / Intl  ──>  openai/whisper-small (FP32, Mel reuse)
   • Cuts Telugu Word Error Rate (WER) from 107.42% to 19.79%
       │
       ▼
[ Stage 2: Neural Machine Translation (NMT) ]
   • facebook/nllb-200-distilled-600M with QNNPACK INT8 Quantization
   • Dynamic clause-boundary splitting & Unicode NFC normalization
   • Beam search (num_beams=5, max_new_tokens bounded)
       │
       ▼
[ Stage 3: Dual-Route Speech Synthesis (TTS) ]
   • Indian Languages ──> ai4bharat/indic-parler-tts (Pruned Flan-T5, -1.36 GB RAM)
   • International    ──> Native Host OS streaming speech (pyttsx3)
   • 8 structural WAV validation guards (frame count, 1D mono, 44.1 kHz, non-zero flatline check)
       │
       ▼
[ Memory Lifecycle: Stage-and-Flush ]
   • Models unloaded sequentially between compute stages
   • Peak RAM stays strictly under 2.5 GB (plateaus at ~350-450 MB post-flush)
```

1. **Adaptive Dual-ASR Routing ($\tau = 0.70$):**
   Standard Whisper models suffer severe phonetic degradation on low-resource Indian languages (e.g. 107.42% WER on Telugu FLEURS). The pipeline deploys an adaptive confidence router ($E_{\text{routed}}$): when LID detects Indic script with confidence $\ge 0.70$, audio routes to `IndicConformer-600M` pinned to 4 Performance cores via ONNX Runtime; otherwise it falls back to `Whisper-Small`.
2. **Stage-and-Flush Memory Lifecycle:**
   Instead of retaining all multi-gigabyte models concurrently in RAM, each neural component is loaded, executed, and immediately dereferenced with explicit garbage collection. This maintains peak resident memory below **2.5 GB**, enabling full execution on 8 GB laptops without disk paging or out-of-memory crashes.
3. **Quantized Machine Translation:**
   Uses `facebook/nllb-200-distilled-600M` with dynamic INT8 quantization via PyTorch's QNNPACK engine on ARM architectures, reducing memory footprint while accelerating multi-clause translation.
4. **Flan-T5 Encoder Pruning for Indic TTS:**
   Precomputes fixed description embeddings (`parler_default_desc_tensor.pt`), completely bypassing the 1.5 GB Flan-T5 text encoder at runtime and saving **1.36 GB** of resident memory during voice synthesis.
5. **Structural Audio Integrity Verification:**
   All synthesized audio artifacts undergo rigorous post-generation validation (file existence, non-zero frame count, 1D mono topology, 44.1 kHz sample rate, numerical finiteness, and digital-zero flatline detection) before dispatching to audio drivers.

---

## Project Structure

```
Voice2Voice/
├── start.sh                          # Unified CLI entry-point & environment launcher
├── run_pipeline.py                   # Master pipeline orchestrator (Stages 0–6)
├── test_sprints.py                   # Automated 23-test unit & integration test suite
├── requirements.txt                  # Pinned core runtime dependencies
├── requirements-eval.txt             # Academic evaluation dependencies (WER/BLEU)
├── LICENSE                           # MIT License
├── README.md                         # Reference documentation
│
├── models/
│   ├── asr.py                        # Dual-ASR routing, Whisper LID & IndicConformer ONNX
│   ├── translation.py                # NLLB-200 INT8 quantization & clause splitting
│   ├── tts.py                        # Dual-path TTS (Indic-Parler-TTS & pyttsx3) + WAV guards
│   └── checkpoints/
│       └── .gitkeep                  # Checkpoints anchor (large .pt binaries gitignored)
│
├── utils/
│   ├── audio.py                      # Audio capture, RMS silence trimming & downmixing
│   ├── language.py                   # Bidirectional ISO / FLORES-200 language registries
│   ├── system.py                     # Hardware telemetry & compute thread allocation
│   └── ui.py                         # StudioCanvas terminal rendering & live token streamer
│
├── scripts/
│   └── export_nllb_int8.py           # Offline utility to serialize NLLB-200 INT8 weights
│
├── setup/
│   ├── setup_env.py                  # Environment verification & system diagnostic check
│   ├── warmup_models.py              # Model weight pre-loader & warmup accelerator
│   └── README.md                     # Setup instructions
│
└── notebooks/
    └── notebook_reference.ipynb      # Controlled research notebook implementation
```

---

## System Prerequisites

Ensure system-level binaries and audio libraries are present before setup:

* **FFmpeg** (Required by Whisper for audio decoding):
  * **macOS**: `brew install ffmpeg`
  * **Ubuntu/Debian**: `sudo apt update && sudo apt install -y ffmpeg`
  * **Windows**: `choco install ffmpeg` or `winget install Gyan.FFmpeg`
* **Python**: 3.10 to 3.12 (Tested on Python 3.12.4 on Apple Silicon M3)
* **Microphone Access**: On macOS, verify terminal microphone permission under:
  * `System Settings > Privacy & Security > Microphone`

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SRIVIGHNATEJA/Voice2Voice-v2.git
   cd Voice2Voice-v2
   ```

2. **Create and activate an isolated virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS / Linux
   # venv\Scripts\activate       # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For academic benchmark scoring (WER, CER, BLEU):
   ```bash
   pip install -r requirements-eval.txt
   ```

---

## Pre-Flight Check & Model Warmup

Verify system configuration, microphone permissions, and cache integrity:

```bash
python setup/setup_env.py
```

### Optional Offline Model Optimization:
To pre-cache local INT8 translation weights and precomputed description tensors for sub-second cold starts, run:
```bash
python setup/warmup_models.py
python scripts/export_nllb_int8.py
```
*(Note: If uncached, `models/tts.py` and `models/translation.py` automatically fall back to downloading and synthesizing directly via Hugging Face checkpoints without interruption).*

---

## Quick Start CLI

Execute turns interactively or through deterministic command-line flags:

```bash
# 1. Bi-Directional Auto-Dialogue Mode (Alternates English <-> Telugu turns seamlessly):
./start.sh -p en:te

# 2. Target-Locked Mode (Direct translation to Hindi):
./start.sh -t hi

# 3. Source-Locked Mode (Bypasses LID compute overhead; Telugu input -> English):
./start.sh -S te -t en

# 4. Pre-Recorded Audio File Translation:
./start.sh -a sample.wav -t en

# 5. Run the complete automated test suite (23 unit & integration tests):
python test_sprints.py
```

---

## Automated Verification Suite

The repository includes a comprehensive 23-test automated test suite in `test_sprints.py` covering:
- RMS silence trimming and acoustic threshold bounds.
- Dual-ASR routing, isolated telemetry, and out-of-pair acoustic guards.
- Bi-directional dialogue pair switching and $\tau = 0.70$ routing.
- INT8 dynamic token bounding and clause preservation.
- Parler-TTS prompt embedding cache and dynamic DAC framing bounds (`[170, 420]`).
- Host OS vs. Neural TTS delivery contracts and audio artifact validation.
- Subprocess exit-code propagation on pipeline failures.

Run the test suite:
```bash
python test_sprints.py
```
*(The suite self-provisions temporary test fixtures on clean clones and requires zero external model downloads).*

---

## Licensing & Third-Party Model Terms

### 1. Repository Source Code
The core source code, orchestration logic, and utility scripts in this repository are licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```text
Copyright (c) 2025-2026 DASARI SRI VIGHNA TEJA
```

### 2. Pre-trained Model Weights & Upstream Attribution
This repository provides orchestration, optimization, and inference interfaces for pretrained neural models. Model weights and upstream architectures are governed by their respective licenses:
* **Meta NLLB-200 (`facebook/nllb-200-distilled-600M`):** Model weights are released by Meta under the Creative Commons Attribution-NonCommercial 4.0 International license (**CC-BY-NC 4.0**), which restricts commercial usage of the weights.
* **AI4Bharat IndicConformer & Indic Parler-TTS:** Governed by the **Apache License 2.0**. Model weights hosted on Hugging Face are subject to gated-access terms and AI4Bharat research conditions.
* **OpenAI Whisper (`whisper-small`):** Released under the **MIT License**.
* **System Speech (`pyttsx3`):** Delegates to the host operating system's native speech synthesizer (Apple `NSSpeechSynthesizer`, Microsoft SAPI5, or Linux eSpeak).

---

## Citation

If you use this codebase, methodology, or benchmarks in your research, please cite the published paper:

```bibtex
@article{dasari2026unified,
  title={A unified neural framework for offline multilingual speech-to-speech translation with adaptive ASR routing and cross-lingual TTS},
  author={Dasari, Sri Vighna Teja},
  journal={Scientific Reports},
  volume={16},
  number={1},
  pages={36511},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-026-67788-0}
}
```
