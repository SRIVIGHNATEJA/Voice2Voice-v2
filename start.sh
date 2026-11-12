#!/bin/bash

# Voice2Voice Translation System - Startup Script
# Automates environment setup and pipeline execution

set -e  # Exit on error

# 🛡️ Force 100% strict offline execution across Hugging Face & Transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Ensure execution in the voice2voice directory regardless of where start.sh was called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"

echo "=========================================="
echo "Voice2Voice Translation System"
echo "=========================================="
echo ""

# Pre-flight system dependency checks
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ [ERROR] ffmpeg is not installed or not in PATH!"
    echo "👉 Required by Whisper for audio decoding."
    echo "   macOS:   brew install ffmpeg"
    echo "   Ubuntu:  sudo apt update && sudo apt install -y ffmpeg"
    exit 1
fi
echo "🎬 ffmpeg dependency verified"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/.dependencies_installed" ]; then
    echo "📥 Installing dependencies..."
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r requirements.txt
    touch venv/.dependencies_installed
    echo "✅ Dependencies installed"
    echo ""

    # Run environment setup
    echo "⚙️  Running environment setup..."
    PYTHONPATH=. "$PYTHON_BIN" setup/setup_env.py
    echo ""
else
    echo "✅ Dependencies already installed"
    echo ""
fi

# Run the main pipeline (supports single-turn and auto-dialogue)
echo "=========================================="
echo "🚀 Starting Voice2Voice Pipeline"
echo "=========================================="
echo ""

if [ $# -eq 0 ]; then
    echo "💡 Tip: Pass -t <lang> to lock target language (e.g. -t te, -t hi, -t en)."
    echo "         Pass -p en:te for bi-directional auto-dialogue."
    echo "         Pass -h or --help to see all options."
    echo ""
    PYTHONPATH=. "$PYTHON_BIN" run_pipeline.py
else
    PYTHONPATH=. "$PYTHON_BIN" run_pipeline.py "$@"
fi

echo ""
echo "=========================================="
echo "✅ Pipeline execution completed"
echo "=========================================="
