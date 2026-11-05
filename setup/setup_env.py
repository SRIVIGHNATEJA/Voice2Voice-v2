"""
Environment verification & pre-flight health check utility.

Validates:
1. System ffmpeg binary (required by Whisper for audio loading)
2. Python runtime version (>= 3.10)
3. Audio hardware / PortAudio input device availability
4. Hugging Face local model cache status

Safe to run multiple times.
"""

import os
import sys
import shutil
import platform


def check_ffmpeg():
    print("🎬 Checking ffmpeg system binary...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  ✅ ffmpeg found at: {ffmpeg_path}")
        return True

    print("\n❌ [ERROR] ffmpeg is not installed or not in PATH!")
    print("👉 Whisper requires ffmpeg for audio loading and resynthesis.")
    print("   macOS:   brew install ffmpeg")
    print("   Ubuntu:  sudo apt update && sudo apt install -y ffmpeg")
    print("   Arch:    sudo pacman -S ffmpeg")
    print("   Windows: choco install ffmpeg (or winget install Gyan.FFmpeg)\n")
    return False


def check_python_version():
    print("🐍 Checking Python version...")
    v = sys.version_info
    print(f"  ℹ️ Detected Python: {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor >= 10:
        print("  ✅ Python version is compatible (>= 3.10)")
        return True
    else:
        print("  ⚠️ Warning: Recommended Python version is 3.10 - 3.12")
        return False


def check_audio_devices():
    print("🎙️ Checking audio capture hardware...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if input_devices:
            default_in = sd.query_devices(kind="input")
            print(f"  ✅ Default microphone found: '{default_in.get('name')}'")
            return True
        else:
            print("  ⚠️ Warning: No audio input devices detected.")
            return False
    except Exception as e:
        print(f"  ⚠️ Warning: Could not query sounddevice ({e}).")
        return False


def scan_huggingface_cache():
    print("🔍 Scanning Hugging Face cache...")
    if shutil.which("huggingface-cli") is None:
        print("  ℹ️ huggingface-cli not in PATH. Skipping cache scan.")
        return

    os.system("huggingface-cli scan-cache")


def run_preflight_checks():
    print("==========================================")
    print("🚀 Voice2Voice System Pre-Flight Check")
    print("==========================================\n")

    py_ok = check_python_version()
    ffmpeg_ok = check_ffmpeg()
    audio_ok = check_audio_devices()
    print("")
    scan_huggingface_cache()

    print("\n==========================================")
    if not ffmpeg_ok:
        print("❌ Pre-flight checks FAILED: Missing required ffmpeg.")
        sys.exit(1)
    else:
        print("✅ Pre-flight checks PASSED! System ready.")
    print("==========================================\n")


if __name__ == "__main__":
    run_preflight_checks()
