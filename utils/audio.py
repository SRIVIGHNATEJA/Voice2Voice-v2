import os
import sys
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from utils.ui import countdown_alert, render_recording_progress


def record_audio(
    duration: float = 2.5,
    fs: int = 16000,
    filename: str = "user_input.wav",
    playback: bool = False,
    vad: bool = False,
    max_duration: float = 4.0,
    threshold_db: float = -35.0,
    countdown: int = 0,
    show_progress: bool = False,
    verbose: bool = False
):
    """
    Records raw audio at 16 kHz mono and saves directly to disk.
    Preserves pristine acoustic harmonics for authoritative ASR decoding without phase distortion.
    Default duration is 2.5s using deterministic sd.rec() for robust, zero-jitter capture.
    Includes optional countdown preparation alert and live progress bar visualization.
    Includes optional Two-State Energy VAD if vad=True:
      - State 1 (Pre-speech grace window): Waits up to 1.5s for speech onset (RMS > -35 dBFS).
      - State 2 (Active speech capture): Cuts off after 0.6s continuous trailing silence (max 4.0s).
      - Fallback: Gracefully records fixed duration if streaming fails.
    """
    if countdown > 0 and not verbose:
        countdown_alert(seconds=countdown)
        play_beep()

    if not vad:
        if verbose:
            print(f"🎤 Recording ({duration:.1f}s fixed)…")
        try:
            total_samples = int(duration * fs)
            data = sd.rec(total_samples, samplerate=fs, channels=1, dtype="float32")
            if show_progress and sys.stdout.isatty():
                t_start = time.perf_counter()
                while True:
                    elapsed = time.perf_counter() - t_start
                    if elapsed >= duration:
                        break
                    render_recording_progress(elapsed, duration)
                    time.sleep(0.04)
                render_recording_progress(duration, duration)
                out = sys.__stdout__ or sys.stdout
                out.write("\r\033[2K")
                out.flush()
            sd.wait()
            raw = data.flatten()
        except (sd.PortAudioError, Exception) as e:
            print(f"\n❌ Microphone recording failed: {e}")
            print("👉 On macOS, verify terminal microphone access in:")
            print("   System Settings > Privacy & Security > Microphone")
            raise e
    else:
        if verbose:
            print(f"🎤 Listening (Adaptive VAD, max {max_duration:.1f}s)…")
        chunk_ms = 50
        chunk_samples = int(fs * (chunk_ms / 1000.0))  # 800 samples at 16kHz
        max_pre_speech_chunks = int(1.5 / (chunk_ms / 1000.0))  # 30 chunks = 1.5s
        max_total_chunks = int(max_duration / (chunk_ms / 1000.0))
        silence_cutoff_chunks = int(0.6 / (chunk_ms / 1000.0))  # 12 chunks = 0.6s

        pre_roll_chunks = 3  # 150ms pre-roll to catch consonant onsets
        rolling_buffer = []
        recorded_chunks = []

        speech_started = False
        trailing_silence_count = 0

        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype="float32", blocksize=chunk_samples) as stream:
                for chunk_idx in range(max_total_chunks):
                    chunk, overflow = stream.read(chunk_samples)
                    chunk_flat = chunk.flatten()

                    if show_progress and sys.stdout.isatty():
                        render_recording_progress((chunk_idx + 1) * (chunk_ms / 1000.0), max_duration)

                    # Compute chunk RMS in dBFS
                    rms = np.sqrt(np.mean(chunk_flat ** 2))
                    dbfs = 20.0 * np.log10(max(rms, 1e-7))

                    if not speech_started:
                        rolling_buffer.append(chunk_flat)
                        if len(rolling_buffer) > pre_roll_chunks:
                            rolling_buffer.pop(0)

                        if dbfs > threshold_db:
                            speech_started = True
                            # Flush pre-roll buffer
                            recorded_chunks.extend(rolling_buffer)
                        elif chunk_idx >= max_pre_speech_chunks:
                            # User hesitated or spoke softly; proceed recording
                            speech_started = True
                            recorded_chunks.extend(rolling_buffer)
                    else:
                        recorded_chunks.append(chunk_flat)
                        if dbfs < threshold_db:
                            trailing_silence_count += 1
                            if trailing_silence_count >= silence_cutoff_chunks:
                                # Natural speech completion detected
                                break
                        else:
                            trailing_silence_count = 0

            if show_progress and sys.stdout.isatty():
                out = sys.__stdout__ or sys.stdout
                out.write("\r\033[2K")
                out.flush()

            if recorded_chunks:
                raw = np.concatenate(recorded_chunks)
            else:
                raw = np.zeros(int(duration * fs), dtype=np.float32)
        except Exception as e:
            print(f"⚠️ VAD streaming error ({e}), falling back to fixed recording...")
            try:
                data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
                sd.wait()
                raw = data.flatten()
            except Exception as rec_err:
                print(f"❌ Microphone recording fallback failed: {rec_err}")
                raise rec_err

    raw = np.nan_to_num(raw).astype(np.float32)

    if playback:
        if verbose:
            print("🔊 Playing back recorded audio…")
        sd.play(raw, fs)
        sd.wait()

    sf.write(filename, raw, fs)
    rec_sec = len(raw) / fs
    if verbose:
        print(f"✅ Audio saved: {filename} ({rec_sec:.2f}s)")

    return filename


def play_beep(duration=0.07, freq=1000, fs=44100):
    """
    Plays a short beep sound to cue recording.
    Includes a 100ms settling delay to prevent acoustic cue bleed into the microphone.
    """
    try:
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        beep = np.sin(2 * np.pi * freq * t).astype(np.float32)

        sd.play(beep, fs)
        sd.wait()
        # Guard delay: allows DAC decay and CoreAudio buffer flush before mic starts
        time.sleep(0.1)
    except Exception as e:
        print(f"⚠️ Audio cue beep failed: {e}")


def play_wav(filename) -> bool:
    """
    Plays a WAV file using sounddevice (blocking).
    Returns True if playback completed successfully, False otherwise.
    """
    try:
        data, fs = sf.read(filename, dtype="float32")
        sd.play(data, fs)
        sd.wait()
        return True
    except Exception as e:
        print(f"⚠️ Audio playback failed: {e}")
        return False


def trim_silence_rms(
    waveform,
    sample_rate: int = 16000,
    threshold_db: float = -35.0,
    min_speech_duration: float = 0.3,
    pad_duration: float = 0.08
):
    """
    Trims leading and trailing silence from an audio waveform using short-time RMS energy in dB.
    Accepts torch.Tensor (shape [C, T] or [T]) or np.ndarray (shape [C, T] or [T]).
    Returns the trimmed waveform preserving input type, dtype, and device.
    Guards:
      - If audio is all silence or active duration < min_speech_duration, returns original waveform.
      - Retains pad_duration (default 80ms) before/after to prevent consonant clipping.
    """
    is_torch = isinstance(waveform, torch.Tensor)
    if is_torch:
        orig_device = waveform.device
        orig_dtype = waveform.dtype
        audio_np = waveform.detach().cpu().float().numpy()
    else:
        audio_np = np.asarray(waveform, dtype=np.float32)

    orig_ndim = audio_np.ndim
    if orig_ndim == 2:
        # If shape is [channels, samples], average channels for energy detection
        mono_signal = np.mean(audio_np, axis=0) if audio_np.shape[0] < audio_np.shape[1] else np.mean(audio_np, axis=1)
    else:
        mono_signal = audio_np.flatten()

    total_samples = len(mono_signal)
    if total_samples == 0:
        return waveform

    # Frame-based RMS calculation (25ms window, 10ms hop)
    frame_len = max(int(sample_rate * 0.025), 1)
    hop_len = max(int(sample_rate * 0.010), 1)

    if total_samples < frame_len:
        return waveform

    num_frames = 1 + (total_samples - frame_len) // hop_len
    sq_signal = mono_signal ** 2
    cum_sq = np.concatenate(([0.0], np.cumsum(sq_signal)))
    frame_starts = np.arange(num_frames) * hop_len
    frame_ends = frame_starts + frame_len
    frame_energies = (cum_sq[frame_ends] - cum_sq[frame_starts]) / frame_len
    frame_rms = np.sqrt(np.maximum(frame_energies, 1e-12))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-7))

    active_frames = np.where(frame_db > threshold_db)[0]

    # If no speech detected, return original
    if len(active_frames) == 0:
        return waveform

    start_frame = active_frames[0]
    end_frame = active_frames[-1]

    start_sample = start_frame * hop_len
    end_sample = min(end_frame * hop_len + frame_len, total_samples)

    active_duration = (end_sample - start_sample) / sample_rate
    if active_duration < min_speech_duration:
        return waveform

    pad_samples = int(pad_duration * sample_rate)
    trimmed_start = max(0, start_sample - pad_samples)
    trimmed_end = min(total_samples, end_sample + pad_samples)

    if is_torch:
        if orig_ndim == 2:
            if waveform.shape[0] < waveform.shape[1]:
                return waveform[:, trimmed_start:trimmed_end]
            else:
                return waveform[trimmed_start:trimmed_end, :]
        else:
            return waveform[trimmed_start:trimmed_end]
    else:
        if orig_ndim == 2:
            if audio_np.shape[0] < audio_np.shape[1]:
                return audio_np[:, trimmed_start:trimmed_end]
            else:
                return audio_np[trimmed_start:trimmed_end, :]
        else:
            return audio_np[trimmed_start:trimmed_end]


def check_audio_silence(
    waveform_or_path,
    threshold_db: float = -48.0,
    window_ms: int = 100,
    fs: int = 16000
) -> tuple:
    """
    Checks if an audio signal or file contains no audible speech / is essentially silence.
    Uses windowed RMS (window_ms=100) to find the peak energy frame across the signal.
    Returns (is_silent: bool, peak_db: float).
    Does NOT alter, filter, or touch the audio data in any way (pure read-only measurement).
    """
    if isinstance(waveform_or_path, str):
        if not os.path.exists(waveform_or_path):
            raise FileNotFoundError(f"Audio file not found: {waveform_or_path}")
        audio_np, file_fs = sf.read(waveform_or_path, dtype="float32")
        fs = file_fs
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)
    elif isinstance(waveform_or_path, torch.Tensor):
        audio_np = waveform_or_path.detach().cpu().float().numpy()
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0) if audio_np.shape[0] < audio_np.shape[1] else np.mean(audio_np, axis=1)
    else:
        audio_np = np.asarray(waveform_or_path, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)

    audio_flat = audio_np.flatten()
    if len(audio_flat) == 0:
        return True, -100.0

    window_samples = max(int(fs * (window_ms / 1000.0)), 1)
    if len(audio_flat) < window_samples:
        rms = np.sqrt(np.mean(audio_flat ** 2))
        db = 20.0 * np.log10(max(rms, 1e-7))
        return (db < threshold_db), float(db)

    # Rolling window RMS using stride chunks for O(N) calculation (< 0.1ms)
    num_windows = len(audio_flat) // window_samples
    reshaped = audio_flat[:num_windows * window_samples].reshape(num_windows, window_samples)
    rms_per_window = np.sqrt(np.mean(reshaped ** 2, axis=1))
    peak_rms = np.max(rms_per_window)
    peak_db = float(20.0 * np.log10(max(peak_rms, 1e-7)))

    is_silent = bool(peak_db < threshold_db)
    return is_silent, peak_db
