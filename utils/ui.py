import sys
import os
import shutil
import platform
import ctypes
import time
import threading
import re
import unicodedata
from transformers.generation.streamers import BaseStreamer

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


def hide_cursor():
    out = sys.__stdout__ or sys.stdout
    if out.isatty():
        out.write("\033[?25l")
        out.flush()


def show_cursor():
    out = sys.__stdout__ or sys.stdout
    if out.isatty():
        out.write("\033[?25h")
        out.flush()


def countdown_alert(seconds: int = 3, label: str = "Get ready to speak"):
    """
    Renders an in-place preparation countdown before microphone capture.
    Provides sensory alertness to prevent audio onset detachment.
    """
    out = sys.__stdout__ or sys.stdout
    if not out.isatty() or seconds <= 0:
        return
    hide_cursor()
    try:
        for s in range(seconds, 0, -1):
            out.write(f"\r\033[90m│\033[0m  \033[33m⏳\033[0m  {label}: \033[1;33m{s}...\033[0m   ")
            out.flush()
            time.sleep(1.0)
        out.write("\r\033[2K")
        out.flush()
    finally:
        show_cursor()


def render_recording_progress(elapsed: float, duration: float, bar_width: int = 20):
    """
    Renders a live, deterministic progress bar for audio capture in the terminal canvas.
    Example: │  🎤  Recording [██████████░░░░░░░░░░] 1.25s / 2.50s
    """
    out = sys.__stdout__ or sys.stdout
    if not out.isatty():
        return
    try:
        term_cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        term_cols = 80
    effective_bar_width = min(bar_width, max(5, term_cols - 38))
    frac = min(1.0, max(0.0, elapsed / duration)) if duration > 0 else 1.0
    filled = int(round(frac * effective_bar_width))
    bar = "█" * filled + "░" * (effective_bar_width - filled)
    out.write(
        f"\r\033[90m│\033[0m  \033[36m🎤\033[0m  \033[1;36mRecording\033[0m \033[90m[\033[36m{bar}\033[90m]\033[0m \033[37m{elapsed:.2f}s\033[0m\033[90m/\033[0m\033[90m{duration:.2f}s\033[0m   "
    )
    out.flush()


class LiveTokenStreamer(BaseStreamer):
    """
    ChatGPT-style live token streamer for NLLB-200 and causal language models.
    Streams generated subword tokens directly to stdout as they are sampled.
    """
    SPECIAL_TAGS = {
        "</s>", "<s>", "<pad>", "<unk>",
        "tel_Telu", "hin_Deva", "tam_Taml", "eng_Latn", "kan_Knda",
        "mal_Mlym", "ben_Beng", "mar_Deva", "guj_Gujr", "pan_Guru",
        "urd_Arab", "asm_Beng", "bod_Deva", "kas_Arab", "mai_Deva",
        "mni_Beng", "npi_Deva", "ory_Orya", "san_Deva", "snd_Arab"
    }

    def __init__(self, tokenizer, prefix: str = "", suffix: str = ""):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.suffix = suffix
        self.first_token = True
        self.token_cache = []
        self.print_len = 0

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, clean_up_tokenization_spaces=False)

        for tag in self.SPECIAL_TAGS:
            text = text.replace(tag, "")

        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = text[self.print_len: text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        if printable_text:
            if self.first_token:
                if self.prefix:
                    sys.stdout.write(self.prefix)
                self.first_token = False
                printable_text = printable_text.lstrip()
            sys.stdout.write(printable_text)
            sys.stdout.flush()

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, clean_up_tokenization_spaces=False)
            for tag in self.SPECIAL_TAGS:
                text = text.replace(tag, "")
            printable_text = text[self.print_len:]
            if printable_text:
                if self.first_token:
                    if self.prefix:
                        sys.stdout.write(self.prefix)
                    printable_text = printable_text.lstrip()
                sys.stdout.write(printable_text)
        if self.suffix:
            sys.stdout.write(self.suffix)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.token_cache = []
        self.print_len = 0
        self.first_token = True


def typewrite_text(text: str, delay: float = 0.010, prefix: str = "", suffix: str = ""):
    """
    Renders text with a smooth micro-cadence token stream effect (ChatGPT feel).
    Applies Unicode NFC normalization and word/grapheme-safe token chunking to prevent
    Indic script corruption (dotted circles '◌' on vowels/viramas in Telugu/Hindi).
    """
    if prefix:
        sys.stdout.write(prefix)
        sys.stdout.flush()
    if not text:
        if suffix:
            sys.stdout.write(suffix)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    # Ensure canonical Unicode composition (NFC)
    clean_text = unicodedata.normalize("NFC", text)

    if sys.stdout.isatty():
        # Token-level chunking: matches ChatGPT's natural subword/word token emission.
        # Preserves complete Brahmic/Indic grapheme clusters (consonant + virama + vattu + matra)
        # in a single write buffer so the terminal font engine renders ligatures cleanly
        # without emitting fallback dotted circles (◌) or detached matras.
        chunks = re.findall(r"\S+\s*|\s+", clean_text)
        # Smooth cadence: ~25-30ms per word chunk provides fluid ChatGPT-style streaming
        chunk_delay = max(0.025, delay * 2.5) if len(chunks) > 1 else delay
        for chunk in chunks:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            time.sleep(chunk_delay)
    else:
        sys.stdout.write(clean_text)
        sys.stdout.flush()

    if suffix:
        sys.stdout.write(suffix)
    sys.stdout.write("\n")
    sys.stdout.flush()


class Spinner:
    """
    Lightweight, thread-safe in-place spinner for neural operations.
    Runs on background efficiency cores with zero frame stutter.
    Binds directly to sys.__stdout__ to prevent global redirection hijacking.
    Rapidly updates with millisecond precision and cuts off immediately upon completion.
    """
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing...", done_message: str = None, enabled: bool = True):
        self.message = message
        self.done_message = done_message
        self.enabled = enabled
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self.elapsed = 0.0

    def _spin(self):
        out = sys.__stdout__ or sys.stdout
        if not out or not hasattr(out, "isatty") or not out.isatty():
            return
        idx = 0
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            proc = None

        while not self._stop_event.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            now = time.time()
            elapsed = now - (self._start_time or now)

            # Determine dynamic terminal width safely
            try:
                term_cols = shutil.get_terminal_size((80, 24)).columns
            except Exception:
                term_cols = 80
            # Safety margin: strictly stay 2 columns inside right margin to prevent auto-wrap
            max_visible = max(20, term_cols - 2)

            prefix_plain = f"  {frame}  "
            elapsed_plain = f" ({elapsed:.1f}s)" if elapsed >= 0.1 else ""
            elapsed_ansi = f" \033[90m({elapsed:.1f}s)\033[0m" if elapsed >= 0.1 else ""

            ram_plain = ""
            ram_ansi = ""
            if proc is not None:
                try:
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    ram_plain = f" · {rss_mb:.1f} MB"
                    ram_ansi = f" \033[90m·\033[0m \033[32m{rss_mb:.1f} MB\033[0m"
                except Exception:
                    pass

            msg = self.message
            base_visible = len(prefix_plain) + len(msg) + len(elapsed_plain)

            if base_visible + len(ram_plain) <= max_visible:
                line = f"\r\033[2K  \033[36m{frame}\033[0m  {msg}{elapsed_ansi}{ram_ansi}"
            elif base_visible <= max_visible:
                line = f"\r\033[2K  \033[36m{frame}\033[0m  {msg}{elapsed_ansi}"
            else:
                avail = max(8, max_visible - len(prefix_plain) - len(elapsed_plain) - 3)
                truncated_msg = msg[:avail] + "..."
                line = f"\r\033[2K  \033[36m{frame}\033[0m  {truncated_msg}{elapsed_ansi}"

            out.write(line)
            out.flush()

            # Responsive wait: instantly wakes up when stop() is called without waiting for sleep
            if self._stop_event.wait(0.08):
                break
            idx += 1

    def update_message(self, new_message: str):
        self.message = new_message

    def stop(self) -> float:
        """
        Immediately stops the spinner, freezes the elapsed timer, and clears the terminal line.
        Guarantees zero overrun or false hope after an operation completes.
        """
        if self._stop_event.is_set():
            return self.elapsed
        self._stop_event.set()
        if self._start_time is not None:
            self.elapsed = time.time() - self._start_time
        if self._thread is not None and threading.current_thread() != self._thread:
            self._thread.join(timeout=0.5)
        out = sys.__stdout__ or sys.stdout
        if out and hasattr(out, "isatty") and out.isatty():
            out.write("\r\033[2K")
            out.flush()
        show_cursor()
        return self.elapsed

    def __enter__(self):
        if not self.enabled:
            return self
        hide_cursor()
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        self.stop()
        if exc_type is not None:
            out = sys.__stdout__ or sys.stdout
            if out.isatty():
                out.write(f"  \033[31m✖\033[0m  {self.message} (Failed)\n")
                out.flush()
        elif self.done_message:
            out = sys.__stdout__ or sys.stdout
            if out.isatty():
                out.write(f"  \033[32m✔\033[0m  {self.done_message}\n")
                out.flush()


class StudioCanvas:
    """
    Studio-Grade Terminal Canvas for Voice2Voice.
    Renders a unified, geometric glass card with live confidence routing and zero flickering.
    """
    def __init__(self, stage_and_flush: bool = True, enabled: bool = True):
        self.stage_and_flush = stage_and_flush
        self.enabled = enabled
        self.active_spinner = None

    def print_header(self):
        if not self.enabled:
            return
        tag = "Stage-and-Flush · 4 Compute Threads" if self.stage_and_flush else "Resident Mode"
        print(f"\n\033[90m╭─\033[0m \033[1;37mVOICE2VOICE\033[0m \033[90m─────────────────────────────────────── [{tag}] ─╮\033[0m")
        print("\033[90m│\033[0m")

    def print_step(self, label: str, content: str, icon: str = "\033[32m✔\033[0m"):
        if not self.enabled:
            return
        # Clean 16-char label column alignment
        padded_label = f"{label:<15}"
        print(f"\033[90m│\033[0m  {icon}  \033[36m{padded_label}\033[0m \033[90m│\033[0m {content}")

    def stream_step(self, label: str, text: str, prefix_quote: bool = True, suffix_tag: str = ""):
        if not self.enabled:
            return
        padded_label = f"{label:<15}"
        lead = f"\033[90m│\033[0m  \033[32m✔\033[0m  \033[36m{padded_label}\033[0m \033[90m│\033[0m "
        # 🪷 Terminal font typography: macOS Terminal.app synthetically double-strikes bold glyphs,
        # which breaks Brahmic/Indic vowel signs (matras, viramas) and conjunct ligatures.
        # Use bright non-bold \033[97m for Indic scripts; preserve bold \033[1;37m for Latin scripts.
        is_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in text) if text else False
        color_code = '\033[97m' if is_indic else '\033[1;37m'
        if prefix_quote:
            lead += color_code + '"'
            suffix = f'"\033[0m {suffix_tag}'
        else:
            lead += color_code
            suffix = f'\033[0m {suffix_tag}'
        typewrite_text(text, delay=0.010, prefix=lead, suffix=suffix)

    def render_stage_sync(self, stage_text: str):
        """
        Renders the live 3-stage pipeline progress indicator on a single updating line.
        Strictly uses \\r\\033[K (in-place line update) to prevent ANSI cursor-up math errors on wrapped text.
        """
        if not self.enabled:
            return
        out = sys.__stdout__ or sys.stdout
        if out and hasattr(out, "isatty") and out.isatty():
            out.write(f"\r\033[K\033[90m│\033[0m  \033[33m⚡ Pipeline:\033[0m {stage_text}")
            out.flush()

    def clear_stage_sync(self):
        """Clears the live stage indicator line cleanly."""
        if not self.enabled:
            return
        out = sys.__stdout__ or sys.stdout
        if out and hasattr(out, "isatty") and out.isatty():
            out.write("\r\033[K")
            out.flush()

    def print_footer(
        self,
        timings: dict,
        active_ram_mb: float,
        avg_ram_mb: float = None,
        peak_ram_mb: float = None,
        cores: int = 4
    ):
        if not self.enabled:
            return
        self.clear_stage_sync()
        asr = timings.get("asr_time_sec", 0.0)
        nmt = timings.get("translation_time_sec", 0.0)
        tts = timings.get("tts_synthesis_time_sec", 0.0)
        playback = timings.get("playback_time_sec", 0.0)
        compute = asr + nmt + tts
        avg_model = compute / 3.0

        if tts >= 0.005:
            tts_display = f"TTS \033[36m{tts:.2f}s\033[0m"
        elif playback > 0.0:
            tts_display = "TTS \033[90m(OS Voice)\033[0m"
        else:
            tts_display = f"TTS \033[36m{tts:.2f}s\033[0m"

        avg_gb = (avg_ram_mb or active_ram_mb) / 1024.0
        peak_gb = (peak_ram_mb or active_ram_mb) / 1024.0

        print("\033[90m│\033[0m")
        print("\033[90m├──────────────────────────────────────────────────────────────────────────┤\033[0m")
        # Line 1: Pure 3-Stage Neural Compute + per-stage average (eliminates misleading wall-clock total)
        print(f"\033[90m│\033[0m  \033[33m⚡ 3-Stage Compute: {compute:.2f}s\033[0m (avg \033[36m{avg_model:.2f}s\033[0m/model: ASR \033[36m{asr:.2f}s\033[0m │ NMT \033[36m{nmt:.2f}s\033[0m │ {tts_display})")

        # Line 2: Realistic Average RAM (~1.2 GB) and Peak RAM (<2.5 GB) + Thread budget
        print(f"\033[90m│\033[0m  \033[34m💾 RAM:\033[0m \033[32m{avg_gb:.2f} GB avg\033[0m · peak \033[35m{peak_gb:.2f} GB\033[0m · \033[90m{cores} Threads\033[0m")
        print("\033[90m╰──────────────────────────────────────────────────────────────────────────╯\033[0m\n")


def print_compact_header(stage_and_flush: bool = True):
    canvas = StudioCanvas(stage_and_flush=stage_and_flush, enabled=True)
    canvas.print_header()
