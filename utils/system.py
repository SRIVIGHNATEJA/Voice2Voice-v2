import os
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import platform
import psutil
import warnings
import logging
import transformers

# 🤫 Suppress noisy logs & warnings for clean terminal output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# Device Configuration (CPU-ONLY for research)
# ─────────────────────────────────────────────

device = "cpu"

# ⚙️ Deterministic PyTorch CPU intra-op thread pool configuration
# Sets 4 compute threads to prevent CPU thread oversubscription on host CPU,
# reserving host OS scheduling and audio I/O capacity.
configured_compute_threads = 4
actual_physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
actual_logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
physical_cores = actual_physical_cores  # Accurate physical core count
torch.set_num_threads(configured_compute_threads)
try:
    torch.set_num_interop_threads(2)
except Exception:
    pass

# ─────────────────────────────────────────────
# System Metadata (for reproducibility & logging)
# ─────────────────────────────────────────────

def get_system_info():
    return {
        "device": device,
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor(),
        "configured_compute_threads": configured_compute_threads,
        "physical_cores": actual_physical_cores,
        "logical_cores": actual_logical_cores,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }
