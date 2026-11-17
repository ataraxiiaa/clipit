"""GPU scheduling: one lock so Whisper and the LLM never run at the same moment, VRAM readout, DLL setup."""
import logging
import os
import sys
import threading
from pathlib import Path

log = logging.getLogger("clipper.gpu")
GPU_LOCK = threading.RLock()
_nvml = None


def setup_cuda_dlls():
    """faster-whisper (ctranslate2) needs cuBLAS/cuDNN; they come from pip wheels, so add them to the DLL path."""
    for sp in map(Path, sys.path):
        base = sp / "nvidia"
        if base.is_dir():
            for sub in base.iterdir():
                b = sub / "bin"
                if b.is_dir():
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(str(b))
                    except Exception:  # noqa: BLE001
                        pass


def vram():
    """(used_mb, total_mb, util_pct) or None if NVML is unavailable."""
    global _nvml
    try:
        import pynvml
        if _nvml is None:
            pynvml.nvmlInit()
            _nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(_nvml)
        u = pynvml.nvmlDeviceGetUtilizationRates(_nvml)
        return round(m.used / 2**20), round(m.total / 2**20), u.gpu
    except Exception:  # noqa: BLE001
        return None


def free_mb():
    v = vram()
    return (v[1] - v[0]) if v else 0
