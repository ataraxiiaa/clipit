"""Logging ring buffer (shown in the dashboard), subprocess helpers, small utilities."""
import collections
import logging
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler

from . import config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
RING: collections.deque = collections.deque(maxlen=500)


class _Ring(logging.Handler):
    def emit(self, record):
        RING.append({"t": round(record.created, 1), "level": record.levelname, "src": record.name.replace("clipper.", ""),
                     "msg": record.getMessage()})


def setup_logging():
    root = logging.getLogger("clipper")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(config.SETTINGS.data_dir / "clipper.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    root.addHandler(_Ring())


def run(cmd, timeout=600, **kw) -> subprocess.CompletedProcess:
    """Run a command, raising a readable error with the tail of stderr on failure."""
    p = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NO_WINDOW, **kw)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "ignore")[-600:] if isinstance(p.stderr, bytes) else str(p.stderr)[-600:]
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}): {tail}")
    return p


def probe_duration(path) -> float:
    try:
        p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], timeout=20)
        return float(p.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        return 0.0


def now() -> float:
    return time.time()
