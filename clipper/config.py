"""Settings: config.yaml defaults + data/settings.json overrides (edited from the dashboard)."""
import copy
import json
import threading
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_lock = threading.RLock()


def _merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class Settings:
    # sections the dashboard may change at runtime
    EDITABLE = {"scoring", "clips", "export", "analysis", "asr", "llm", "capture", "codex"}

    def __init__(self, path: Path = ROOT / "config.yaml"):
        self.path = path
        self.d = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.data_dir = (ROOT / self.d["paths"]["data"]).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_path = self.data_dir / "settings.json"
        if self.overlay_path.exists():
            try:
                _merge(self.d, json.loads(self.overlay_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - a corrupt overlay must not stop startup
                pass

    def get(self, dotted: str, default=None):
        cur = self.d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def snapshot(self) -> dict:
        with _lock:
            return copy.deepcopy({k: v for k, v in self.d.items() if k in self.EDITABLE})

    def update(self, patch: dict) -> dict:
        patch = {k: v for k, v in patch.items() if k in self.EDITABLE}
        with _lock:
            _merge(self.d, patch)
            self.d["capture"]["buffer_sec"] = max(90, int(self.d["capture"]["buffer_sec"]))
            try:
                old = json.loads(self.overlay_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                old = {}
            self.overlay_path.write_text(json.dumps(_merge(old, patch), indent=1), encoding="utf-8")
        return self.snapshot()


SETTINGS = Settings()


def get(dotted: str, default=None):
    return SETTINGS.get(dotted, default)
