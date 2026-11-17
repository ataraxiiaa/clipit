"""Cheap, always-on signals: audio excitement, visual change, chat activity. No GPU, no big models."""
import collections
import math
import subprocess

import cv2
import numpy as np

from . import util

SR = 16000


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def decode_audio(path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                       capture_output=True, creationflags=util.NO_WINDOW)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def audio_features(pcm: np.ndarray) -> list[dict]:
    """Per-second loudness (dB), high-band ratio (shouting has more energy above 1.5 kHz), voice-band ratio."""
    out = []
    win = np.hanning(SR)
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    hi, voice, tot = freqs >= 1500, (freqs >= 300) & (freqs <= 3400), (freqs >= 80)
    for i in range(len(pcm) // SR):
        x = pcm[i * SR:(i + 1) * SR]
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-10))
        spec = np.abs(np.fft.rfft(x * win)) ** 2
        t = spec[tot].sum() + 1e-12
        out.append({"db": 20 * math.log10(rms), "hf": float(spec[hi].sum() / t), "voice": float(spec[voice].sum() / t)})
    return out


class Baseline:
    """Rolling robust baseline (median / MAD) so 'loud' means loud *for this stream right now*."""

    def __init__(self, n):
        self.v = collections.deque(maxlen=n)

    def z(self, x, floor):
        if len(self.v) < 20:
            return 0.0
        a = np.fromiter(self.v, dtype=np.float32)
        med = float(np.median(a))
        mad = max(1.4826 * float(np.median(np.abs(a - med))), floor)
        return (x - med) / mad

    def push(self, x):
        self.v.append(x)


class AudioScorer:
    def __init__(self, baseline_sec):
        self.db, self.hf = Baseline(baseline_sec), Baseline(baseline_sec)
        self.speech_db = Baseline(baseline_sec)    # loudness of *voiced* seconds only: "how loud does this streamer normally talk"
        self.recent = collections.deque(maxlen=6)

    def score(self, f: dict) -> dict:
        floor = float(np.percentile(np.fromiter(self.db.v, dtype=np.float32), 10)) + 8.0 if len(self.db.v) >= 20 else None
        active = floor is not None and f["db"] > floor and f["db"] > -55      # clearly above this stream's noise floor
        if active and len(self.speech_db.v) >= 15:
            # judged against the stream's normal *active* level (talking / gameplay), not against silence
            z_db = self.speech_db.z(f["db"], 2.0)
            onset = f["db"] - float(np.median(np.fromiter(self.speech_db.v, dtype=np.float32)))
        elif active:
            # still learning what normal active audio looks like: only very loud audio may count meanwhile
            z_db = self.db.z(f["db"], 1.5) if f["db"] > -22 else 0.0
            onset = 12.0 if f["db"] > -22 else 0.0
        else:
            z_db = self.db.z(f["db"], 1.5)
            onset = f["db"] - (np.mean(self.recent) if len(self.recent) >= 4 else f["db"])
        z_hf = self.hf.z(f["hf"], 0.03)
        if active and z_db < 5.0:
            self.speech_db.push(f["db"])
        loud = sigmoid((z_db - 2.2) / 0.9)
        on = sigmoid((onset - 7.0) / 3.0)
        tone = sigmoid((z_hf - 2.0) / 1.0) * (1.0 if f["voice"] > 0.45 else 0.4)   # excited voice, not just noise
        s = min(1.0, 0.55 * loud + 0.25 * on + 0.20 * tone)
        # the baseline learns everything except extreme outliers (median/MAD already resist short bursts), so ordinary
        # talking becomes "normal" instead of being re-detected as a spike forever
        if z_db < 5.0:
            self.db.push(f["db"])
            self.hf.push(f["hf"])
        self.recent.append(f["db"])
        return {"score": s, "z_db": z_db, "onset_db": float(onset), "z_hf": z_hf, "active": bool(active)}


class VisualScorer:
    """Samples small grayscale frames; scores scene cuts and motion bursts relative to the stream's normal motion."""

    def __init__(self, baseline_sec, interval):
        self.interval = interval
        self.motion = Baseline(max(20, int(baseline_sec / interval)))
        self.prev = None
        self.prev_hist = None

    def frames(self, path, dur):
        w, h = 160, 90
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
                            f"fps=1/{self.interval},scale={w}:{h},format=gray", "-f", "rawvideo", "pipe:1"],
                           capture_output=True, creationflags=util.NO_WINDOW)
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
        n = len(buf) // (w * h)
        return buf[: n * w * h].reshape(n, h, w)

    def analyze(self, path, dur, t0) -> list[dict]:
        out = []
        for i, fr in enumerate(self.frames(path, dur)):
            hist = cv2.calcHist([fr], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            m = 0.0
            cut = False
            if self.prev is not None:
                m = float(np.mean(cv2.absdiff(fr, self.prev))) / 255.0
                cut = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL) < 0.55
            z = self.motion.z(m, 0.01) if self.prev is not None else 0.0
            burst = sigmoid((z - 2.5) / 1.0)
            score = 1.0 if cut else burst * 0.8
            if not cut:
                self.motion.push(m)
            out.append({"t": t0 + i * self.interval, "score": score, "motion": m, "cut": bool(cut)})
            self.prev, self.prev_hist = fr, hist
        return out


class ChatCounter:
    """Messages per stream-second; spike = last 8 s versus a longer average."""

    def __init__(self):
        self.counts = collections.Counter()
        self.total = 0
        self.t_first = None

    def add(self, t):
        self.counts[int(t)] += 1
        self.total += 1
        self.t_first = self.t_first or t

    def score(self, t) -> float:
        if not self.total or t - (self.t_first or t) < 60:
            return 0.0
        t = int(t)
        base = sum(self.counts.get(k, 0) for k in range(t - 180, t - 10)) / 170.0
        now = sum(self.counts.get(k, 0) for k in range(t - 8, t + 1)) / 9.0
        return sigmoid((now / max(base, 0.2) - 2.2) / 0.8) if now >= 1.0 else 0.0
