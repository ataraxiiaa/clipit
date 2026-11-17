"""Recording into a rolling buffer of stream-copied .ts pieces.

Design adapted from PeakCut's live engine (MIT, github.com/haziqahmedqasmi/peakcut): ffmpeg's segment muxer
copies the stream without re-encoding, so the recorder never depends on AI speed. Unlike PeakCut, analysis reads the
finished pieces themselves (one clock, no audio/video drift), and pieces covered by a pending clip are pinned.
"""
import logging
import re
import shutil
import sys
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import config, util

log = logging.getLogger("clipper.capture")
def _tool(name):
    """Find a CLI tool: the project's venv first (start.bat does not activate it), then PATH."""
    for d in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
        for ext in (".exe", ""):
            if (d / (name + ext)).is_file():
                return str(d / (name + ext))
    return shutil.which(name) or name


STREAMLINK = _tool("streamlink")
PLATFORM_RE = re.compile(r"^https?://(?:www\.|m\.)?(twitch\.tv|youtube\.com|youtu\.be|kick\.com)/", re.I)


def platform_of(url: str) -> str:
    m = PLATFORM_RE.match(url)
    if not m:
        return "direct"
    return {"twitch.tv": "twitch", "kick.com": "kick"}.get(m.group(1).lower(), "youtube")


@dataclass
class Segment:
    path: Path
    t0: float       # stream-timeline start (epoch seconds, continuous inside one connection)
    dur: float

    @property
    def t1(self):
        return self.t0 + self.dur


class SegmentStore:
    """Thread-safe index of buffer pieces with pinning and size/age pruning."""

    def __init__(self, folder: Path):
        self.dir = folder
        self.dir.mkdir(parents=True, exist_ok=True)
        self.segs: list[Segment] = []
        self.pins: dict = {}
        self.lock = threading.RLock()

    def add(self, path: Path, label_t: float, dur: float) -> Segment:
        with self.lock:
            prev = self.segs[-1] if self.segs else None
            # continuous timeline across pieces; a gap bigger than 3 s (reconnect) restarts from the wall clock label
            t0 = prev.t1 if prev and abs(label_t - prev.t1) < 3.0 else label_t
            s = Segment(path, t0, dur)
            self.segs.append(s)
            return s

    def pin(self, key, t0, t1):
        with self.lock:
            self.pins[key] = (t0, t1)

    def unpin(self, key):
        with self.lock:
            self.pins.pop(key, None)

    def range(self, t0, t1) -> list[Segment]:
        """Contiguous pieces overlapping [t0, t1], restricted to the run that contains the midpoint."""
        with self.lock:
            sel = [s for s in self.segs if s.t1 > t0 and s.t0 < t1 and s.path.exists()]
        if not sel:
            return []
        runs, cur = [], [sel[0]]
        for s in sel[1:]:
            if abs(s.t0 - cur[-1].t1) < 0.5:
                cur.append(s)
            else:
                runs.append(cur)
                cur = [s]
        runs.append(cur)
        mid = (t0 + t1) / 2
        return min(runs, key=lambda r: 0 if r[0].t0 <= mid <= r[-1].t1 else min(abs(r[0].t0 - mid), abs(r[-1].t1 - mid)))

    def latest_end(self):
        with self.lock:
            return self.segs[-1].t1 if self.segs else 0.0

    def buffer_seconds(self):
        with self.lock:
            return round(self.segs[-1].t1 - self.segs[0].t0) if self.segs else 0

    def disk_bytes(self):
        with self.lock:
            return sum(s.path.stat().st_size for s in self.segs if s.path.exists())

    def prune(self, keep_sec, max_bytes):
        with self.lock:
            if not self.segs:
                return
            cutoff = self.segs[-1].t1 - keep_sec
            pins = list(self.pins.values())
            total = self.disk_bytes()
            keep = []
            for s in self.segs:
                pinned = any(s.t1 > a and s.t0 < b for a, b in pins)
                too_old = s.t1 < cutoff
                over = total > max_bytes
                if not pinned and (too_old or over) and s is not self.segs[-1]:
                    try:
                        total -= s.path.stat().st_size
                        s.path.unlink()
                    except OSError:
                        pass
                else:
                    keep.append(s)
            self.segs = keep


class Capture:
    """Watches one URL: waits until it is live, records, reconnects after drops. Runs in its own thread."""

    def __init__(self, stream: dict, on_segment):
        self.stream = stream
        self.url = stream["url"]
        self.platform = platform_of(self.url)
        self.folder = config.SETTINGS.data_dir / "buffers" / str(stream["id"])
        shutil.rmtree(self.folder, ignore_errors=True)
        self.store = SegmentStore(self.folder)
        self.on_segment = on_segment
        self.stop_ev = threading.Event()
        self.state = "idle"       # idle | offline | connecting | recording | reconnecting | stopped
        self.state_note = ""
        self.started_at = None
        self.reconnects = 0
        self.meta = {}
        self.thread = None
        self._seen: set = set()
        self._procs: list = []

    # -------------------------------------------------------------- lifecycle
    def start(self):
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"capture-{self.stream['id']}")
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        self._kill()
        self.state = "stopped"

    def _kill(self):
        for p in self._procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------------------- liveness
    def _stream_url_probe(self):
        """Returns True if the channel is live (streamlink can resolve a stream)."""
        if self.platform == "direct":
            return True
        q = config.get("capture.quality")
        p = subprocess.run([STREAMLINK, "--json", self.url, q], capture_output=True, timeout=45,
                           creationflags=util.NO_WINDOW)
        try:
            import json
            data = json.loads(p.stdout.decode("utf-8", "ignore") or "{}")
        except Exception:  # noqa: BLE001
            return False
        # with a quality argument streamlink returns the chosen stream ({"type","url"}); without one, a {"streams": {...}} map
        if data.get("error") or not (data.get("url") or data.get("streams")):
            return False
        self.meta = {"title": (data.get("metadata") or {}).get("title"), "author": (data.get("metadata") or {}).get("author"),
                     "category": (data.get("metadata") or {}).get("category")}
        return True

    # -------------------------------------------------------------- main loop
    def _run(self):
        delay = config.get("capture.reconnect_delay_sec")
        while not self.stop_ev.is_set():
            try:
                self.state = "connecting"
                if not self._stream_url_probe():
                    self.state, self.state_note = "offline", "Waiting for the stream to go live"
                    self.stop_ev.wait(config.get("capture.offline_poll_sec"))
                    continue
                got = self._record_once()
                if self.stop_ev.is_set():
                    break
                self.reconnects += 1
                self.state, self.state_note = "reconnecting", f"Connection lost after {got} pieces; retrying"
                log.warning("stream %s dropped (%d pieces); reconnecting in %ss", self.stream["id"], got, delay)
                self.stop_ev.wait(delay)
                delay = 5 if got >= 3 else min(60, delay * 2)
            except Exception as e:  # noqa: BLE001 - the loop must survive anything
                log.exception("capture loop error: %s", e)
                self.state, self.state_note = "reconnecting", str(e)[-120:]
                self.stop_ev.wait(delay)
        self.state = "stopped"

    def _record_once(self) -> int:
        seg_sec = config.get("capture.segment_sec")
        part = time.strftime("%H%M%S")
        pattern = str(self.folder / f"{part}_%05d.ts")
        ff = ["ffmpeg", "-v", "error", "-y"]
        src_proc = None
        if self.platform == "direct":
            if Path(self.url).exists():          # local file (testing): play it at real-time speed like a live source
                ff += ["-re"]
            ff += ["-i", self.url]
        else:
            src_proc = subprocess.Popen(
                [STREAMLINK, "--stdout", "--quiet", "--retry-streams", "2", "--retry-max", "3", "--hls-live-edge", "3",
                 *(["--twitch-low-latency"] if self.platform == "twitch" else []), self.url, config.get("capture.quality")],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=util.NO_WINDOW)
            self._procs.append(src_proc)
            ff += ["-i", "pipe:0"]
        ff += ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-f", "segment", "-segment_time", str(seg_sec),
               "-segment_format", "mpegts", "-reset_timestamps", "1", pattern]
        errlog = open(self.folder.parent / f"ffmpeg_{self.stream['id']}.log", "ab")
        proc = subprocess.Popen(ff, stdin=src_proc.stdout if src_proc else None, stderr=errlog,
                                creationflags=util.NO_WINDOW)
        self._procs.append(proc)
        if src_proc:
            src_proc.stdout.close()
        self.state, self.state_note = "recording", ""
        self.started_at = self.started_at or time.time()
        pieces, last_new = 0, time.time()
        self._seen = set()
        prefix = f"{part}_"
        try:
            while not self.stop_ev.is_set():
                time.sleep(1.0)
                done = sorted(p for p in self.folder.glob(f"{prefix}*.ts"))
                finished = done if proc.poll() is not None else done[:-1]   # newest piece is still being written
                for p in finished:
                    if p.name in self._seen:
                        continue
                    self._seen.add(p.name)
                    dur = util.probe_duration(p)
                    if dur <= 0.2:
                        p.unlink(missing_ok=True)
                        continue
                    label = p.stat().st_mtime - dur
                    seg = self.store.add(p, label, dur)
                    pieces += 1
                    last_new = time.time()
                    try:
                        self.on_segment(seg)
                    except Exception:  # noqa: BLE001
                        log.exception("segment callback failed")
                self.store.prune(config.get("capture.buffer_sec"), int(config.get("capture.max_buffer_gb") * 2**30))
                if proc.poll() is not None and len(self._seen) >= len(done):
                    break
                if time.time() - last_new > config.get("capture.stall_sec") + seg_sec * 2:
                    log.warning("stream %s stalled; restarting recorder", self.stream["id"])
                    break
        finally:
            self._kill()
            for p in (proc, src_proc):
                try:
                    if p:
                        p.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            self._procs.clear()
            errlog.close()
        return pieces
