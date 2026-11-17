"""Clip an existing video: a YouTube (or other yt-dlp supported) link, or a local file. Runs at full speed, not real time.

The video is downloaded, cut into the same 4 s pieces a live recording produces, and fed through the *same* analyzer and
candidate pipeline as a live stream, so scoring, merging, dedupe and export behave identically.
Only use videos you are authorized to clip; nothing here bypasses logins, DRM or paywalls.
"""
import logging
import re
import shutil
import threading
import time
from pathlib import Path

from . import capture, config, db, pipeline, util

log = logging.getLogger("clipper.vod")
JOBS: dict[int, "VodJob"] = {}


class _Quiet:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


class VodJob:
    def __init__(self, url: str, streamer: str):
        self.url, self.streamer = url.strip(), streamer.strip()
        self.is_file = Path(self.url).exists()
        self.title = Path(self.url).stem if self.is_file else ""
        self.state, self.note, self.progress = "queued", "", 0.0
        self.clips = 0
        self.candidates = 0
        self.cancel = threading.Event()
        self.started, self.finished = time.time(), None
        self.dir = None
        self.stream_id = db.x("INSERT INTO streams(url,name,streamer,platform,created) VALUES(?,?,?,?,?)",
                              (self.url, self.url, self.streamer or "video", "vod", time.time()))
        self.id = self.stream_id
        self.dir = config.SETTINGS.data_dir / "vods" / str(self.id)
        self.dir.mkdir(parents=True, exist_ok=True)
        JOBS[self.id] = self
        threading.Thread(target=self._run, daemon=True, name=f"vod-{self.id}").start()

    # ------------------------------------------------------------------ steps
    def _download(self, height: int) -> Path:
        import yt_dlp
        self.state, self.note = "downloading", "Starting download"

        def hook(d):
            if self.cancel.is_set():
                raise yt_dlp.utils.DownloadCancelled()
            if d.get("status") == "downloading":
                tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                self.progress = (d.get("downloaded_bytes", 0) / tot) if tot else 0.0
                self.note = f"Downloading {self.progress * 100:.0f}%"

        h = int(height)
        opts = {"format": f"bv*[height<={h}][vcodec^=avc1]+ba[acodec^=mp4a]/b[height<={h}][ext=mp4]/bv*[height<={h}]+ba/b",
                "outtmpl": str(self.dir / "source.%(ext)s"), "noplaylist": True, "quiet": True, "no_warnings": True,
                "merge_output_format": "mp4", "progress_hooks": [hook], "logger": _Quiet(), "retries": 5}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(self.url, download=False)
            if info.get("is_live"):
                raise RuntimeError("This is a live stream: add it as a stream instead (it is recorded live).")
            max_h = float(config.get("vod.max_hours", 6))
            if (info.get("duration") or 0) > max_h * 3600:
                raise RuntimeError(f"Video is longer than {max_h:g} h (vod.max_hours in config.yaml).")
            self.title = info.get("title") or self.title
            if not self.streamer:
                self.streamer = info.get("uploader") or info.get("channel") or "video"
                db.update("streams", self.id, streamer=self.streamer, name=self.title)
            ydl.download([self.url])
        files = [p for p in self.dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
        if not files:
            raise RuntimeError("download produced no file")
        return max(files, key=lambda p: p.stat().st_size)

    def _fetch_hq(self, start: float, end: float, out: Path):
        """Download only [start, end] of the video in full quality, then trim it frame-accurately into `out`."""
        import yt_dlp
        from yt_dlp.utils import download_range_func
        from . import render
        pad = 2.0
        a, b = max(0.0, start - pad), end + pad
        tmp = self.dir / f"hq_{int(start)}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        h = int(config.get("vod.clip_height", 1080))
        opts = {"format": f"bv*[height<={h}][vcodec^=avc1]+ba[acodec^=mp4a]/b[height<={h}][ext=mp4]/bv*[height<={h}]+ba/b",
                "outtmpl": str(tmp / "hq.%(ext)s"), "noplaylist": True, "quiet": True, "no_warnings": True,
                "merge_output_format": "mp4", "logger": _Quiet(), "retries": 5,
                "download_ranges": download_range_func(None, [(a, b)]), "force_keyframes_at_cuts": True}
        try:
            self.note = f"Downloading HD section at {int(start // 60)}:{int(start % 60):02d}"
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([self.url])
            files = sorted(tmp.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True) or \
                [p for p in tmp.iterdir() if p.suffix not in (".part", ".ytdl")]
            if not files:
                raise RuntimeError("no HD file produced")
            out.parent.mkdir(parents=True, exist_ok=True)
            base = ["ffmpeg", "-y", "-v", "error", "-i", str(files[0]), "-ss", f"{start - a:.3f}", "-t", f"{end - start:.3f}"]
            render._encode(base, [], out, cq=19, audio=True, full_fps=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            self.note = "Finding highlights"

    def _segment(self, src: Path) -> list[Path]:
        self.state, self.note, self.progress = "preparing", "Cutting into analysis pieces", 0.0
        p = util.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "csv=p=0", str(src)])
        codecs = p.stdout.decode().split()
        copy_ok = any("h264" in c for c in codecs) and (not any("audio" in c for c in codecs) or any("aac" in c for c in codecs))
        enc = ["-c", "copy"] if copy_ok else ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-g", "60", "-c:a", "aac"]
        seg = self.dir / "seg"
        seg.mkdir(exist_ok=True)
        util.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", *enc, "-f", "segment",
                  "-segment_time", str(config.get("capture.segment_sec")), "-segment_format", "mpegts", "-reset_timestamps", "1",
                  str(seg / "s_%05d.ts")], timeout=7200)
        return sorted(seg.glob("s_*.ts"))

    def _run(self):
        try:
            lowres = (not self.is_file) and config.get("vod.quality_mode", "lowres_first") == "lowres_first"
            src = Path(self.url) if self.is_file else self._download(
                int(config.get("vod.analysis_height", 360)) if lowres else int(config.get("vod.clip_height", 1080)))
            if self.cancel.is_set():
                raise InterruptedError
            pieces = self._segment(src)
            if not pieces:
                raise RuntimeError("could not read any video from this source")
            store = capture.SegmentStore(self.dir / "seg")
            stream = {"id": self.id, "url": self.url, "streamer": self.streamer, "title": self.title}
            ready: list = []
            an = pipeline.Analyzer(stream, store, lambda a, c: ready.append(c))
            if lowres:
                an.fetch_hq = self._fetch_hq
            self.state, self.note = "analyzing", "Finding highlights"
            clock = 0.0
            durs = [util.probe_duration(p) for p in pieces]
            total = sum(durs) or 1.0
            for p, d in zip(pieces, durs):
                if self.cancel.is_set():
                    raise InterruptedError
                if d <= 0.2:
                    continue
                seg = store.add(p, clock, d)
                clock += d
                an.process_segment(seg)
                self._drain(an, ready)
                self.progress = clock / total
                self.candidates = an.stats["candidates"]
            for c in list(an.pending):          # candidates still waiting for post-roll when the video ended
                ready.append(c)
            an.pending.clear()
            self._drain(an, ready)
            self.state, self.note, self.progress = "done", f"{self.clips} clip(s) found", 1.0
        except InterruptedError:
            self.state, self.note = "cancelled", "Cancelled"
        except Exception as e:  # noqa: BLE001
            log.exception("video job %s failed", self.id)
            msg = str(e)
            self.state, self.note = "error", (msg[-300:] if "DownloadCancelled" not in msg else "Cancelled")
        finally:
            self.finished = time.time()
            shutil.rmtree(self.dir / "seg", ignore_errors=True)
            if not (self.is_file or config.get("vod.keep_source", False)):
                for f in self.dir.glob("source.*"):
                    f.unlink(missing_ok=True)
            log.info("video job %s finished: %s (%s)", self.id, self.state, self.note)

    def _drain(self, an, ready):
        while ready:
            c = ready.pop(0)
            if self.cancel.is_set():
                return
            try:
                res = pipeline.process_candidate(an, c)
                if res.get("ok"):
                    self.clips += 1
            except Exception:  # noqa: BLE001
                log.exception("candidate failed")
            finally:
                an.store.unpin(c["id"])

    def status(self):
        return {"id": self.id, "url": self.url, "title": self.title, "streamer": self.streamer, "state": self.state,
                "note": self.note, "progress": round(self.progress, 3), "clips": self.clips, "candidates": self.candidates,
                "started": self.started, "finished": self.finished}


def start(url: str, streamer: str = "") -> VodJob:
    url = url.strip()
    if not (re.match(r"^https?://", url) or Path(url).exists()):
        raise ValueError("Enter a video link (YouTube etc.) or the path of a video file.")
    if sum(1 for j in JOBS.values() if j.state in ("queued", "downloading", "preparing", "analyzing")) >= 2:
        raise ValueError("Two videos are already being processed. Wait for one to finish.")
    return VodJob(url, streamer)
