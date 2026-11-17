"""Orchestration: per-stream analyzer (cheap signals every segment) -> candidates -> serialized GPU job worker
(Whisper -> lexicon -> Ollama judge -> boundaries -> dedupe -> raw cut -> optional vertical export)."""
import collections
import json
import logging
import queue
import re
import socket
import ssl
import threading
import time
from pathlib import Path

import numpy as np

from . import asr, capture, config, db, gpu, judge, render, scoring, signals

log = logging.getLogger("clipper.pipeline")


# =============================================================== analyzer (cheap, CPU only)
class Analyzer:
    def __init__(self, stream: dict, store: capture.SegmentStore, on_candidate):
        self.stream, self.store, self.on_candidate = stream, store, on_candidate
        bs = config.get("analysis.baseline_sec")
        self.audio = signals.AudioScorer(bs)
        self.visual = signals.VisualScorer(bs, config.get("analysis.frame_interval_sec"))
        self.chat = signals.ChatCounter()
        self.chat_on = False
        self.timeline: collections.OrderedDict = collections.OrderedDict()   # int second -> dict
        self.pending: list[dict] = []
        self.last_trigger = -1e9
        self.last_scan = None
        self.t_first = None
        self.stats = {"segments": 0, "cheap_now": 0.0, "candidates": 0, "last_audio": 0.0}
        self.lock = threading.RLock()

    def process_segment(self, seg: capture.Segment):
        pcm = signals.decode_audio(seg.path)
        feats = signals.audio_features(pcm)
        vis = self.visual.analyze(seg.path, seg.dur, seg.t0)
        self.t_first = self.t_first if self.t_first is not None else seg.t0
        w = config.get("analysis.warmup_sec")
        for i, f in enumerate(feats):
            t = seg.t0 + i
            a = self.audio.score(f)
            v = max([x["score"] for x in vis if t - 1 <= x["t"] < t + 2.5] or [0.0])
            cut = any(x["cut"] for x in vis if t - 1 <= x["t"] < t + 2.5)
            c = self.chat.score(t) if (self.chat_on or self.chat.total) else 0.0
            entry = {"db": round(f["db"], 1), "voice": round(f["voice"], 2), "a": round(a["score"], 3), "v": round(v, 3),
                     "c": round(c, 3), "cut": cut, "act": a["active"]}
            with self.lock:
                self.timeline[int(t)] = entry
                while len(self.timeline) > 1800:
                    self.timeline.popitem(last=False)
            cheap = scoring.cheap_score(a["score"], v, c, self.chat_on or self.chat.total > 0)
            self.stats.update(cheap_now=round(cheap, 2), last_audio=round(a["score"], 2))
            if t - self.t_first >= w and cheap >= config.get("scoring.candidate_threshold") \
                    and t - self.last_trigger >= config.get("scoring.cooldown_sec"):
                self.last_trigger = t
                self._add(t, "signal", cheap)
        self.stats["segments"] += 1
        self._scan(seg.t1)
        self.release_ready()

    def _signals_near(self, t):
        with self.lock:
            rows = [self.timeline[k] for k in range(int(t) - 8, int(t) + 3) if k in self.timeline]
        if not rows:
            return {"audio": 0, "visual": 0, "chat": 0}
        return {"audio": max(r["a"] for r in rows), "visual": max(r["v"] for r in rows), "chat": max(r["c"] for r in rows)}

    def _add(self, event_t, trigger, cheap):
        pre, post = config.get("clips.pre_sec"), config.get("clips.post_sec")
        if trigger == "scan":
            post = max(post, config.get("analysis.scan_every_sec") * 0.4)
        cid = f"{self.stream['id']}:{int(event_t)}"
        self.store.pin(cid, event_t - pre - 10, event_t + post + 10)
        cand = {"id": cid, "event_t": event_t, "trigger": trigger, "cheap": cheap, "signals": self._signals_near(event_t),
                "ready_at": event_t + post + 6}
        with self.lock:
            self.pending.append(cand)
        self.stats["candidates"] += 1
        log.info("stream %s candidate (%s) cheap=%.2f at %s", self.stream["id"], trigger, cheap, time.strftime("%H:%M:%S", time.localtime(event_t)))

    def _scan(self, t_now):
        every = config.get("analysis.scan_every_sec")
        if not every:
            return
        if self.last_scan is None:
            self.last_scan = t_now
        if t_now - self.last_scan < every or t_now - self.last_trigger < config.get("scoring.cooldown_sec"):
            return
        self.last_scan = t_now
        win = config.get("analysis.scan_window_sec")
        with self.lock:
            rows = [self.timeline[k] for k in range(int(t_now - win), int(t_now)) if k in self.timeline]
        if len(rows) > win * 0.6 and np.mean([r["act"] for r in rows]) > 0.25:     # someone is talking / playing: worth a look
            self._add(t_now - 22, "scan", 0.0)

    def release_ready(self):
        end = self.store.latest_end()
        with self.lock:
            ready = [c for c in self.pending if c["ready_at"] <= end]
            self.pending = [c for c in self.pending if c["ready_at"] > end]
        for c in ready:
            self.on_candidate(self, c)

    def manual(self, last_sec):
        end = self.store.latest_end()
        if end <= 0:
            raise ValueError("Nothing recorded yet")
        post = config.get("clips.post_sec")
        ev = end - min(last_sec, 30) / 2
        cid = f"{self.stream['id']}:m{int(end)}"
        self.store.pin(cid, end - last_sec - 10, end + 5)
        cand = {"id": cid, "event_t": ev, "trigger": "manual", "cheap": 1.0, "signals": self._signals_near(ev),
                "ready_at": 0, "manual_range": (end - last_sec, end)}
        self.on_candidate(self, cand)


# =============================================================== GPU job worker
class JobWorker:
    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=int(config.get("analysis.queue_max", 12)))
        self.current = None
        self.done = 0
        self.failed = 0
        self.thread = threading.Thread(target=self._loop, daemon=True, name="gpu-jobs")
        self.thread.start()

    def submit(self, analyzer, cand):
        try:
            self.q.put_nowait((analyzer, cand))
        except queue.Full:
            analyzer.store.unpin(cand["id"])
            log.warning("job queue full: candidate %s dropped", cand["id"])

    def _loop(self):
        while True:
            try:
                analyzer, cand = self.q.get(timeout=30)
            except queue.Empty:
                asr.idle_check()
                continue
            self.current = cand["id"]
            try:
                process_candidate(analyzer, cand)
                self.done += 1
            except Exception as e:  # noqa: BLE001 - one bad job must not kill the worker
                self.failed += 1
                log.exception("job %s failed: %s", cand["id"], e)
            finally:
                analyzer.store.unpin(cand["id"])
                self.current = None


def _read_audio(segs):
    origin = segs[0].t0
    n = int((segs[-1].t1 - origin) * signals.SR) + signals.SR
    arr = np.zeros(n, dtype=np.float32)
    for s in segs:
        pcm = signals.decode_audio(s.path)
        i = int(round((s.t0 - origin) * signals.SR))
        arr[i:i + len(pcm)] = pcm[: max(0, n - i)]
    return arr, origin


def _norm_words(words):
    return set(re.findall(r"[a-z']{3,}", " ".join(w["w"] for w in words).lower()))


def process_candidate(an: Analyzer, cand: dict, evaluate_only=False) -> dict:
    """Runs the expensive stages for one candidate. Returns the result dict (also written to the DB unless evaluate_only)."""
    t0 = time.time()
    pre, post = config.get("clips.pre_sec"), config.get("clips.post_sec")
    if cand["trigger"] == "scan":      # scan windows tile the timeline exactly (no gaps between consecutive scans)
        every = config.get("analysis.scan_every_sec")
        pre, post = every * 0.6, every * 0.4
    ev = cand["event_t"]
    if cand.get("manual_range"):
        w0, w1 = cand["manual_range"]
    else:
        w0, w1 = ev - pre - 6, ev + post + 6
    segs = an.store.range(w0, w1)
    if not segs:
        log.warning("candidate %s: footage gone", cand["id"])
        return {"status": "error"}
    pcm, origin = _read_audio(segs)
    words = asr.transcribe(pcm, offset=origin)
    lex, tags = judge.lexicon_score([w for w in words if ev - 14 <= w["s"] <= ev + 10] or words)
    sig = cand["signals"]
    j = judge.judge([w for w in words if ev - pre <= w["s"] <= ev + post], ev, sig, {"streamer": an.stream.get("streamer"),
                    "title": an.stream.get("title")})
    llm_ran = j is not None
    speech = min(1.0, j["score"] + 0.15 * lex) if llm_ran else lex
    chat_on = an.chat_on or an.chat.total > 0
    scan = cand["trigger"] == "scan"
    score = speech if scan else scoring.final_score(sig["audio"], sig["visual"], sig["chat"], speech, chat_on)
    ok, why = scoring.decide(score, speech, llm_ran, scan)
    if cand["trigger"] == "manual":
        ok, why = True, "manual clip"
    start, end = scoring.snap_boundaries(words, ev, segs[0].t0, segs[-1].t1, pre, post)
    if cand.get("manual_range"):
        start, end = max(w0, segs[0].t0), min(w1, segs[-1].t1)
    transcript = " ".join(w["w"] for w in words if start <= w["s"] <= end)
    scores = {"audio": sig["audio"], "visual": sig["visual"], "chat": sig["chat"], "lexicon": round(lex, 2),
              "llm": round(j["score"], 2) if llm_ran else None, "speech": round(speech, 2), "llm_used": llm_ran,
              "tags": tags, "decision": why, "latency_sec": round(time.time() - t0, 1), "trigger": cand["trigger"],
              "tokens": j["tokens"] if llm_ran else 0}
    res = {"ok": ok, "score": score, "scores": scores, "start": start, "end": end, "words": words, "transcript": transcript,
           "category": (j or {}).get("category") or (tags[0] if tags else "none"),
           "reason": ((j or {}).get("reason") or ("Keywords: " + ", ".join(tags) if tags else "Loud/visual spike")) + ("" if llm_ran else " (LLM unavailable: keyword scoring)"),
           "title": (j or {}).get("title", ""), "status": "accepted" if ok else "filtered"}
    if evaluate_only:
        return res
    return _store_result(an, cand, res, segs)


def _store_result(an, cand, res, segs):
    sid = an.stream["id"]
    start, end = res["start"], res["end"]
    if not res["ok"]:
        db.x("INSERT INTO clips(stream_id,streamer,source_url,t_start,t_end,event_t,trigger,score,scores,category,reason,title,transcript,status,review,created)"
             " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
             (sid, an.stream.get("streamer"), an.stream["url"], start, end, cand["event_t"], cand["trigger"], res["score"],
              json.dumps(res["scores"]), res["category"], res["reason"], res["title"], res["transcript"], "filtered", "auto-rejected", time.time()))
        log.info("candidate %s filtered (score %.2f: %s)", cand["id"], res["score"], res["scores"]["decision"])
        return res
    # duplicates / merging against clips already made for this stream
    mg, ov = config.get("clips.merge_gap_sec"), config.get("clips.dedupe_overlap")
    words_set = _norm_words(res["words"])
    for ex in db.q("SELECT * FROM clips WHERE stream_id=? AND status IN ('ready','processing') AND review!='rejected'", (sid,)):
        close = scoring.overlap(start, end, ex["t_start"], ex["t_end"]) >= ov or (start - mg <= ex["t_end"] and ex["t_start"] - mg <= end)
        exw = set(re.findall(r"[a-z']{3,}", (ex["transcript"] or "").lower()))
        same_text = len(words_set & exw) / max(1, len(words_set | exw)) > 0.7 and len(words_set) > 8
        if close or same_text:
            return _merge(an, ex, res, segs, start, end)
    clip_id = db.x("INSERT INTO clips(stream_id,streamer,source_url,t_start,t_end,event_t,trigger,score,scores,category,reason,title,transcript,words,status,created)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (sid, an.stream.get("streamer"), an.stream["url"], start, end, cand["event_t"], cand["trigger"], res["score"],
                    json.dumps(res["scores"]), res["category"], res["reason"], res["title"], res["transcript"],
                    json.dumps(res["words"]), "processing", time.time()))
    _make_raw(an, clip_id, start, end)
    return res


def _make_raw(an, clip_id, start, end):
    folder = config.SETTINGS.data_dir / "clips" / str(clip_id)
    raw = folder / "raw.mp4"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        hq = getattr(an, "fetch_hq", None)
        done = False
        if hq:                                  # video mode: pull just this section in full quality
            try:
                hq(start, end, raw)
                done = True
            except Exception as e:  # noqa: BLE001 - fall back to the (lower quality) analysis copy
                log.warning("HD section download failed for clip %s (%s); using the analysis copy", clip_id, str(e)[-160:])
        if not done:
            render.cut_raw(an.store.range(start, end), start, end, raw)
        db.update("clips", clip_id, status="ready", raw_path=str(raw), t_start=start, t_end=end)
        log.info("clip %s ready (%.0fs)", clip_id, end - start)
    except Exception as e:  # noqa: BLE001
        db.update("clips", clip_id, status="error", error=str(e)[-300:])
        log.error("clip %s failed: %s", clip_id, e)
        return
    if config.get("codex.auto"):
        def _cx():
            from . import codex_judge
            try:
                codex_judge.review(clip_id)
            except Exception as e:  # noqa: BLE001
                log.warning("Codex review of clip %s failed: %s", clip_id, str(e)[-200:])
        threading.Thread(target=_cx, daemon=True).start()
    if config.get("export.auto_export", True):      # a failed export never invalidates the (playable) raw clip
        try:
            export(clip_id, config.get("export.layout"), "tiktok", config.get("export.captions"))
        except Exception as e:  # noqa: BLE001
            db.update("clips", clip_id, error="export: " + str(e)[-200:])
            log.error("auto-export of clip %s failed: %s", clip_id, e)


def _merge(an, ex, res, segs, start, end):
    """Merge a new highlight into an existing clip: union of the two spans (capped), best score wins."""
    hi = config.get("clips.max_len")
    n0, n1 = min(start, ex["t_start"]), max(end, ex["t_end"])
    better = res["score"] > ex["score"]
    if n1 - n0 > hi:     # too long: keep the stronger moment's span
        if not better:
            log.info("clip %s: duplicate of %s ignored", res["title"], ex["id"])
            return res
        n0, n1 = start, end
    if (n0, n1) == (ex["t_start"], ex["t_end"]) and not better:
        log.info("duplicate highlight merged into clip %s (no change)", ex["id"])
        return res
    words = json.loads(ex["words"] or "[]")
    seen = {(w["s"], w["w"]) for w in words}
    words += [w for w in res["words"] if (w["s"], w["w"]) not in seen]
    words.sort(key=lambda w: w["s"])
    db.update("clips", ex["id"], t_start=n0, t_end=n1, score=max(res["score"], ex["score"]),
              scores=res["scores"] if better else json.loads(ex["scores"]), reason=res["reason"] if better else ex["reason"],
              title=res["title"] if better and res["title"] else ex["title"], words=words,
              transcript=" ".join(w["w"] for w in words if n0 <= w["s"] <= n1), status="processing")
    log.info("merged highlight into clip %s -> %.0fs", ex["id"], n1 - n0)
    _make_raw(an, ex["id"], n0, n1)
    return res


def export(clip_id, layout="auto", platform="tiktok", captions=True):
    """Platform-ready export; identical requests reuse the existing file (no duplicate exports)."""
    clip = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not clip or not clip["raw_path"] or not Path(clip["raw_path"]).exists():
        raise ValueError("clip has no source file")
    key = f"{clip_id}|{layout}|{platform}|{int(bool(captions))}|{clip['t_start']:.1f}-{clip['t_end']:.1f}|{config.get('export.width')}"
    ex = db.one("SELECT * FROM exports WHERE key=?", (key,))
    if ex and Path(ex["path"]).exists():
        return ex
    out = config.SETTINGS.data_dir / "clips" / str(clip_id) / f"{platform}_{layout}_{'cap' if captions else 'nocap'}.mp4"
    words = json.loads(clip["words"] or "[]")
    info = render.export_clip(Path(clip["raw_path"]), words, clip["t_start"], clip["t_end"], out, layout, platform, captions)
    if ex:
        db.update("exports", ex["id"], path=str(out), created=time.time())
        return db.one("SELECT * FROM exports WHERE id=?", (ex["id"],))
    db.x("INSERT INTO exports(clip_id,key,layout,platform,path,created) VALUES(?,?,?,?,?,?)",
         (clip_id, key, info.get("layout_used", layout), platform, str(out), time.time()))
    log.info("exported clip %s -> %s (%s)", clip_id, out.name, info)
    return db.one("SELECT * FROM exports WHERE key=?", (key,))


# =============================================================== stream manager
class StreamRuntime:
    def __init__(self, stream, jobs: JobWorker):
        self.stream, self.jobs = stream, jobs
        self.cap = capture.Capture(stream, self._on_segment)
        self.an = Analyzer(stream, self.cap.store, lambda a, c: jobs.submit(a, c))
        self.segq: queue.Queue = queue.Queue(maxsize=200)
        self.chat_thread = None
        self.stop_ev = threading.Event()
        self.an_thread = None

    def _on_segment(self, seg):
        try:
            self.segq.put_nowait(seg)
        except queue.Full:
            log.warning("analysis is behind: dropping segment analysis (recording unaffected)")

    def _analysis_loop(self):
        while not self.stop_ev.is_set():
            try:
                seg = self.segq.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self.an.process_segment(seg)
            except Exception:  # noqa: BLE001
                log.exception("segment analysis failed")

    def start(self):
        self.stop_ev.clear()
        self.cap.start()
        self.an_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.an_thread.start()
        m = re.match(r"^https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", self.stream["url"])
        if m and config.get("scoring.chat_enabled"):
            self.chat_thread = threading.Thread(target=self._chat, args=(m.group(1).lower(),), daemon=True)
            self.chat_thread.start()

    def stop(self):
        self.stop_ev.set()
        self.cap.stop()

    def _chat(self, channel):
        """Anonymous, read-only Twitch chat (justinfan login): only message *counts* per second are kept."""
        while not self.stop_ev.is_set():
            try:
                raw = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=20)
                s = ssl.create_default_context().wrap_socket(raw, server_hostname="irc.chat.twitch.tv")
                s.sendall(f"NICK justinfan{int(time.time()) % 90000 + 10000}\r\nJOIN #{channel}\r\n".encode())
                s.settimeout(30)
                rest, self.an.chat_on = b"", True
                while not self.stop_ev.is_set():
                    try:
                        data = s.recv(8192)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    *lines, rest = (rest + data).split(b"\r\n")
                    for line in lines:
                        if line.startswith(b"PING"):
                            s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG #" in line:
                            self.an.chat.add(time.time() - 4)
                s.close()
            except Exception:  # noqa: BLE001
                pass
            self.an.chat_on = False
            self.stop_ev.wait(5)

    def status(self):
        c, a = self.cap, self.an
        return {"id": self.stream["id"], "state": c.state, "note": c.state_note, "platform": c.platform, "meta": c.meta,
                "buffer_sec": c.store.buffer_seconds(), "disk_mb": round(c.store.disk_bytes() / 2**20),
                "reconnects": c.reconnects, "since": c.started_at, "analysis": a.stats, "chat": a.chat_on,
                "pending": len(a.pending), "recent": [a.timeline[k] for k in list(a.timeline)[-60:]]}


class Manager:
    def __init__(self):
        self.jobs = JobWorker()
        self.rt: dict[int, StreamRuntime] = {}
        self.lock = threading.RLock()

    def add(self, url, name="", streamer=""):
        url = url.strip()
        if not (capture.PLATFORM_RE.match(url) or re.match(r"^(https?|rtmp)://", url) or Path(url).exists()):
            raise ValueError("Enter a Twitch, YouTube or Kick URL (or a direct HLS/RTMP URL).")
        plat = capture.platform_of(url)
        sid = db.x("INSERT INTO streams(url,name,streamer,platform,created) VALUES(?,?,?,?,?)",
                   (url, name or url, streamer or re.sub(r"^https?://[^/]+/@?", "", url).split("/")[0], plat, time.time()))
        return sid

    def remove(self, sid):
        self.stop(sid)
        db.x("DELETE FROM streams WHERE id=?", (sid,))

    def start(self, sid):
        with self.lock:
            if sid in self.rt and self.rt[sid].cap.state not in ("stopped", "idle"):
                return
            s = db.one("SELECT * FROM streams WHERE id=?", (sid,))
            if not s:
                raise ValueError("unknown stream")
            rt = StreamRuntime(s, self.jobs)
            self.rt[sid] = rt
            rt.start()
            db.update("streams", sid, auto_start=1)

    def stop(self, sid):
        with self.lock:
            rt = self.rt.get(sid)
            if rt:
                rt.stop()
            db.update("streams", sid, auto_start=0) if db.one("SELECT id FROM streams WHERE id=?", (sid,)) else None

    def resume_all(self):
        for s in db.q("SELECT * FROM streams WHERE auto_start=1 AND platform!='vod'"):
            try:
                self.start(s["id"])
            except Exception:  # noqa: BLE001
                log.exception("resume failed")

    def state(self):
        v = gpu.vram()
        streams = []
        for s in db.q("SELECT * FROM streams WHERE platform!='vod' ORDER BY id"):
            rt = self.rt.get(s["id"])
            streams.append({**s, "runtime": rt.status() if rt else {"state": "stopped"}})
        from . import vod
        return {"vods": [j.status() for j in sorted(vod.JOBS.values(), key=lambda j: -j.id)], "streams": streams, "gpu": {"used_mb": v[0], "total_mb": v[1], "util": v[2]} if v else None,
                "asr": asr.status(), "llm": judge.llm_status(), "queue": self.jobs.q.qsize(), "current_job": self.jobs.current,
                "jobs_done": self.jobs.done, "jobs_failed": self.jobs.failed}
