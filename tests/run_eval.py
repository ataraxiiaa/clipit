"""Offline evaluation of detection on a labeled sample: audio-only vs audio+visual+keywords vs full (Whisper + Qwen).
Usage: python tests/run_eval.py data/sample  (needs sample.mp4 + labels.json)"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import asr, capture, config, gpu, pipeline, scoring, signals, util  # noqa: E402

util.setup_logging()
src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
labels = json.loads((src / "labels.json").read_text())
tmp = Path(tempfile.mkdtemp(prefix="eval_"))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src / "sample.mp4"), "-c", "copy", "-f", "segment", "-segment_time", "4",
                "-reset_timestamps", "1", str(tmp / "s_%05d.ts")], check=True)
store = capture.SegmentStore(tmp)
cands = []
an = pipeline.Analyzer({"id": 0, "url": "eval", "streamer": "test"}, store, lambda a, c: cands.append(c))
results, audio_only, t_start = [], [], time.time()
clock, last_aud = 0.0, -1e9
peak_vram = 0
for p in sorted(tmp.glob("s_*.ts")):
    d = util.probe_duration(p)
    seg = store.add(p, clock, d)
    clock += d
    an.process_segment(seg)
    for c in list(cands):
        cands.remove(c)
        t0 = time.time()
        r = pipeline.process_candidate(an, c, evaluate_only=True)
        r["cand"] = c
        r["secs"] = time.time() - t0
        results.append(r)
        v = gpu.vram()
        peak_vram = max(peak_vram, v[0] if v else 0)
# audio-only baseline: any second whose audio score passes the same threshold, with the same cooldown
thr = config.get("scoring.candidate_threshold")
for t, e in an.timeline.items():
    if t - an.t_first >= config.get("analysis.warmup_sec") and e["a"] >= thr and t - last_aud >= config.get("scoring.cooldown_sec"):
        audio_only.append(t)
        last_aud = t


def match(t):
    for L in labels:
        if L["start"] - 5 <= t <= L["end"] + 12:
            return L
    return None


def merged(ivs):
    """Same rule as the pipeline: nearby accepted clips merge while the union stays under max_len."""
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1] + config.get("clips.merge_gap_sec") and b - out[-1][0] <= config.get("clips.max_len"):
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def match_iv(a, b):
    for L in labels:
        ov = min(b, L["end"]) - max(a, L["start"])
        if ov >= 0.5 * (L["end"] - L["start"]):
            return L
    return None


def report(name, dets):
    hit = {}
    fp = []
    for t in dets:
        L = match_iv(*t) if isinstance(t, tuple) else match(t)
        if L and L["highlight"]:
            hit.setdefault(L["kind"], t)
        else:
            fp.append((round(t[0] if isinstance(t, tuple) else t), L["kind"] if L else "ordinary"))
    n_hl = sum(L["highlight"] for L in labels)
    prec = len(hit) / max(1, len(hit) + len(fp))
    print(f"{name:34s} detections={len(dets):2d}  TP={len(hit)}/{n_hl} recall={len(hit)/n_hl:.2f} precision={prec:.2f}  FP={fp}  missed={[L['kind'] for L in labels if L['highlight'] and L['kind'] not in hit]}")
    return {"name": name, "detections": len(dets), "tp": len(hit), "highlights": n_hl, "recall": len(hit) / n_hl, "precision": prec, "false_positives": fp,
            "missed": [L["kind"] for L in labels if L["highlight"] and L["kind"] not in hit]}


print(f"\nCandidates produced by cheap signals: {len(results)}; eval wall time {time.time()-t_start:.0f}s for {clock:.0f}s of stream")
summary = [report("A. audio-only (loudness)", audio_only)]
nollm = []
for r in results:
    s = r["scores"]
    scan = r["cand"]["trigger"] == "scan"
    sc = s["lexicon"] if scan else scoring.final_score(s["audio"], s["visual"], s["chat"], s["lexicon"], False)
    ok, _ = scoring.decide(sc, s["lexicon"], False, scan)
    if ok:
        nollm.append((r["start"], r["end"]))
summary.append(report("B. audio+visual+keywords (no LLM)", nollm))
nollm = merged(nollm)
full = merged([(r["start"], r["end"]) for r in results if r["ok"]])
summary.append(report("C. full (Whisper+Qwen2.5-3B blend)", full))
print("\nPer-candidate detail:")
cover = []
for r in results:
    t = r["cand"]["event_t"]
    L = match(t)
    s = r["scores"]
    print(f"  t={t:6.0f} trig={r['cand']['trigger']:6s} label={(L or {}).get('kind','-'):18s} a={s['audio']:.2f} v={s['visual']:.2f} lex={s['lexicon']:.2f} llm={s['llm']} "
          f"final={r['score']:.2f} -> {r['status']:8s} [{r['start']:.0f}-{r['end']:.0f}] {r['category']}: {r['reason'][:70]} ({r['secs']:.1f}s)")
    if r["ok"] and L and L["highlight"]:
        cover.append({"kind": L["kind"], "clip": [r["start"], r["end"]], "label": [L["start"], L["end"]],
                      "covers_label": r["start"] <= L["start"] + 0.5 and r["end"] >= L["end"] - 0.5, "len": round(r["end"] - r["start"], 1)})
print("\nClip timing (does the clip contain the whole labeled moment?):")
for c in cover:
    print("  ", c)
lat = [r["secs"] for r in results]
print(f"\nProcessing latency per candidate (ASR+LLM): mean {sum(lat)/max(1,len(lat)):.1f}s max {max(lat or [0]):.1f}s; peak VRAM used {peak_vram} MB")
Path(src / "eval_result.json").write_text(json.dumps({"summary": summary, "clips": cover, "latency": lat, "peak_vram_mb": peak_vram,
                                                       "stream_sec": clock}, indent=1, default=str))
