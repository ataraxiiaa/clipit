"""Diagnose caption timing: re-transcribe each clip's own audio and compare with the word times stored for captions."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import asr, db, signals  # noqa: E402

ids = [int(x) for x in sys.argv[1:]]
key = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
for cid in ids:
    c = db.one("SELECT * FROM clips WHERE id=?", (cid,))
    stored = json.loads(c["words"] or "[]")
    t0, t1 = c["t_start"], c["t_end"]
    pcm = signals.decode_audio(c["raw_path"])
    real = asr.transcribe(pcm)                       # times relative to the clip's own audio
    rel = [(key(w["w"]), w["s"] - t0) for w in stored if t0 <= w["s"] <= t1 and len(key(w["w"])) >= 3]
    diffs = []
    for r in real:
        k = key(r["w"])
        if len(k) < 3:
            continue
        cand = [s for kk, s in rel if kk == k and abs(s - r["s"]) < 15]
        if cand:
            diffs.append(min(cand, key=lambda s: abs(s - r["s"])) - r["s"])   # +ve = caption is LATE vs real audio
    if len(diffs) >= 5:
        d = np.array(diffs)
        print(f"clip {cid}: dur={t1 - t0:.1f}s matched={len(d)} caption_offset median={np.median(d):+.2f}s  p10={np.percentile(d, 10):+.2f} p90={np.percentile(d, 90):+.2f}")
    else:
        print(f"clip {cid}: only {len(diffs)} matched words (stored {len(rel)}, real {len(real)})")
