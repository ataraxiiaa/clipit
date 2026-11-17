"""Fast unit tests (no GPU, no network): python -m pytest tests/test_units.py"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import asr, capture, judge, render, scoring, signals  # noqa: E402


def test_overlap_and_boundaries():
    assert scoring.overlap(0, 10, 5, 15) == 0.5
    words = [{"w": "a", "s": 10.0, "e": 10.3}, {"w": "b", "s": 10.35, "e": 10.6}, {"w": "c", "s": 12.0, "e": 12.4}]
    s, e = scoring.snap_boundaries(words, 20, 0, 100)
    assert 20 <= e - s <= 60 and s >= 0


def test_boundary_never_exceeds_available_footage():
    s, e = scoring.snap_boundaries([], 5, 0, 18)
    assert s >= 0 and e <= 18


def test_lexicon():
    hi, tags = judge.lexicon_score([{"w": w} for w in "oh my god no way that was insane let's go".split()])
    lo, _ = judge.lexicon_score([{"w": w} for w in "i will open the menu now".split()])
    assert hi > 0.5 > lo and "surprise" in tags


def test_chunks_cover_audio_and_cut_in_quiet():
    x = np.random.default_rng(1).standard_normal(16000 * 70).astype(np.float32) * 0.2
    x[16000 * 17:16000 * 18] = 0.0      # a quiet second between 13 and 20 s
    ch = asr._chunks(x)
    assert ch[0][0] == 0 and ch[-1][1] == len(x) and all(a < b for a, b in ch)
    assert 16000 * 17 <= ch[0][1] <= 16000 * 18


def test_audio_scorer_flags_shout_not_talk():
    sc = signals.AudioScorer(180)
    rng = np.random.default_rng(0)
    for i in range(120):                 # ambience + regular talking around -25 dB
        talk = (i % 7) < 3
        sc.score({"db": -25 + rng.normal(0, 1.5) if talk else -44 + rng.normal(0, .5), "hf": .1, "voice": .6})
    calm = sc.score({"db": -25, "hf": .1, "voice": .6})["score"]
    loud = sc.score({"db": -11, "hf": .15, "voice": .6})["score"]
    assert calm < 0.3 < 0.6 < loud


def test_segment_store_pins_and_prunes():
    d = Path(tempfile.mkdtemp())
    st = capture.SegmentStore(d)
    for i in range(30):
        p = d / f"s{i}.ts"
        p.write_bytes(b"x" * 100)
        st.add(p, 1000 + i * 4, 4.0)
    st.pin("c", 1000, 1010)
    st.prune(keep_sec=40, max_bytes=10**9)
    names = {s.path.name for s in st.segs}
    assert "s0.ts" in names and "s1.ts" in names and "s29.ts" in names     # pinned + newest kept
    assert "s5.ts" not in names                                            # old and unpinned: removed
    st.unpin("c")
    st.prune(keep_sec=40, max_bytes=10**9)
    assert "s0.ts" not in {s.path.name for s in st.segs}


def test_segment_timeline_gap_resets():
    d = Path(tempfile.mkdtemp())
    st = capture.SegmentStore(d)
    a = st.add(d / "a.ts", 100.0, 4.0)
    b = st.add(d / "b.ts", 104.2, 4.0)     # small jitter: stays continuous
    c = st.add(d / "c.ts", 200.0, 4.0)     # reconnect gap: restarts at the wall-clock label
    assert abs(b.t0 - a.t1) < 1e-9 and c.t0 == 200.0


def test_ass_captions_and_chat_spike():
    out = Path(tempfile.mkdtemp()) / "s.ass"
    n = render.make_ass([{"w": "hello", "s": 1.0, "e": 1.4, "p": .9}, {"w": "there.", "s": 1.5, "e": 2.0, "p": .9}], 0, 10, 500, out)
    assert n == 1 and "\\kf" in out.read_text()
    ch = signals.ChatCounter()
    for t in range(0, 200):
        ch.add(1000 + t)                       # 1 msg/s baseline
    for t in range(200, 210):
        for _ in range(8):
            ch.add(1000 + t)                   # burst
    assert ch.score(1209) > 0.6 > ch.score(1150)


def test_platform_detection():
    assert capture.platform_of("https://www.twitch.tv/x") == "twitch"
    assert capture.platform_of("https://kick.com/x") == "kick"
    assert capture.platform_of("https://youtube.com/@x/live") == "youtube"
    assert capture.platform_of("rtmp://host/app") == "direct"
