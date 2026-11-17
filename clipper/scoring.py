"""Blends signals into scores and decides clip boundaries. Pure functions: easy to test."""
from . import config


def cheap_score(a: float, v: float, c: float, chat_on: bool) -> float:
    """Candidate gate from signals that need no GPU. Weighted mean over the signals that exist."""
    w = config.get("scoring.weights")
    parts = [(w["audio"], a), (w["visual"], v)] + ([(w["chat"], c)] if chat_on else [])
    return sum(x * y for x, y in parts) / sum(x for x, _ in parts)


def final_score(a, v, c, speech, chat_on):
    """Blend of all four signals. `speech` None = language models unavailable: weight redistributed, flagged."""
    w = config.get("scoring.weights")
    parts = [(w["audio"], a), (w["visual"], v)]
    if chat_on:
        parts.append((w["chat"], c))
    if speech is not None:
        parts.append((w["speech"], speech))
    return sum(x * y for x, y in parts) / sum(x for x, _ in parts)


def decide(score, speech, llm_ran, scan=False):
    acc = config.get("scoring.accept_threshold")
    if scan:   # found by reading speech alone: no loudness/visual corroboration, so the LLM verdict is mandatory and stricter
        if not llm_ran:
            return False, "scan candidates need the language model"
        return (speech >= config.get("scoring.scan_accept_threshold"), "scan: speech score " + ("ok" if speech >= config.get("scoring.scan_accept_threshold") else "below scan threshold"))
    if llm_ran and speech is not None and speech < config.get("scoring.min_speech_score"):
        return False, "speech/semantic score too low"
    # without speech understanding, demand a higher bar so loud noise alone is rarely accepted
    if score < (acc if llm_ran else acc + 0.1):
        return False, "blended score below threshold"
    return True, "accepted"


def snap_boundaries(words, event_t, avail_start, avail_end, pre=None, post=None):
    """Clip [start, end] around event_t: pre/post padding, snapped to pauses so no sentence is cut mid-word."""
    pre = config.get("clips.pre_sec") if pre is None else pre
    post = config.get("clips.post_sec") if post is None else post
    lo, hi = config.get("clips.min_len"), config.get("clips.max_len")
    start, end = event_t - pre, event_t + post
    if words:
        gaps = [(words[i]["e"], words[i + 1]["s"]) for i in range(len(words) - 1) if words[i + 1]["s"] - words[i]["e"] >= 0.35]
        before = [(a + b) / 2 for a, b in gaps if start - 4 <= (a + b) / 2 <= start + 1.5]
        after = [(a + b) / 2 for a, b in gaps if end - 1 <= (a + b) / 2 <= end + 4.5]
        if before:
            start = min(before, key=lambda g: abs(g - start))
        else:   # starting inside a word run: move to just before the word that contains `start`
            for w in words:
                if w["s"] <= start <= w["e"]:
                    start = w["s"] - 0.15
        if after:
            end = min(after, key=lambda g: abs(g - end))
        else:
            for w in words:
                if w["s"] <= end <= w["e"]:
                    end = w["e"] + 0.2
    start, end = max(start, avail_start), min(end, avail_end)
    if end - start > hi:     # keep the payoff, trim the lead-in
        start = end - hi
    if end - start < lo:
        extra = lo - (end - start)
        start = max(avail_start, start - extra * 0.6)
        end = min(avail_end, end + extra * 0.4)
    return round(start, 2), round(end, 2)


def overlap(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-6, min(a1 - a0, b1 - b0))
