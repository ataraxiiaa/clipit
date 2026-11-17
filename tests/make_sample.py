"""Builds a SYNTHETIC labeled test stream (TTS speech + game-like noise + synthetic video).
It is not real gameplay: it is a smoke/regression benchmark with known ground truth, nothing more.
Usage: python tests/make_sample.py OUT_DIR  -> OUT_DIR/sample.mp4 and OUT_DIR/labels.json"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
OUT.mkdir(parents=True, exist_ok=True)
TOTAL = 780
rng = np.random.default_rng(7)


def tts(text, voice, name):
    p = OUT / f"{name}.wav"
    if not p.exists():
        ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
              f"$s.SelectVoice('{voice}'); $s.Rate=1; $s.SetOutputToWaveFile('{p.resolve()}'); $s.Speak('{text.replace("'", "''")}'); $s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    x = np.interp(np.arange(0, len(x) * SR / sr), np.arange(len(x)) * SR / sr, x) if sr != SR else x
    return x


def level(x, dbfs):
    r = np.sqrt(np.mean(x ** 2) + 1e-12)
    return x * (10 ** (dbfs / 20) / r)


Z, D = "Microsoft Zira Desktop", "Microsoft David Desktop"
calm = ["Okay so I'm going to rotate to the other side of the map now.", "Let me check my inventory real quick and see what I have.",
        "Alright chat, thanks for the follows, appreciate everybody being here today.", "I think we should buy armor this round.",
        "Hold on, let me open the settings and change my sensitivity.", "Okay going into the next match, let's see how this goes.",
        "So the plan is to push mid and then hold the corner.", "Yeah I'm just farming resources for a bit before we move on."]
bed = np.zeros(TOTAL * SR, dtype=np.float32)
t = np.arange(TOTAL * SR) / SR
bed += level(rng.standard_normal(len(t)).astype(np.float32), -42)          # game ambience
bed += 0.004 * np.sin(2 * np.pi * 110 * t) * (1 + np.sin(2 * np.pi * 0.2 * t))  # drone


def place(x, at):
    i = int(at * SR)
    bed[i:i + len(x)] += x[: max(0, len(bed) - i)]


labels = []
speech_level = -26
k = 0
at = 8.0
events = {  # start -> (kind, label, voice, text, loud)
    120: ("shout", True, Z, "Oh my god! No way! He just did that! That is insane! I can't believe it!", True),
    210: ("explosion", False, None, None, True),
    290: ("funny_quiet", True, D, "Dude, I just tripped over my own grenade and killed myself. I am literally crying right now. That is so dumb. Haha. Chat, please, never let me live this down.", False),
    380: ("loud_music_boring", False, Z, "Okay let me check my inventory and then open the menu to look at the map again.", False),
    470: ("clutch", True, D, "One versus four! Let's go! Let's go! Ace! Clutch! I cannot believe that actually worked!", True),
    560: ("scene_cut_boring", False, Z, "Alright the match is loading now, I will just wait a second here.", False),
    630: ("funny_story", True, D, "Chat, you will not believe this. I walked into the wall for like five minutes. Then my mom walked in and asked who I was yelling at. And I said nobody, it is just the wall. Haha, she did not even blink.", False),
    700: ("rage", True, Z, "What the hell! That is unplayable! The hitboxes are completely broken! Are you kidding me! I am so done with this game!", True),
}
busy = [(s, s + 28) for s in events]
while at < TOTAL - 20:
    if not any(a - 12 <= at <= b + 3 for a, b in busy):
        place(level(tts(calm[k % len(calm)], Z if k % 2 else D, f"calm{k % len(calm)}_{k % 2}"), speech_level), at)
        k += 1
    at += 19
for s, (kind, is_hl, voice, text, loud) in events.items():
    if kind == "explosion":
        n = rng.standard_normal(3 * SR).astype(np.float32) * np.exp(-np.arange(3 * SR) / SR * 1.2)
        place(level(n, -12) * 1.0, s)
        dur = 3
    else:
        x = tts(text, voice, kind)
        if loud:
            x = np.tanh(level(x, -14) * 2.2) * 0.9   # shouted: louder + harmonics
        else:
            x = level(x, speech_level)
        if kind == "loud_music_boring":
            m = 0.12 * np.sin(2 * np.pi * 220 * t[:25 * SR]) * np.sin(2 * np.pi * 2 * t[:25 * SR]) + 0.08 * np.sin(2 * np.pi * 330 * t[:25 * SR])
            place(m.astype(np.float32), s - 6)
        place(x, s)
        dur = len(x) / SR
    labels.append({"kind": kind, "highlight": is_hl, "start": s, "end": round(s + dur, 1)})

bed = np.clip(bed, -1, 1)
wavp = OUT / "mix.wav"
with wave.open(str(wavp), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((bed * 32767).astype(np.int16).tobytes())
cut_start = 560
vf = (f"[0:v][1:v]overlay=enable='between(t,{cut_start + 4},{cut_start + 22})'[v]")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={TOTAL}",
                "-f", "lavfi", "-i", f"smptebars=size=1280x720:rate=30", "-i", str(wavp), "-filter_complex", vf, "-map", "[v]",
                "-map", "2:a", "-t", str(TOTAL), "-c:v", "libx264", "-preset", "veryfast", "-g", "60", "-c:a", "aac",
                str(OUT / "sample.mp4")], check=True)
(OUT / "labels.json").write_text(json.dumps(labels, indent=1))
print("wrote", OUT / "sample.mp4", len(labels), "labeled events")
