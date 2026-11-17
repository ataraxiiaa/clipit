"""Cutting, captions (ASS, word-highlight), layouts (blur-fill / action crop / facecam split / landscape), NVENC."""
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import config, util

log = logging.getLogger("clipper.render")


# ------------------------------------------------------------------ source cutting
def cut_raw(segs, t0, t1, out: Path):
    """Accurate cut from buffer pieces into a high-quality landscape MP4 (re-encoded so the cut is frame-exact)."""
    if not segs:
        raise RuntimeError("footage is no longer in the buffer")
    tmp = Path(tempfile.mkdtemp(prefix="clp_"))
    try:
        lst = tmp / "l.txt"
        lst.write_text("".join(f"file '{s.path.as_posix()}'\n" for s in segs), encoding="utf-8")
        ss = max(0.0, t0 - segs[0].t0)
        base = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{ss:.3f}", "-t", f"{t1 - t0:.3f}"]
        _encode(base, [], out, cq=19, audio=True, full_fps=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _venc(cq, nvenc):
    if nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(cq - 2), "-pix_fmt", "yuv420p"]


def _encode(pre_args, vf_args, out: Path, cq=None, audio=True, full_fps=False, cwd=None):
    cq = cq or config.get("export.cq")
    last = None
    for nv in ([True, False] if config.get("export.nvenc") else [False]):
        cmd = pre_args + vf_args + _venc(cq, nv) + (["-r", str(config.get("export.fps"))] if not full_fps else []) + \
              (["-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]) + ["-movflags", "+faststart", str(out)]
        try:
            util.run(cmd, cwd=cwd, timeout=900)
            return nv
        except RuntimeError as e:
            last = e
            log.warning("encode failed with %s: %s", "NVENC" if nv else "libx264", str(e)[:200])
    raise last


# ------------------------------------------------------------------ captions
def _ass_color(hexstr):
    h = hexstr.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()   # ASS is BGR


def _t(x):
    x = max(0.0, x)
    return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"


def make_ass(words, clip_start, clip_end, margin_bottom, out: Path, w=1080, h=1920):
    cap = config.get("export.caption")
    n = int(cap["words_per_line"])
    ws = [x for x in words if x["e"] > clip_start and x["s"] < clip_end and x.get("p", 1) >= 0.2]
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
            "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"Style: Cap,{cap['font']},{cap['size']},{_ass_color(cap['primary'])},{_ass_color(cap['secondary'])},"
            f"{_ass_color(cap['outline'])},&H64000000,0,0,0,0,100,100,0,0,1,{cap['outline_px']},2,2,60,60,{margin_bottom},1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines, cur = [], []
    for i, x in enumerate(ws):
        cur.append(x)
        gap = (ws[i + 1]["s"] - x["e"]) if i + 1 < len(ws) else 9
        if len(cur) >= n or gap > 0.5 or x["w"].endswith((".", "?", "!")):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    ev = []
    for g in lines:
        a, b = g[0]["s"] - clip_start, g[-1]["e"] - clip_start + 0.15
        text = ""
        for x in g:
            cs = max(1, round((x["e"] - x["s"]) * 100))
            word = x["w"].upper() if cap["uppercase"] else x["w"]
            text += f"{{\\kf{cs}}}{word.replace('{', '(').replace('}', ')')} "
        ev.append(f"Dialogue: 0,{_t(a)},{_t(b)},Cap,,0,0,0,,{text.strip()}")
    out.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")
    return len(ev)


# ------------------------------------------------------------------ analysis helpers for layouts
def _sample_frames(video: Path, n=6):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames, W, H


def detect_facecam(frames, W, H):
    """Haar face detection on a few frames. A small face = facecam overlay. Returns (x, y, w, h) in source px or None."""
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    scale = 960 / W
    boxes = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (960, int(H * scale))), cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in casc.detectMultiScale(g, 1.15, 6, minSize=(36, 36)):
            area = (w * h) / (g.shape[0] * g.shape[1])
            fx, fy = (x + w / 2) / g.shape[1], (y + h / 2) / g.shape[0]
            in_corner = (fx < 0.3 or fx > 0.7) and (fy < 0.35 or fy > 0.65)   # overlays sit in a corner; a centred face is the main shot
            if 0.002 < area < 0.06 and in_corner:
                boxes.append((x / scale, y / scale, w / scale, h / scale))
    if len(boxes) < max(2, len(frames) // 3):
        return None
    cx, cy, bw = np.median([b[0] + b[2] / 2 for b in boxes]), np.median([b[1] + b[3] / 2 for b in boxes]), np.median([b[2] for b in boxes])
    cw = min(W, bw * 3.0)
    ch = cw * 640 / 1080
    x0 = int(min(max(0, cx - cw / 2), W - cw)) // 2 * 2
    y0 = int(min(max(0, cy - ch / 2), H - ch)) // 2 * 2
    return x0, y0, int(cw) // 2 * 2, int(ch) // 2 * 2


def action_focus_x(frames, W):
    """Motion-weighted horizontal centre (0..1) for the action-aware 9:16 crop."""
    if len(frames) < 2:
        return 0.5
    acc = np.zeros(W // 8, dtype=np.float64)
    prev = None
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (W // 8, max(1, f.shape[0] // 8))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            acc += cv2.absdiff(g, prev).sum(axis=0)
        prev = g
    if acc.sum() < 1:
        return 0.5
    return float((acc * np.arange(len(acc))).sum() / acc.sum() / len(acc))


# ------------------------------------------------------------------ export
def export_clip(raw: Path, words, clip_start, clip_end, out: Path, layout="auto", platform="tiktok", captions=True):
    """Make a platform-ready file from the raw landscape clip. Returns info dict."""
    ex = config.get("export")
    W_o, H_o = ex["width"], ex["height"]
    frames, W, H = _sample_frames(raw)
    info = {"layout": layout}
    if layout == "landscape":
        W_o, H_o = 1920, 1080
    tmp = Path(tempfile.mkdtemp(prefix="exp_"))
    try:
        shutil.copy(raw, tmp / "in.mp4")
        sub = ""
        if captions and words:
            margin = ex["platforms"].get(platform, ex["caption"]["margin_bottom"]) if layout != "landscape" else 60
            if layout == "landscape":
                make_ass(words, clip_start, clip_end, 70, tmp / "s.ass", 1920, 1080)
            else:
                make_ass(words, clip_start, clip_end, margin, tmp / "s.ass", W_o, H_o)
            (tmp / "fonts").mkdir()
            for f in ("ariblk.ttf", "impact.ttf", "arialbd.ttf", "arial.ttf", "segoeuib.ttf"):
                fp = Path("C:/Windows/Fonts") / f
                if fp.exists():
                    shutil.copy(fp, tmp / "fonts" / f)
            sub = ",ass=s.ass:fontsdir=fonts"      # relative path (ffmpeg runs with cwd=tmp): no drive-letter escaping
        if layout == "auto":
            cam = detect_facecam(frames, W, H)
            layout = "split" if cam else "blur"
            info["facecam"] = cam
        else:
            cam = detect_facecam(frames, W, H) if layout == "split" else None
            if layout == "split" and not cam:
                layout = "blur"
        info["layout_used"] = layout
        if layout == "landscape":
            fc = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2{sub}[v]"
        elif layout == "crop":
            fx = action_focus_x(frames, W)
            cw = int(H * 9 / 16) // 2 * 2
            x0 = int(min(max(0, fx * W - cw / 2), W - cw)) // 2 * 2
            info["focus_x"] = round(fx, 2)
            fc = f"[0:v]crop={cw}:{H}:{x0}:0,scale={W_o}:{H_o}:flags=lanczos{sub}[v]"
        elif layout == "split":
            cx, cy, cw, ch = cam
            gh = H_o - 640
            fc = (f"[0:v]split=3[a][b][c];[a]crop={cw}:{ch}:{cx}:{cy},scale={W_o}:640:flags=lanczos[cam];"
                  f"[b]scale={W_o}:{gh}:force_original_aspect_ratio=increase,crop={W_o}:{gh},boxblur=30:5[bg];"
                  f"[c]scale={W_o}:-2:flags=lanczos[fg];[bg][fg]overlay=0:(H-h)/2[game];[cam][game]vstack{sub}[v]")
        else:   # blur fill: whole frame stays visible, nothing important is cropped away
            fc = (f"[0:v]split=2[b][c];[b]scale={W_o}:{H_o}:force_original_aspect_ratio=increase,crop={W_o}:{H_o},boxblur=30:5[bg];"
                  f"[c]scale={W_o}:-2:flags=lanczos[fg];[bg][fg]overlay=0:(H-h)/2{sub}[v]")
        pre = ["ffmpeg", "-y", "-v", "error", "-i", "in.mp4", "-filter_complex", fc, "-map", "[v]", "-map", "0:a?"]
        nv = _encode(pre, [], tmp / "out.mp4", cwd=tmp)
        info["encoder"] = "h264_nvenc" if nv else "libx264"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / "out.mp4"), out)
        return info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
