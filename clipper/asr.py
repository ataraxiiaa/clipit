"""faster-whisper wrapper: lazy load, VRAM-aware device choice, CPU fallback, unload when idle."""
import logging
import threading
import time

import numpy as np

from . import config, gpu

log = logging.getLogger("clipper.asr")
_model = None
_info = {"device": None, "loaded_at": None}
_last_used = 0.0
_lock = threading.RLock()


def _load(force_cpu=False):
    global _model
    from faster_whisper import WhisperModel
    gpu.setup_cuda_dlls()
    name = config.get("asr.model")
    want = config.get("asr.device")
    use_cuda = not force_cpu and want != "cpu" and (want == "cuda" or gpu.free_mb() >= config.get("asr.min_free_vram_mb"))
    if use_cuda:
        try:
            _model = WhisperModel(name, device="cuda", compute_type=config.get("asr.compute_type"))
            _info.update(device="cuda", loaded_at=time.time())
            log.info("Whisper %s loaded on GPU", name)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("GPU load failed (%s); using CPU", str(e)[:120])
    _model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=6)
    _info.update(device="cpu", loaded_at=time.time())
    log.info("Whisper %s loaded on CPU (int8)", name)


def _chunks(pcm: np.ndarray, sr=16000, target=20.0, lo=13.0):
    """Split at the quietest 0.25 s between `lo` and `target` seconds, so no word is cut and Whisper never sees a long,
    mostly-quiet window (measured: a 50 s window dropped a shouted sentence that a 20 s window transcribed fine)."""
    out, i, n = [], 0, len(pcm)
    while n - i > target * sr * 1.1:
        a, b = i + int(lo * sr), i + int(target * sr)
        hop = sr // 4
        frames = [(float(np.mean(np.abs(pcm[k:k + hop]))), k) for k in range(a, b, hop)]
        cut = min(frames)[1] + hop // 2
        out.append((i, cut))
        i = cut
    out.append((i, n))
    return out


def transcribe(pcm: np.ndarray, offset: float = 0.0) -> list[dict]:
    """Returns words [{w, s, e, p}] with times shifted by `offset` seconds."""
    words = []
    for a, b in _chunks(pcm):
        words += _transcribe_chunk(pcm[a:b], offset + a / 16000)
    return words


def _transcribe_chunk(pcm: np.ndarray, offset: float) -> list[dict]:
    global _last_used
    with gpu.GPU_LOCK, _lock:
        if _model is None:
            _load()
        for attempt in (0, 1):
            try:
                segs, _ = _model.transcribe(pcm, language=config.get("asr.language"), word_timestamps=True, beam_size=1,
                                            vad_filter=True, condition_on_previous_text=False,
                                            vad_parameters={"min_silence_duration_ms": 400})
                words = [{"w": w.word.strip(), "s": round(offset + w.start, 2), "e": round(offset + w.end, 2),
                          "p": round(w.probability, 2)} for sg in segs for w in (sg.words or []) if w.word.strip()]
                _last_used = time.time()
                return words
            except Exception as e:  # noqa: BLE001 - typically CUDA OOM: drop to CPU and retry once
                log.warning("transcription failed (%s)%s", str(e)[:150], "; retrying on CPU" if attempt == 0 else "")
                if attempt == 1:
                    raise
                unload()
                _load(force_cpu=True)
    return []


def unload():
    global _model
    with _lock:
        _model = None
    import gc
    gc.collect()


def idle_check():
    if _model is not None and _info["device"] == "cuda" and time.time() - _last_used > config.get("asr.unload_after_sec"):
        with gpu.GPU_LOCK:
            log.info("Whisper idle: unloading to free VRAM")
            unload()


def status():
    return {"loaded": _model is not None, **_info}
