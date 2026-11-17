# Repository evaluation and reuse report

Method: each repository was cloned (shallow) and inspected: license file, README, dependency files, and targeted source
greps for live capture, paid-API use, Ollama/local-model support and GPU handling. **I did not install or run any of
them end-to-end**, so "works" below means "the code path exists in source", never "I measured it". The `gh` CLI was not
available, so star counts and commit history depth were not collected (shallow clones expose only the latest commit date).

| | PeakCut | DAWASSI AI Clipper | SharkyClips | LocalClip | Auto Video Editor | OBS AI Highlights |
|---|---|---|---|---|---|---|
| Accessible | yes | yes | yes | yes | yes | **no: GitHub 404 (repo not found)** |
| License | MIT | MIT | MIT | **AGPL-3.0** | MIT | n/a |
| Latest commit seen | 2026-11-15 | 2026-07-18 | 2026-11-16 | 2026-11-15 | 2026-10-26 | n/a |
| Stack | Python/FastAPI, single 2.4k-line HTML UI, SQLite or Postgres | Node.js (server.js 1.4k lines) + yt-dlp | Python backend + web frontend, 326 source files | Python + frontend, Docker option | Python, desktop GUI (file-based) | n/a |
| Live capture in source | **yes**: yt-dlp URL to ffmpeg segment muxer (stream copy), auto-reconnect, auto-record when channel goes live | **yes**: 4 s segment buffer via yt-dlp/ffmpeg, stall detection, up to 5 streams | VOD/file focused; has an `is_live` check in ingest and live-chat reader; no rolling live buffer found | No (VODs/clips; no live path found) | No (recorded files) |
| Highlight detection | loudness + chat-rate spikes (+60% over 3-min average) then transcript judged by an LLM, second "reaction" judge | **loudness only** (`volumedetect` per segment vs median) | loudness, laughter, reaction words, chat replay, game HUD (Delta Force, League of Legends) + LLM ranking | LLM reads the whole VOD transcript + chat | audio analysis, hype phrases, laughter, faces/motion |
| Paid API required | no: "no key" = loudness only; Ollama supported (OpenAI-compatible endpoint). Cloud providers optional | no (Whisper in browser, no LLM) | no: Ollama, Qwen3.5 4B/9B picked by VRAM | no with local model (Claude Code optional) | no (faster-whisper) |
| Local ASR | sherpa-onnx Parakeet (CPU default, CUDA optional) | Whisper in browser (~145 MB) | Whisper large-v3-turbo / medium int8 by VRAM | on-GPU transcription | faster-whisper / openai-whisper |
| 8 GB VRAM story | CPU-first ASR; LLM external | n/a | explicit tiers; "silent mode" yields VRAM to games | local model dependent | large-v3 picked on NVIDIA (heavy) |
| Vertical + captions | yes, word-by-word, 3 styles | yes (manual editor) | yes + bleeping, titles, posting | yes, facecam/gameplay layouts | yes, burned-in animated captions |
| Integration effort | medium-high (multi-user server, auth, posting, Postgres) | medium (different language) | high (large app, HUD detectors game-specific) | blocked by AGPL | medium (GUI-coupled) |

## Decision: Option D (custom hybrid), implemented as a new small codebase

* **PeakCut** has the right *live design* (segment-muxer stream copy so the recorder never waits for AI; loudness/chat
  spike watcher; ask the LLM only around a spike). But it is a multi-user hosted app (accounts, OAuth posting, admin,
  Postgres, usage metering) and uses sherpa-onnx rather than faster-whisper; carrying that would cost more than
  re-implementing the live core. Forking it (option A) would also leave a 2.4k-line single-file UI to restructure.
* **DAWASSI** (option B, part 1) has a good buffer, but its detection is loudness-only and it is a Node app, so mixing
  it with Python AI code adds a process boundary for little gain.
* **SharkyClips** (option B, part 2) is the closest in spirit for local gaming analysis, but it is VOD-oriented and its
  HUD detectors are game-specific (not Valorant/CS2/Fortnite). I reimplemented the generic parts only.
* **OBS AI Highlights** (option C) does not exist at the given URL, so it could not be evaluated.
* **LocalClip** is AGPL-3.0: copying any of its code would force the whole combined work (including network-served use)
  under AGPL. **No LocalClip code was used.**

## What was actually reused, and obligations

No source files were copied from any repository. The following *designs* were adapted and are credited in code comments:

| Source | Idea adopted | Obligation |
|---|---|---|
| PeakCut (MIT) | ffmpeg segment-muxer stream-copy rolling buffer; cut speech chunks at the quietest point before ASR; "ask the LLM only around a spike" | MIT requires keeping the copyright notice **if code is copied**; none was, but attribution is given in `clipper/capture.py` |
| DAWASSI (MIT) | segment-buffer + stall-watchdog concept | none (ideas only) |
| SharkyClips (MIT) | tiering ASR/LLM by available VRAM; Ollama auto-start | none (ideas only) |

Third-party packages installed from PyPI keep their own licenses: faster-whisper (MIT), CTranslate2 (MIT),
OpenCV (Apache-2.0), FastAPI/uvicorn (MIT/BSD), streamlink (BSD-2), yt-dlp (Unlicense), NVIDIA cuBLAS/cuDNN wheels
(NVIDIA proprietary redistribution terms; used locally, not redistributed by this project).
Models: Whisper (MIT weights), Qwen2.5-3B-Instruct (Qwen Research License for the 3B size: check terms for commercial use).

## Bugs and limitations observed in the evaluated projects (from source inspection only)

* PeakCut: ASR defaults to CPU; the "live" path clock-aligns audio and video by re-transcribing each clip, which hints at
  real drift (its own comment cites ~4 s/hour). This project avoids the problem by analysing the recorded pieces themselves.
* DAWASSI: highlight detection is volume only (cannot separate a shout from game explosions or music).
* SharkyClips: HUD detection only for two games; no rolling live buffer.
* LocalClip: no live support; AGPL.
