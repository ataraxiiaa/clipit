# Local AI Livestream Clipper

Fully local livestream clipper: records an authorized Twitch / YouTube / Kick stream into a rolling buffer, finds highlights
with audio + speech + visual (+ Twitch chat) signals, judges them with a local LLM, and exports captioned 9:16 clips.
No paid APIs; inference runs on your GPU through faster-whisper and Ollama. Tuned for an RTX 4060 Ti (8 GB).

## Install (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```
This installs (if missing) uv, ffmpeg and Ollama via winget, creates `.venv`, installs `requirements.txt`, pulls
`qwen2.5:3b-instruct` (~1.9 GB) and downloads Whisper `small` (~480 MB). After that everything works offline except
reaching the stream itself. Manual equivalents: `ollama pull qwen2.5:3b-instruct`; the Whisper model downloads on first use.

## Run

```
start.bat            # starts the server and opens http://127.0.0.1:8780
```
1. Paste an **authorized** channel URL (`twitch.tv/name`, `youtube.com/@name/live`, `kick.com/name`) and press *Add & start*.
2. The app waits until the channel is live (polls every 30 s), records, and shows buffer, analysis score and AI queue.
3. Clips appear on the right with score, category, explanation, per-signal breakdown and transcript. Accept / Reject, or
   *Export* with a layout and platform. *Clip last 45s* makes a manual clip from the buffer.
4. Thresholds, clip lengths, weights and export settings are editable in the dashboard (saved to `data/settings.json`).

### Clip an existing video (YouTube link or file)
Use the **Clip a video** card: paste a YouTube link (or any link yt-dlp supports, or a local file path) and press *Find clips*.
The video is downloaded (up to 1080p, `vod.max_height`), then analysed at full speed through the same pipeline as a live
stream. A 13-minute test file took about 3 minutes (sharing the GPU with a live recording); a progress bar shows the state.
Clips land in the same list, with their position shown as "m:ss in video". Live streams must be added as streams instead.
Source videos are deleted after processing (`vod.keep_source: true` keeps them). Videos over 6 h are refused (`vod.max_hours`).
Direct HLS/RTMP URLs also work as live sources.

## How it works

```
streamlink -> ffmpeg (stream copy, 4 s .ts pieces)  ->  rolling buffer (pinned while a clip is pending, disk-capped)
                     |
   per finished piece (CPU only): audio loudness/vocal-intensity + visual cut/motion + chat rate  -> cheap score
                     |  score >= candidate_threshold  (or periodic conversation scan of speech-active stretches)
   single GPU worker (serialized): faster-whisper small (word timestamps) -> keyword score -> Qwen2.5-3B judge (JSON)
                     |  blended score + decision + explanation
   boundaries snapped to pauses, merge/dedupe, accurate cut (NVENC) -> raw.mp4 -> vertical export (captions, layouts)
```
Key design points: the recorder never waits for AI; the GPU lock guarantees Whisper and the LLM never run simultaneously;
Whisper unloads after 5 idle minutes; CUDA failure falls back to CPU int8; if Ollama is missing the keyword fallback is used
(with a higher acceptance bar and an explicit flag on the clip); analysis falling behind drops analysis, never recording.

Layouts: `auto` (facecam split if a small face is detected, else blur-fill), `blur` (whole frame visible over blurred
background), `crop` (9:16 window centred on the motion-weighted action), `split` (facecam on top), `landscape` (16:9).
Captions are ASS with the active word highlighted; bottom margins per platform are approximate UI-safe values.

## Layout of the project

`clipper/` code · `web/index.html` dashboard · `config.yaml` all settings with comments · `tests/` sample generator,
evaluation, unit tests · `docs/` repository report, test report, limitations · `data/` (git-ignored) db, buffers, clips, logs.

## Responsible use

Only record and repurpose streams you have permission to use. The app does not bypass authentication, DRM or paywalls,
reads Twitch chat anonymously (read-only, counts only), and never posts anything automatically.
