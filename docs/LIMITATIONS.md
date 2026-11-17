# Remaining limitations and future work

## Not verified (could not be tested in this environment)
* **No real Twitch / YouTube / Kick live stream was recorded.** No stream URL or permission was supplied, so streamlink
  capture was verified only for *offline detection* (a non-existent Twitch channel is correctly reported as offline). The
  recording/segmenting/reconnect pipeline was verified with a local file replayed in real time through the identical
  ffmpeg segment muxer. Platform-specific behaviour (Twitch ads/pre-roll, YouTube throttling, Kick HLS quirks, streamlink
  plugin breakage) is untested. YouTube in particular frequently changes and may need `yt-dlp`/`streamlink` upgrades.
* **Twitch chat spikes** are implemented and unit-tested on synthetic counts, but never ran against a real chat.
* **Facecam detection** (OpenCV Haar cascade) found no face in the synthetic sample (correct) but was **never tested on a
  real facecam**. Haar is weak with unusual angles, masks, or stylised cams; a better detector (YuNet/MediaPipe) is future work.
* **Accuracy numbers come from a small synthetic set** (TTS speech + generated noise/video, 5 highlights, 3 decoys) that
  the pipeline was tuned against, so they demonstrate that the machinery works, not real-world precision/recall.
  A real labelled set of actual streams is the single most valuable next step.

## Known weaknesses
* Highlight quality of a 3B model is modest: it rates plausible-sounding chatter generously and is blind to what is
  happening on screen. It cannot judge pure gameplay highlights (clutch plays with no speech) beyond loudness/visual change.
* Visual analysis is a lightweight change detector (scene cuts, motion bursts). It cannot recognise kills, round wins,
  or kill-feed events. The game-specific HUD plug-in point is designed (frames are already sampled) but **no game
  detector is implemented** (Valorant, CS2, Fortnite are future work, e.g. template matching on kill-feed regions).
* Facecam *reaction* detection (laughing/shouting face) is not implemented.
* Captions: Whisper `small` mis-hears names and slang; no profanity filtering / bleeping; ASS captions use the system
  fonts shipped with Windows (Arial Black default). Platform safe-margins are approximations, not official specs.
* Word timing across a reconnect gap inside one clip is not corrected (clips are restricted to one contiguous run).
* Buffer timing uses stream-copy piece durations; keyframe spacing of the source limits segment-cut precision before the
  accurate re-encode cut (clips are cut from the pieces with a frame-accurate re-encode, so output precision is fine).
* The cheap signal gate still decides *when* to wake the GPU; a speech-only moment with no audio change is found only by
  the periodic conversation scan (default 40 s), which costs a Whisper + LLM call about every 40 s of active stream.
* One GPU worker serializes jobs: a very busy stream can build a queue (bounded at 12; extra candidates are dropped with a log line).
* Multiple streams are supported structurally (one runtime per stream, shared GPU worker) but only one was exercised.
* No authentication on the dashboard: it binds to 127.0.0.1 only. Do not expose it to a network.
* Posting to platforms is intentionally not implemented.
* Windows only has been exercised (paths, `CREATE_NO_WINDOW`, install script).

## Future improvements
1. Real labelled benchmark from authorized VODs; tune thresholds per game with it.
2. Game plug-ins (kill-feed/round-win template matching; Haar replaced by YuNet for facecam).
3. A small vision-language model run only on 1-2 key frames of accepted candidates (sequential GPU scheduling already exists).
4. Learn weights from Accept/Reject decisions in the dashboard (the reviews are already stored).
5. Per-streamer baselines persisted across sessions; ads/intermission detection.
6. Profanity bleeping, emoji/keyword emphasis in captions, hook title overlay.
