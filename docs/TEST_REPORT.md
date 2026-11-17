# Test report (measured on this machine)

Hardware: RTX 4060 Ti 8 GB, i5-14400F, 32 GB RAM, Windows 11. Software: faster-whisper `small` fp16 on GPU, Ollama
`qwen2.5:3b-instruct`, ffmpeg 9.0.1 (h264_nvenc), OpenCV 4.14.

## IMPORTANT caveats
* **No real livestream and no real gameplay footage was available**, and no permitted stream URL was provided. All detection
  tests use a **synthetic labelled stream** (`tests/make_sample.py`: Windows TTS voices over generated game-like noise and
  a generated video, 13 min, 5 labelled highlights, 3 decoys). It is far easier than real content in some ways (clean
  speech) and different in others (no real gameplay). **Thresholds and detector logic were iterated against this same set, so
  the scores below are over-fitted by construction and must not be read as expected real-world accuracy.**
* The live pipeline was exercised by replaying the sample file at real-time speed (`ffmpeg -re`) through the real recorder.
  streamlink was only tested for offline detection (non-existent channel reported offline).

## Detection on the labelled sample (final configuration)
Highlights: shout, funny-but-quiet joke, clutch, funny-but-quiet story, rage rant. Decoys: explosion noise, loud music with
boring talk, scene cut with calm talk. (`tests/run_eval.py`, output in `data/sample/eval_log.txt`)

| Detector | Detections | Found | Recall | Precision | Missed |
|---|---|---|---|---|---|
| A. audio-only (loudness / vocal-intensity) | 3 | 3/5 | 0.60 | 1.00 | both quiet funny moments |
| B. audio + visual + keyword lexicon (no LLM) | 3 | 3/5 | 0.60 | 1.00 | both quiet funny moments |
| C. full: + Whisper + Qwen2.5-3B + conversation scan | 5 | 5/5 | 1.00 | 1.00 | none |

What the numbers do and don't say: quiet jokes cannot be found by loudness at all; the LLM + scan found them. The decoys were
rejected by the LLM when they got through (explosion: no speech; boring chatter scored 0.15-0.3).
**Journey, for honesty:** the first full run scored audio-only precision 0.26 (19 detections) because calm talking was
re-detected as a spike forever (baseline bug); fixing the baseline to compare against the stream's own *active* loudness
raised audio-only precision to 1.00 on this set. Whisper also silently dropped a shouted sentence in a 50 s window (fixed by
chunking at quiet points); and an early conversation-scan design left 8 s gaps (fixed by tiling windows).

Clip timing: all 5 accepted clips fully contained the labelled moment (clips 32-42 s; pre-roll/post-roll snapped to pauses).
Category labels from the 3B model are unreliable (a funny-quiet joke was labelled "rage"; clutch phrasing was fine).

## Performance (measured)
| Metric | Result |
|---|---|
| Whisper + Qwen processing per candidate (eval, GPU) | mean 5.6 s, max 6.9 s (earlier run 9.0 s mean / 12 s max when Ollama had to reload; first-ever call ~100 s including model load) |
| Eval throughput | 791 s of stream analysed + judged in 67-123 s (offline, faster than real time) |
| VRAM | 5.7 GB used at idle with Whisper loaded + Qwen resident (incl. ~0.8 GB desktop); peak 6.5 GB; **no OOM** over all runs. Whisper unloads after 5 idle minutes; Ollama after 3 |
| GPU utilization while recording, no job running | 1-3 % (sampled 5x over 10 s; spikes to ~10 % while a job ran) |
| Disk (buffer) | 240 s of 720p buffer = 80 MB, stayed bounded for ~10 min of recording; pinned pieces are kept until their clip is cut |
| Disk (clips) | raw 16:9 clip ~26 MB for 32 s at CQ19; vertical export ~28 MB (CQ21, 1080x1920, 30 fps); landscape 45 MB |
| Export time | 3-20 s per export with NVENC (first one slower: font + model warm-up) |
| Jobs | 5 completed, 0 failed in the live run |

## Capture robustness
* Recorder killed with `taskkill` mid-stream: the loop logged "dropped (30 pieces); reconnecting in 5s", resumed, state returned
  to `recording` and buffer/analysis continued (observed within 20 s of the kill; only one drop tested).
* Server restart: streams with auto-start resumed on their own; existing clips and exports persisted (SQLite).
* A failed export no longer marks the clip as failed (bug found by screenshotting the dashboard, fixed).
* Not tested: real network loss, stream going offline and returning, multi-hour runs, multiple streams, YouTube/Kick.

## Export quality
Verified: 1080x1920 H.264+AAC for blur/crop/auto layouts, 1920x1080 for landscape, NVENC used, word-highlight captions visible
and positioned above the bottom UI margin (visual check of frames). Not measured: VMAF/PSNR, real facecam split, caption
overlap with real platform UI.

## Unit tests
`python -m pytest tests/test_units.py`: 9 passed (boundaries, lexicon, ASR chunking, audio scorer, segment pin/prune, timeline gaps,
ASS captions, chat spike, platform detection).

## Transcription quality
Spot-checked only: sentences of the synthetic speech were transcribed verbatim, including the shouted/clipped ones. No WER was
computed and TTS speech is much easier than real streamers with game audio underneath.
