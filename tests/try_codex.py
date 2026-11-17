"""Manual check of the Codex judge on a throwaway clip cut from the synthetic sample. Sends synthetic data only."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import codex_judge, config, db, util  # noqa: E402

util.setup_logging()
raw = config.SETTINGS.data_dir / "sample" / "codex_test.mp4"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "102", "-t", "32", "-i", str(config.SETTINGS.data_dir / "sample" / "sample.mp4"),
                "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(raw)], check=True)
cid = db.x("INSERT INTO clips(stream_id,streamer,t_start,t_end,score,scores,category,reason,transcript,status,raw_path,created)"
           " VALUES(0,'synthetic_test',0,32,0.68,?,?,?,?,?,?,0)",
           (json.dumps({"audio": 0.76, "visual": 0.07, "chat": 0}), "surprise", "Unexpected reaction to a surprising event",
            "Okay going into the next match let's see how this goes. Oh my god. No way. He just did that. That is insane. I can't believe it.",
            "ready", str(raw)))
try:
    print(json.dumps(codex_judge.review(cid), indent=1))
finally:
    db.x("DELETE FROM clips WHERE id=?", (cid,))
    raw.unlink(missing_ok=True)
