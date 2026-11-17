"""Re-align captions of existing clips by transcribing each clip's own audio, then re-export. Usage: repair_captions.py ID..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipper import db, pipeline, util
util.setup_logging(); db.conn()
for cid in map(int, sys.argv[1:]):
    c = db.one("SELECT * FROM clips WHERE id=?", (cid,))
    if not c or not c["raw_path"]:
        print(cid, "skipped"); continue
    pipeline._realign_words(cid, Path(c["raw_path"]), c["t_start"])
    for e in db.q("SELECT * FROM exports WHERE clip_id=?", (cid,)):
        Path(e["path"]).unlink(missing_ok=True)
    db.x("DELETE FROM exports WHERE clip_id=?", (cid,))
    ex = pipeline.export(cid, "auto", "tiktok", True)
    print(cid, "re-exported ->", Path(ex["path"]).name, ex["layout"])
