"""FastAPI app: JSON API + static dashboard. Heavy work runs in background threads, never in request handlers."""
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask

from . import codex_judge, config, db, pipeline, util, vod

log = logging.getLogger("clipper.server")
app = FastAPI(title="Livestream Clipper")
MGR: pipeline.Manager = None
WEB = Path(__file__).resolve().parent.parent / "web"


@app.on_event("startup")
def _startup():
    global MGR
    util.setup_logging()
    db.conn()
    MGR = pipeline.Manager()
    threading.Thread(target=MGR.resume_all, daemon=True).start()
    log.info("Clipper ready on http://%s:%s", config.get("server.host"), config.get("server.port"))


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.post("/api/vod")
def add_vod(body: dict = Body(...)):
    try:
        job = vod.start(body.get("url", ""), body.get("streamer", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job.id}


@app.post("/api/vod/{jid}/cancel")
def cancel_vod(jid: int):
    j = vod.JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such video job")
    j.cancel.set()
    return {"ok": True}


@app.get("/api/state")
def state():
    s = MGR.state()
    s["bulk"] = [BULK[k] for k in sorted(BULK)[-3:]]
    return s


@app.post("/api/streams")
def add_stream(body: dict = Body(...)):
    try:
        sid = MGR.add(body.get("url", ""), body.get("name", ""), body.get("streamer", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if body.get("start", True):
        MGR.start(sid)
    return {"id": sid}


@app.delete("/api/streams/{sid}")
def del_stream(sid: int):
    MGR.remove(sid)
    return {"ok": True}


@app.post("/api/streams/{sid}/{action}")
def stream_action(sid: int, action: str, body: dict = Body(default={})):
    try:
        if action == "start":
            MGR.start(sid)
        elif action == "stop":
            MGR.stop(sid)
        elif action == "clip":
            rt = MGR.rt.get(sid)
            if not rt:
                raise ValueError("stream is not running")
            rt.an.manual(float(body.get("seconds", 45)))
        else:
            raise ValueError("unknown action")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


def _clip_row(c, exports):
    c = dict(c)
    for k in ("scores", "words", "codex"):
        try:
            c[k] = json.loads(c[k]) if c.get(k) else None
        except Exception:  # noqa: BLE001
            c[k] = None
    c.pop("words", None)
    c["exports"] = [e for e in exports if e["clip_id"] == c["id"]]
    return c


@app.get("/api/clips")
def clips(stream_id: int | None = None, status: str = "ready", review: str | None = None, limit: int = 100):
    sql, args = "SELECT * FROM clips WHERE 1=1", []
    if status != "all":
        sql += " AND status=?"
        args.append(status)
    if stream_id:
        sql += " AND stream_id=?"
        args.append(stream_id)
    if review:
        sql += " AND review=?"
        args.append(review)
    rows = db.q(sql + " ORDER BY score DESC, created DESC LIMIT ?", (*args, limit))
    ex = db.q("SELECT * FROM exports")
    return [_clip_row(r, ex) for r in rows]


def _reject(cid: int) -> bool:
    """reject = delete: remove the video files and exports, keep a tiny record so the moment is not re-clipped."""
    if not db.one("SELECT id FROM clips WHERE id=?", (cid,)):
        return False
    shutil.rmtree(config.SETTINGS.data_dir / "clips" / str(cid), ignore_errors=True)
    db.x("DELETE FROM exports WHERE clip_id=?", (cid,))
    db.update("clips", cid, review="rejected", status="rejected", raw_path=None, words=None)
    return True


BULK: dict = {}


def _run_bulk(job: dict, ids: list, opts: dict):
    for cid in ids:
        try:
            if job["action"] == "export":
                pipeline.export(cid, opts["layout"], opts["platform"], opts["captions"])
            else:
                codex_judge.review(cid)
        except Exception as e:  # noqa: BLE001 - one failing clip must not stop the rest
            job["errors"].append(f"clip {cid}: {str(e)[-140:]}")
        job["done"] += 1
    job["state"] = "done"
    job["finished"] = time.time()


@app.post("/api/bulk")
def bulk(body: dict = Body(...)):
    ids = [int(i) for i in body.get("ids", []) if str(i).isdigit()][:500]
    action = body.get("action")
    if not ids:
        raise HTTPException(400, "no clips selected")
    if action in ("accept", "reject"):
        n = 0
        for cid in ids:
            if action == "reject":
                n += _reject(cid)
            else:
                db.update("clips", cid, review="accepted")
                n += 1
        return {"ok": True, "count": n}
    if action not in ("export", "codex"):
        raise HTTPException(400, "unknown action")
    if action == "codex" and not codex_judge.available():
        raise HTTPException(400, "Codex CLI not found. Install it and run `codex login`.")
    if any(j["state"] == "running" for j in BULK.values()):
        raise HTTPException(400, "A bulk job is already running. Wait for it to finish.")
    jid = (max(BULK) + 1) if BULK else 1
    job = {"id": jid, "action": action, "total": len(ids), "done": 0, "errors": [], "state": "running", "started": time.time()}
    BULK[jid] = job
    opts = {"layout": body.get("layout", config.get("export.layout")), "platform": body.get("platform", "tiktok"),
            "captions": body.get("captions", config.get("export.captions"))}
    threading.Thread(target=_run_bulk, args=(job, ids, opts), daemon=True, name=f"bulk-{jid}").start()
    return {"ok": True, "job": jid}


@app.get("/api/zip")
def zip_clips(ids: str, kind: str = "export"):
    """One .zip of the selected clips: newest export of each (or the raw 16:9 file with kind=raw / when not exported yet)."""
    files = []
    for cid in [int(x) for x in ids.split(",") if x.strip().isdigit()][:200]:
        c = db.one("SELECT * FROM clips WHERE id=?", (cid,))
        if not c:
            continue
        path = None
        if kind == "export":
            path = next((e["path"] for e in db.q("SELECT path FROM exports WHERE clip_id=? ORDER BY id DESC", (cid,))
                         if Path(e["path"]).exists()), None)
        path = path or c["raw_path"]
        if path and Path(path).exists():
            safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or ""))[:40].strip("_")
            files.append((f"{safe(c['streamer'])}_{cid}_{safe(c['title'])}.mp4", path))
    if not files:
        raise HTTPException(404, "none of the selected clips has a video file")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as z:     # mp4 is already compressed
        for name, p in files:
            z.write(p, name)
    return FileResponse(tmp.name, media_type="application/zip", filename="clips.zip", background=BackgroundTask(os.unlink, tmp.name))


@app.post("/api/clips/{cid}/review")
def review(cid: int, body: dict = Body(...)):
    if body.get("review") not in ("accepted", "rejected", "pending"):
        raise HTTPException(400, "review must be accepted/rejected/pending")
    if body["review"] == "rejected":
        if not _reject(cid):
            raise HTTPException(404, "no such clip")
        return {"ok": True, "deleted": True}
    db.update("clips", cid, review=body["review"])
    return {"ok": True}


@app.post("/api/clips/{cid}/export")
def export(cid: int, body: dict = Body(default={})):
    try:
        e = pipeline.export(cid, body.get("layout", config.get("export.layout")), body.get("platform", "tiktok"),
                            body.get("captions", config.get("export.captions")))
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(400, str(ex)[-300:])
    return e


@app.post("/api/clips/{cid}/codex")
def ask_codex(cid: int):
    if not codex_judge.available():
        raise HTTPException(400, "Codex CLI not found. Install it and run `codex login`.")
    try:
        return codex_judge.review(cid)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)[-400:])


@app.get("/media/raw/{cid}")
def media_raw(cid: int):
    c = db.one("SELECT raw_path FROM clips WHERE id=?", (cid,))
    if not c or not c["raw_path"] or not Path(c["raw_path"]).exists():
        raise HTTPException(404)
    return FileResponse(c["raw_path"], media_type="video/mp4")


@app.get("/media/export/{eid}")
def media_export(eid: int):
    e = db.one("SELECT path FROM exports WHERE id=?", (eid,))
    if not e or not Path(e["path"]).exists():
        raise HTTPException(404)
    return FileResponse(e["path"], media_type="video/mp4", filename=Path(e["path"]).name)


@app.get("/api/settings")
def get_settings():
    return config.SETTINGS.snapshot()


@app.put("/api/settings")
def put_settings(body: dict = Body(...)):
    return config.SETTINGS.update(body)


@app.get("/api/logs")
def logs():
    return list(util.RING)[-200:]


def main():
    util.setup_logging()
    uvicorn.run(app, host=config.get("server.host"), port=config.get("server.port"), log_level="warning")


if __name__ == "__main__":
    main()
