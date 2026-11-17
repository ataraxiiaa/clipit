"""FastAPI app: JSON API + static dashboard. Heavy work runs in background threads, never in request handlers."""
import json
import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

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
    return MGR.state()


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


@app.post("/api/clips/{cid}/review")
def review(cid: int, body: dict = Body(...)):
    if body.get("review") not in ("accepted", "rejected", "pending"):
        raise HTTPException(400, "review must be accepted/rejected/pending")
    if body["review"] == "rejected":
        # reject = delete: remove the video files and exports, keep a tiny record so the moment is not re-clipped
        import shutil
        c = db.one("SELECT id FROM clips WHERE id=?", (cid,))
        if not c:
            raise HTTPException(404, "no such clip")
        shutil.rmtree(config.SETTINGS.data_dir / "clips" / str(cid), ignore_errors=True)
        db.x("DELETE FROM exports WHERE clip_id=?", (cid,))
        db.update("clips", cid, review="rejected", status="rejected", raw_path=None, words=None)
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
