"""SQLite storage for streams, clips and exports."""
import json
import sqlite3
import threading
import time

from . import config

_lock = threading.RLock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS streams(
  id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL, name TEXT, streamer TEXT, platform TEXT,
  created REAL, auto_start INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS clips(
  id INTEGER PRIMARY KEY AUTOINCREMENT, stream_id INTEGER, streamer TEXT, source_url TEXT,
  t_start REAL, t_end REAL, event_t REAL, trigger TEXT,
  score REAL, scores TEXT, category TEXT, reason TEXT, title TEXT, transcript TEXT, words TEXT,
  status TEXT, review TEXT DEFAULT 'pending', error TEXT, raw_path TEXT, fingerprint TEXT, created REAL);
CREATE TABLE IF NOT EXISTS exports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, clip_id INTEGER, key TEXT UNIQUE, layout TEXT, platform TEXT,
  path TEXT, created REAL);
"""


def conn():
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(config.SETTINGS.data_dir / "clipper.db", check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            try:
                _conn.execute("ALTER TABLE clips ADD COLUMN codex TEXT")
            except sqlite3.OperationalError:
                pass
        return _conn


def q(sql, args=()):
    with _lock:
        return [dict(r) for r in conn().execute(sql, args).fetchall()]


def one(sql, args=()):
    r = q(sql, args)
    return r[0] if r else None


def x(sql, args=()):
    with _lock:
        c = conn().execute(sql, args)
        conn().commit()
        return c.lastrowid


def update(table, row_id, **fields):
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            fields[k] = json.dumps(v)
    cols = ", ".join(f"{k}=?" for k in fields)
    x(f"UPDATE {table} SET {cols} WHERE id=?", (*fields.values(), row_id))


def now():
    return time.time()
