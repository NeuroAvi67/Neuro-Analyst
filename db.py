"""SQLite library: dataset and report metadata (the data itself lives in files under data/)."""
import sqlite3, time
from . import config

def _c():
    c = sqlite3.connect(config.DB_PATH, timeout=10); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, ext TEXT, size_bytes INT, rows INT, cols INT, project TEXT DEFAULT '', created REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, file_id TEXT, title TEXT, report_type TEXT, mode TEXT, project TEXT DEFAULT '', created REAL)""")
    return c

def q(sql, args=(), one=False, commit=False):
    with _c() as c:
        cur = c.execute(sql, args)
        if commit: return cur.rowcount
        r = [dict(x) for x in cur.fetchall()]
    return (r[0] if r else None) if one else r

def add_dataset(meta, rows, cols, project=""):
    q("INSERT OR REPLACE INTO datasets VALUES(?,?,?,?,?,?,?,?)", (meta["id"], meta["name"], meta["ext"], meta["size_bytes"], rows, cols, project, time.time()), commit=True)

def add_report(rid, file_id, title, rtype, mode):
    proj = (q("SELECT project FROM datasets WHERE id=?", (file_id,), one=True) or {}).get("project", "")
    q("INSERT OR REPLACE INTO reports VALUES(?,?,?,?,?,?,?)", (rid, file_id, title, rtype, mode, proj, time.time()), commit=True)

def search(table, term=""):
    col = "name" if table == "datasets" else "title"
    like = f"%{term}%"
    return q(f"SELECT * FROM {table} WHERE {col} LIKE ? OR project LIKE ? ORDER BY created DESC", (like, like))
