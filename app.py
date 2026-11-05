import re, glob, os, json, time, shutil, collections
from flask import Flask, request, jsonify, Response
import pandas as pd
from . import config, db, pipeline
from .engine.loader import save_upload, load_dataframe, LoadError
from .engine.inspector import inspect
from .engine.cleaner import clean
from .engine import stats
from .engine.advanced import advanced_analysis
from .engine.charts import make_chart
from .engine.compare import compare
from .ai import providers
from .reports import builder

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_MB * 1024 * 1024 * 4
HEX = re.compile(r"[0-9a-f]{32}")
_hits = collections.defaultdict(collections.deque)

@app.before_request
def limit():
    if request.method == "POST" and request.path.startswith("/api/"):
        q, now = _hits[request.remote_addr], time.time()
        while q and now - q[0] > 60: q.popleft()
        if len(q) >= config.RATE_LIMIT_PER_MIN:
            return jsonify(error="Too many requests. Please wait a minute."), 429
        q.append(now)

@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self' https://cdn.plot.ly; style-src 'self' 'unsafe-inline'; "
                                            "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'")
    if request.path.startswith("/api/"): r.headers["Cache-Control"] = "no-store"
    return r

def _err(msg, code=400): return jsonify(error=msg), code

def _fid(fid):
    if not HEX.fullmatch(fid or ""): raise LoadError("Invalid id.")
    return fid

def _find(fid):
    m = glob.glob(str(config.UPLOAD_DIR / f"{_fid(fid)}.*"))
    if not m: raise LoadError("Dataset not found (it may have been deleted).")
    return m[0]

def _working_df(body, fid=None):
    fid = fid or body.get("file_id"); path = _find(fid)
    pk = config.CLEAN_DIR / f"{fid}.pkl"
    if body.get("use_cleaned", True) and pk.exists(): return pd.read_pickle(pk), "cleaned copy"
    return load_dataframe(path), "original (uncleaned)"

def _result(rid):
    p = config.RESULT_DIR / f"{_fid(rid)}.json"
    if not p.exists(): raise LoadError("Report not found.")
    return json.loads(p.read_text())

@app.get("/")
def index(): return app.send_static_file("index.html")

@app.get("/api/health")
def health(): return jsonify(ok=True)

@app.post("/api/upload")
def upload():
    files, project = request.files.getlist("files"), (request.form.get("project") or "")[:60]
    if not files: return _err("No files received.")
    out = []
    for f in files:
        try:
            meta = save_upload(f); df = load_dataframe(meta["path"]); meta["inspection"] = inspect(df)
            db.add_dataset(meta, len(df), df.shape[1], project); meta.pop("path"); out.append(meta)
        except LoadError as e: out.append({"name": f.filename, "error": str(e)})
    return jsonify(files=out)

@app.post("/api/clean")
def api_clean():
    body = request.get_json(silent=True) or {}
    try: df = load_dataframe(_find(body.get("file_id")))
    except LoadError as e: return _err(str(e))
    cleaned, rep = clean(df, body.get("options")); cleaned.to_pickle(config.CLEAN_DIR / f"{body['file_id']}.pkl")
    rep["inspection_after"] = inspect(cleaned); return jsonify(rep)

@app.post("/api/stats")
def api_stats():
    try: df, src = _working_df(request.get_json(silent=True) or {})
    except LoadError as e: return _err(str(e))
    r = stats.full_analysis(df); r["data_source"] = src; return jsonify(r)

@app.post("/api/advanced")
def api_advanced():
    try: df, src = _working_df(request.get_json(silent=True) or {})
    except LoadError as e: return _err(str(e))
    r = advanced_analysis(df); r["data_source"] = src; return jsonify(r)

@app.post("/api/chart")
def api_chart():
    b = request.get_json(silent=True) or {}
    try:
        df, _ = _working_df(b); return jsonify(figure=make_chart(df, b.get("kind"), b.get("x"), b.get("y"), b.get("color"), b.get("units")))
    except (LoadError, ValueError) as e: return _err(str(e))

@app.post("/api/analyze")
def api_analyze():
    b = request.get_json(silent=True) or {}
    try: path = _find(b.get("file_id"))
    except LoadError as e: return _err(str(e))
    ds = db.q("SELECT name FROM datasets WHERE id=?", (b["file_id"],), one=True)
    job = pipeline.start(path, b["file_id"], (b.get("instruction") or "")[:2000], b.get("provider"), b.get("clean_options"), name=(ds or {}).get("name"))
    return jsonify(job_id=job["id"]), 202

@app.get("/api/jobs/<jid>")
def api_job(jid):
    job = pipeline.JOBS.get(jid)
    if not job: return _err("Unknown job (the server may have restarted). Check the Library for saved reports.", 404)
    out = {k: job[k] for k in ("status", "stage", "label", "progress") if k in job}
    if job["status"] == "error": out["error"] = job["error"]
    if job["status"] == "done": out["report_id"] = jid
    return jsonify(out)

# ---- reports ----
@app.get("/api/reports/<rid>")
def api_report(rid):
    try:
        res = _result(rid); rep = builder.build_report(res, request.args.get("type"))
        return jsonify(report=rep, result=res)
    except LoadError as e: return _err(str(e), 404)

def _download(data, fmt, title):
    mime = {"pdf": "application/pdf", "html": "text/html", "md": "text/markdown"}[fmt]
    fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", title)[:60] + "." + fmt
    return Response(data, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="{fn}"'})

def _render(rep, charts, fmt):
    if fmt == "md": return builder.to_markdown(rep, charts)
    if fmt == "html": return builder.to_html(rep, charts)
    if fmt == "pdf": return builder.to_pdf(rep, charts)
    raise ValueError("Format must be pdf, html or md.")

@app.get("/api/reports/<rid>/export")
def api_export(rid):
    fmt = request.args.get("format", "pdf")
    try:
        res = _result(rid); rep = builder.build_report(res, request.args.get("type"))
        return _download(_render(rep, res["charts"], fmt), fmt, rep["title"])
    except (LoadError, ValueError) as e: return _err(str(e))

@app.route("/api/reports/<rid>", methods=["PATCH", "DELETE"])
def api_report_edit(rid):
    try: _fid(rid)
    except LoadError as e: return _err(str(e))
    if request.method == "DELETE":
        (config.RESULT_DIR / f"{rid}.json").unlink(missing_ok=True); db.q("DELETE FROM reports WHERE id=?", (rid,), commit=True)
    else:
        b = request.get_json(silent=True) or {}
        if b.get("title"): db.q("UPDATE reports SET title=? WHERE id=?", (b["title"][:120], rid), commit=True)
        if "project" in b: db.q("UPDATE reports SET project=? WHERE id=?", (b["project"][:60], rid), commit=True)
    return jsonify(ok=True)

# ---- library / datasets ----
@app.get("/api/library")
def api_library():
    term = request.args.get("q", "")
    return jsonify(datasets=db.search("datasets", term), reports=db.search("reports", term), storage=_storage())

@app.route("/api/datasets/<did>", methods=["PATCH", "DELETE"])
def api_dataset(did):
    try: _fid(did)
    except LoadError as e: return _err(str(e))
    if request.method == "DELETE":
        for f in glob.glob(str(config.UPLOAD_DIR / f"{did}.*")): os.remove(f)
        (config.CLEAN_DIR / f"{did}.pkl").unlink(missing_ok=True)
        db.q("DELETE FROM datasets WHERE id=?", (did,), commit=True)
        if request.args.get("with_reports") == "1":
            for r in db.q("SELECT id FROM reports WHERE file_id=?", (did,)):
                (config.RESULT_DIR / f"{r['id']}.json").unlink(missing_ok=True)
            db.q("DELETE FROM reports WHERE file_id=?", (did,), commit=True)
    else:
        b = request.get_json(silent=True) or {}
        if b.get("name"): db.q("UPDATE datasets SET name=? WHERE id=?", (b["name"][:120], did), commit=True)
        if "project" in b: db.q("UPDATE datasets SET project=? WHERE id=?", (b["project"][:60], did), commit=True)
    return jsonify(ok=True)

def _size(d): return sum(f.stat().st_size for f in d.glob("*") if f.is_file())
def _storage():
    return {"uploads_bytes": _size(config.UPLOAD_DIR), "cleaned_bytes": _size(config.CLEAN_DIR), "reports_bytes": _size(config.RESULT_DIR),
            "max_file_mb": config.MAX_FILE_MB, "datasets": len(db.q("SELECT id FROM datasets")), "reports": len(db.q("SELECT id FROM reports"))}

@app.get("/api/storage")
def api_storage(): return jsonify(_storage())

@app.post("/api/storage/clear")
def api_clear():
    for d in (config.UPLOAD_DIR, config.CLEAN_DIR, config.RESULT_DIR):
        for f in d.glob("*"):
            if f.is_file(): f.unlink()
    db.q("DELETE FROM datasets", commit=True); db.q("DELETE FROM reports", commit=True)
    return jsonify(ok=True)

@app.get("/api/privacy")
def api_privacy():
    p = providers.get_provider()
    return jsonify(storage=("Uploaded files, cleaned copies and reports are stored unencrypted on the server running this app, under its data/ folder. "
                            "Whoever operates the server can access them. On free hosting the disk is often temporary and may be wiped on restart."),
                   ai=("Offline mode sends nothing anywhere. In AI mode, only your instruction, column names and computed statistical summaries are sent to the selected provider; raw data rows are never sent. "
                       "Free AI tiers may log or use prompts according to their own terms."),
                   active_provider=p.name, deletion="You can delete any dataset (and optionally its reports) or erase everything from Settings.")

@app.get("/api/ai/status")
def api_ai_status():
    return jsonify(providers=providers.status(), default=providers.get_provider().name, note="API keys are read from server environment variables and never sent to the browser.")

# ---- comparison ----
def _cmp(b):
    ids = b.get("file_ids") or []
    if len(ids) != 2: raise LoadError("Select exactly two datasets to compare.")
    dfs = [_working_df(b, i)[0] for i in ids]
    names = [(db.q("SELECT name FROM datasets WHERE id=?", (i,), one=True) or {}).get("name", "Dataset") for i in ids]
    if names[0] == names[1]: names = [names[0] + " (A)", names[1] + " (B)"]
    return compare(dfs[0], dfs[1], names)

@app.post("/api/compare")
def api_compare():
    try:
        c = _cmp(request.get_json(silent=True) or {}); return jsonify(comparison=c, report=builder.build_comparison_report(c))
    except LoadError as e: return _err(str(e))

@app.post("/api/compare/export")
def api_compare_export():
    b = request.get_json(silent=True) or {}; fmt = b.get("format", "pdf")
    try:
        c = _cmp(b); rep = builder.build_comparison_report(c)
        return _download(_render(rep, c["charts"], fmt), fmt, rep["title"])
    except (LoadError, ValueError) as e: return _err(str(e))

@app.errorhandler(413)
def too_big(_): return _err("Upload too large.", 413)

@app.errorhandler(500)
def oops(_): return _err("Unexpected server error. Please try again.", 500)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
