"""Asynchronous analysis pipeline with live progress. Jobs run in a background thread."""
import threading, uuid, json, time, traceback
import pandas as pd
from . import config, db
from .engine.loader import load_dataframe, LoadError
from .engine.inspector import inspect
from .engine.cleaner import clean
from .engine import stats
from .engine.advanced import advanced_analysis
from .engine.charts import auto_charts
from .ai.providers import get_provider, ProviderError
from .ai.planner import make_plan
from .ai import interpreter as I

JOBS = {}
STAGES = [("inspect", "Inspecting dataset"), ("clean", "Cleaning a copy of the data"), ("plan", "Understanding your request"),
          ("stats", "Running statistical analysis"), ("advanced", "Running advanced analysis"), ("charts", "Building charts"),
          ("interpret", "Interpreting verified results"), ("verify", "Checking conclusions against calculations"), ("done", "Complete")]

def _set(job, stage, **kw):
    idx = [s for s, _ in STAGES].index(stage)
    job.update(stage=stage, label=dict(STAGES)[stage], progress=round(idx / (len(STAGES) - 1) * 100), **kw)

def run(job, path, file_id, instruction, provider_name, clean_opts, name):
    try:
        job["status"] = "running"
        _set(job, "inspect"); df0 = load_dataframe(path); info = inspect(df0)
        _set(job, "clean"); df, creport = clean(df0, clean_opts); df.to_pickle(config.CLEAN_DIR / f"{file_id}.pkl")
        _set(job, "plan"); provider = get_provider(provider_name)
        plan = make_plan(provider, instruction, list(df.columns))
        _set(job, "stats"); st = stats.full_analysis(df)
        _set(job, "advanced"); adv = advanced_analysis(df)
        _set(job, "charts"); charts = auto_charts(df, st, adv)
        _set(job, "interpret"); facts = I.build_facts(info, st, adv, creport)
        mode, warning = "offline", None
        narrative = None
        if provider.name != "offline" and provider.available():
            try:
                narrative, ver = I.ai_narrative(provider, facts, plan, instruction, info, st, adv)
                mode = f"AI ({provider.name})"
                _set(job, "verify")
                if ver["status"] != "verified":
                    warning = f"AI wording contained numbers not found in the calculations ({', '.join(ver['unverified'])}); showing the verified statistical summary instead."
                    narrative, mode = None, "offline (AI output rejected)"
            except ProviderError as e:
                warning = f"AI unavailable ({e}). Showing the offline statistical summary."
        _set(job, "verify")
        if narrative is None:
            narrative = I.offline_narrative(facts, plan, info, st, adv)
            ver = I.verify_narrative(narrative, facts)
        result = {"job_id": job["id"], "file_id": file_id, "instruction": instruction, "plan": plan, "mode": mode,
                  "warning": warning, "verification": ver, "narrative": narrative, "facts": facts, "inspection": info,
                  "dataset_name": name or "Dataset", "cleaning": creport, "stats": st, "advanced": adv, "charts": charts, "created": time.time()}
        (config.RESULT_DIR / f"{job['id']}.json").write_text(json.dumps(result))
        db.add_report(job["id"], file_id, f"Analysis Report: {name or 'Dataset'}", plan["report_type"], mode)
        _set(job, "done", status="done", result_path=str(config.RESULT_DIR / f"{job['id']}.json"))
    except LoadError as e:
        job.update(status="error", error=str(e))
    except Exception as e:
        traceback.print_exc(); job.update(status="error", error=f"Analysis failed: {type(e).__name__}: {e}")

def start(path, file_id, instruction, provider_name=None, clean_opts=None, wait=False, name=None):
    job = {"id": uuid.uuid4().hex, "status": "queued", "stage": "queued", "label": "Queued", "progress": 0}
    JOBS[job["id"]] = job
    t = threading.Thread(target=run, args=(job, path, file_id, instruction, provider_name, clean_opts, name), daemon=True)
    t.start()
    if wait: t.join()
    return job
