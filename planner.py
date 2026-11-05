"""Turns a natural-language instruction into a validated analysis plan."""
import json, re
from .providers import ProviderError, OfflineProvider

METHODS = ["descriptive", "correlation", "regression", "group_tests", "outliers", "anomalies",
           "clusters", "time_series", "distributions"]
REPORT_TYPES = ["scientific", "executive", "business", "statistical", "top_findings", "general"]
_KW = {"scientific": ["scientific", "research", "paper"], "executive": ["executive", "summary", "brief"],
       "business": ["business", "performance", "kpi"], "statistical": ["statistical", "detailed"],
       "top_findings": ["most important", "top "]}
_METHOD_KW = {"correlation": ["correlat", "relationship"], "regression": ["regress", "predict"],
              "group_tests": ["compare", "difference", "hypothesis", "significan"], "outliers": ["outlier"],
              "anomalies": ["anomal"], "clusters": ["cluster", "segment", "group"],
              "time_series": ["trend", "time", "forecast", "season"], "distributions": ["distribution"]}

def offline_plan(instruction, columns):
    t = (instruction or "").lower()
    rtype = next((k for k, ws in _KW.items() if any(w in t for w in ws)), "general")
    focus = [m for m, ws in _METHOD_KW.items() if any(w in t for w in ws)]
    m = re.search(r"(\d+)\s+(?:most\s+)?(?:important|key|top)", t)
    cols = [c for c in columns if c.lower() in t]
    return {"report_type": rtype, "focus_methods": focus, "columns": cols,
            "top_n": min(int(m.group(1)), 30) if m else 10, "planner": "rule-based",
            "rationale": "Plan derived from keywords in the instruction. All analyses are run; focus areas are prioritised."}

def make_plan(provider, instruction, columns):
    base = offline_plan(instruction, columns)
    if isinstance(provider, OfflineProvider) or not provider.available():
        return base
    sys = "You plan data analyses. Reply with JSON only."
    prompt = (f"User instruction: {instruction!r}\nDataset columns: {columns}\n"
              f"Return JSON with keys: report_type (one of {REPORT_TYPES}), focus_methods (subset of {METHODS}), "
              f"columns (subset of the dataset columns relevant to the request), top_n (integer 1-30), rationale (one sentence).")
    try:
        raw = provider.complete(sys, prompt, json_mode=True)
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        plan = {"report_type": d.get("report_type") if d.get("report_type") in REPORT_TYPES else base["report_type"],
                "focus_methods": [m for m in d.get("focus_methods", []) if m in METHODS],
                "columns": [c for c in d.get("columns", []) if c in columns],
                "top_n": max(1, min(int(d.get("top_n", base["top_n"])), 30)),
                "rationale": str(d.get("rationale", ""))[:300], "planner": provider.name}
        return plan
    except (ProviderError, Exception):
        base["planner"] = "rule-based (AI planner failed)"
        return base
