"""Facts -> narrative -> verification.
Facts are generated deterministically from computed results. The AI may only explain them,
and every number in its output is checked against the facts."""
import re

def _f(v, d=3):
    return "n/a" if v is None else (f"{v:.{d}g}" if abs(v) < 0.001 and v != 0 else f"{round(v, d)}")

def _p(v):
    return "< 0.001" if v is not None and v < 0.001 else f"= {_f(v)}"

def build_facts(info, st, adv, clean_report=None):
    facts = []
    def add(cat, text, imp, cols=(), method=None):
        facts.append({"id": f"F{len(facts)+1}", "category": cat, "text": text, "importance": imp,
                      "columns": list(cols), "method": method})
    add("dataset", f"The dataset has {info['rows']} rows and {info['columns']} columns.", 1)
    if info["total_missing_cells"]:
        add("quality", f"{info['total_missing_cells']} cells are missing and {info['duplicate_rows']} rows are duplicated in the original data.", 3)
    if clean_report:
        a, b = clean_report["before"], clean_report["after"]
        add("quality", f"Cleaning changed the data from {a['rows']} to {b['rows']} rows and from {a['missing_cells']} to {b['missing_cells']} missing cells.", 2)
    for c in (st.get("correlations") or [])[:8]:
        if c["significant"] and abs(c["pearson_r"]) >= 0.4:
            add("correlation", f"{c['x']} and {c['y']} have a {c['strength']} {c['direction']} correlation (Pearson r = {_f(c['pearson_r'])}, p {_p(c['pearson_p'])}, n = {c['n']}).",
                3 + abs(c["pearson_r"]) * 6, (c["x"], c["y"]), "correlation")
    for r in (st.get("regressions") or [])[:3]:
        if r and not r.get("error") and r["significant"]:
            add("regression", f"A linear regression of {r['y']} on {r['x']} has slope {_f(r['slope'])} and R² = {_f(r['r_squared'])} (p {_p(r['p_value'])}, n = {r['n']}).",
                3 + r["r_squared"] * 4, (r["x"], r["y"]), "regression")
    for g in st.get("group_tests", []):
        if g["significant"]:
            means = ", ".join(f"{k}: {_f(v)}" for k, v in g["group_means"].items())
            add("group_test", f"{g['value_column']} differs significantly between groups of {g['group_column']} ({g['test']}, p {_p(g['p_value'])}). Group means - {means}.",
                6, (g["group_column"], g["value_column"]), "group_tests")
    for c in st.get("chi_square", []):
        if c["significant"]:
            add("association", f"{c['x']} and {c['y']} are statistically associated (chi-square = {_f(c['chi2'])}, p {_p(c['p_value'])}).", 5, (c["x"], c["y"]), "group_tests")
    for col, o in (adv.get("outliers") or {}).items():
        add("outliers", f"{col} has {o['iqr_outliers']} outlier(s) by the IQR rule ({o['pct_of_data']}% of values) outside the range {_f(o['iqr_bounds'][0])} to {_f(o['iqr_bounds'][1])}.",
            3 + min(o["pct_of_data"], 10) / 2, (col,), "outliers")
    an = adv.get("anomalies") or {}
    if an.get("anomaly_count"):
        add("anomalies", f"{an['anomaly_count']} rows ({an['pct']}%) are multivariate anomalies by Mahalanobis distance.", 4, (), "anomalies")
    cl = adv.get("clusters") or {}
    if cl.get("k"):
        add("clusters", f"k-means found {cl['k']} clusters with {cl['structure']} separation (silhouette = {_f(cl['silhouette'])}); cluster sizes: {', '.join(str(p['size']) for p in cl['profiles'])}.",
            5 if cl["structure"] != "weak" else 2, (), "clusters")
    for col, s in ((adv.get("time_series") or {}).get("series") or {}).items():
        if s["trend"] != "no significant trend":
            add("trend", f"{col} shows {'an' if s['trend'][0] in 'ai' else 'a'} {s['trend']} trend over time (Kendall tau = {_f(s['kendall_tau'])}, p {_p(s['kendall_p'])}), changing by {_f(s['change_pct_first_to_last'], 1)}% from first to last observation.",
                7, (col,), "time_series")
        else:
            add("trend", f"{col} shows no statistically significant trend over time (p {_p(s['kendall_p'])}).", 2, (col,), "time_series")
    for col, d in (adv.get("distributions") or {}).items():
        if d["skew_direction"] != "roughly symmetric":
            add("distribution", f"{col} is {d['skew_direction']}; the best-fitting of the tested distributions is {d['best_fit']}.", 2, (col,), "distributions")
    return facts

def rank_facts(facts, plan):
    focus, cols = set(plan.get("focus_methods", [])), set(plan.get("columns", []))
    def score(f):
        s = f["importance"]
        if f["method"] in focus: s += 3
        if cols and cols & set(f["columns"]): s += 3
        return s
    ranked = sorted(facts, key=lambda f: -score(f))
    return ranked

def limitations(info, st, adv):
    L = ["Correlation and association do not establish causation."]
    if info["rows"] < 30: L.append(f"With only {info['rows']} rows, statistical tests have low power and results are unstable.")
    if info["total_missing_cells"]: L.append("Missing values were present; filling them with the median/mode can understate variability.")
    n_tests = len(st.get("correlations", [])) + len(st.get("group_tests", [])) + len(st.get("chi_square", []))
    if n_tests > 10: L.append(f"{n_tests} tests were run; some 'significant' results may arise by chance (multiple comparisons).")
    if (adv.get("clusters") or {}).get("structure") == "weak": L.append("Cluster separation was weak; groups may not be meaningful.")
    if (adv.get("time_series") or {}).get("series"): L.append("Forecasts are simple linear extrapolations and ignore seasonality.")
    return L

def next_steps(facts):
    cats = {f["category"] for f in facts}
    S = []
    if "correlation" in cats: S.append("Test the strongest relationships with controlled data or additional variables before inferring cause.")
    if "outliers" in cats or "anomalies" in cats: S.append("Verify flagged outliers against the source system; decide whether they are errors or real events.")
    if "trend" in cats: S.append("Collect more time periods and check for seasonality.")
    if "clusters" in cats: S.append("Profile each cluster with domain knowledge to see if the segments are actionable.")
    return S or ["Collect more data or additional variables to enable deeper analysis."]

def offline_narrative(facts, plan, info, st, adv):
    top = rank_facts(facts, plan)
    top = [f for f in top if f["category"] != "dataset"][: plan.get("top_n", 10)]
    lines = ["## Summary", " ".join(f["text"] for f in facts if f["category"] == "dataset"),
             f"The strongest findings are listed below (ranked by statistical strength and relevance to your request).", "",
             "## Key findings"] + [f"{i}. {f['text']} [{f['id']}]" for i, f in enumerate(top, 1)]
    lines += ["", "## Limitations"] + [f"- {x}" for x in limitations(info, st, adv)]
    lines += ["", "## Suggested next steps"] + [f"- {x}" for x in next_steps(facts)]
    return "\n".join(lines)

_NUM = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?:e-?\d+)?")

def _numbers(text):
    text = re.sub(r"\bF\d+\b", " ", text)
    return [t for t in _NUM.findall(text)]

def verify_narrative(narrative, facts):
    """Every number in the narrative must appear (after rounding) in the facts. Small integers (<=10) are exempt (counting words, list numbers)."""
    allowed = set()
    for f in facts:
        for t in _numbers(f["text"]):
            try: allowed.add(float(t))
            except ValueError: pass
    bad = []
    for t in _numbers(narrative):
        try: n = float(t)
        except ValueError: continue
        if abs(n) <= 10 and float(n).is_integer(): continue
        dec = len(t.split(".")[1].split("e")[0]) if "." in t else 0
        if not any(round(a, dec) == n or abs(a - n) < 1e-9 for a in allowed):
            bad.append(t)
    return {"status": "verified" if not bad else "unverified_numbers", "unverified": sorted(set(bad))}

SYSTEM = ("You are a careful data analyst. You are given VERIFIED FACTS computed by Python. "
          "Rules: use ONLY numbers that appear in the facts; never invent statistics; cite fact ids like [F3]; "
          "never claim causation from correlation; be concise; plain English. Output Markdown with sections: "
          "## Summary, ## Key findings, ## Interpretation, ## Limitations, ## Suggested next steps.")

def ai_narrative(provider, facts, plan, instruction, info, st, adv):
    top = rank_facts(facts, plan)[: max(plan.get("top_n", 10) + 5, 12)]
    fact_txt = "\n".join(f"[{f['id']}] {f['text']}" for f in top)
    lim = "\n".join(f"- {x}" for x in limitations(info, st, adv))
    prompt = f"User request: {instruction!r}\nReport type: {plan['report_type']}\n\nFACTS:\n{fact_txt}\n\nKnown limitations:\n{lim}"
    text = provider.complete(SYSTEM, prompt)
    v = verify_narrative(text, facts)
    if v["status"] != "verified":
        retry = prompt + f"\n\nYour previous answer used numbers not in the facts: {v['unverified']}. Rewrite using only fact numbers."
        text = provider.complete(SYSTEM, retry)
        v = verify_narrative(text, facts)
    return text, v
