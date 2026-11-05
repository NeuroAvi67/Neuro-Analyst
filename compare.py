"""Multi-dataset comparison. Assumptions are stated explicitly in the output."""
import re, difflib
import numpy as np
import pandas as pd
from scipy import stats as sp
from .stats import numeric_columns, categorical_columns, _py
from .advanced import time_series

def _norm(s): return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def match_columns(a_cols, b_cols):
    matches, used = [], set()
    for a in a_cols:
        if a in b_cols: matches.append({"a": a, "b": a, "how": "exact name"}); used.add(a)
    for a in a_cols:
        if any(m["a"] == a for m in matches): continue
        for b in b_cols:
            if b not in used and _norm(a) == _norm(b): matches.append({"a": a, "b": b, "how": "same name ignoring case/punctuation"}); used.add(b); break
    for a in a_cols:
        if any(m["a"] == a for m in matches): continue
        rest = [b for b in b_cols if b not in used]
        best = difflib.get_close_matches(_norm(a), [_norm(b) for b in rest], n=1, cutoff=0.85)
        if best:
            b = next(b for b in rest if _norm(b) == best[0]); matches.append({"a": a, "b": b, "how": "similar name (verify)"}); used.add(b)
    return matches

def _d(x, y):
    nx, ny = len(x), len(y)
    sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)) if nx + ny > 2 else 0
    return float((x.mean() - y.mean()) / sd) if sd else 0.0

def compare(a, b, names):
    na, nb = names
    matches = match_columns(list(a.columns), list(b.columns))
    ma, mb = {m["a"] for m in matches}, {m["b"] for m in matches}
    only_a, only_b = [c for c in a.columns if c not in ma], [c for c in b.columns if c not in mb]
    nums_a, nums_b = set(numeric_columns(a)), set(numeric_columns(b))
    numeric, categorical, findings, charts, pairs = [], [], [], [], []
    for m in matches:
        ca, cb = m["a"], m["b"]
        if ca in nums_a and cb in nums_b:
            x, y = a[ca].dropna().astype(float), b[cb].dropna().astype(float)
            if len(x) < 3 or len(y) < 3: continue
            w = sp.ttest_ind(x, y, equal_var=False); ks = sp.ks_2samp(x, y); d = _d(x, y)
            row = {"column": ca if ca == cb else f"{ca} / {cb}", "n_a": len(x), "n_b": len(y), "mean_a": x.mean(), "mean_b": y.mean(),
                   "median_a": x.median(), "median_b": y.median(), "std_a": x.std(), "std_b": y.std(), "cohens_d": d,
                   "welch_p": w.pvalue, "ks_p": ks.pvalue, "significant": bool(w.pvalue < .05)}
            numeric.append(row); pairs.append((ca, cb, row))
            if row["significant"]:
                size = "large" if abs(d) >= .8 else "medium" if abs(d) >= .5 else "small"
                findings.append(f"{row['column']}: mean is {x.mean():.4g} in {na} vs {y.mean():.4g} in {nb} (Welch p {'< 0.001' if w.pvalue < .001 else '= ' + format(w.pvalue, '.3f')}, {size} effect, Cohen's d = {d:.2f}).")
        elif ca not in nums_a and cb not in nums_b and 2 <= a[ca].nunique() <= 20 and 2 <= b[cb].nunique() <= 20:
            ct = pd.concat([a[ca].astype(str).value_counts().rename(na), b[cb].astype(str).value_counts().rename(nb)], axis=1).fillna(0)
            if ct.shape[0] >= 2:
                chi, p, _, _ = sp.chi2_contingency(ct.values)
                categorical.append({"column": ca, "chi2": chi, "p_value": p, "significant": bool(p < .05)})
                if p < .05: findings.append(f"{ca}: category mix differs between datasets (chi-square p {'< 0.001' if p < .001 else '= ' + format(p, '.3f')}).")
    trends = []
    tsa, tsb = time_series(a, [m["a"] for m in matches if m["a"] in nums_a]), time_series(b, [m["b"] for m in matches if m["b"] in nums_b])
    sa, sb = (tsa.get("series") or {}), (tsb.get("series") or {})
    for m in matches:
        if m["a"] in sa and m["b"] in sb:
            trends.append(f"{m['a']}: {na} shows {sa[m['a']]['trend']}, {nb} shows {sb[m['b']]['trend']}.")
    rel = []
    cm = [(x, y, r) for x, y, r in pairs]
    for i in range(len(cm)):
        for j in range(i + 1, len(cm)):
            a1, b1, a2, b2 = cm[i][0], cm[i][1], cm[j][0], cm[j][1]
            xa, xb = a[[a1, a2]].dropna(), b[[b1, b2]].dropna()
            if len(xa) > 5 and len(xb) > 5 and xa[a1].nunique() > 1 and xa[a2].nunique() > 1 and xb[b1].nunique() > 1 and xb[b2].nunique() > 1:
                ra, rb = xa[a1].corr(xa[a2]), xb[b1].corr(xb[b2])
                if abs(ra - rb) >= .3: rel.append(f"Correlation of {a1} with {a2} is r = {ra:.2f} in {na} but r = {rb:.2f} in {nb}.")
    top = sorted(pairs, key=lambda p: -abs(p[2]["cohens_d"]))[:3]
    for ca, cb, row in top:
        charts.append({"id": f"C{len(charts)+1}", "kind": "histogram", "title": f"{row['column']}: {na} vs {nb}", "figure": {"data": [
            {"type": "histogram", "x": a[ca].dropna().tolist(), "name": na, "opacity": .6, "nbinsx": 30},
            {"type": "histogram", "x": b[cb].dropna().tolist(), "name": nb, "opacity": .6, "nbinsx": 30}],
            "layout": {"barmode": "overlay", "title": {"text": f"{row['column']}: {na} vs {nb}", "font": {"size": 14}}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                       "font": {"size": 11, "color": "#cfd6ee"}, "legend": {"orientation": "h", "y": -0.28}, "margin": {"l": 48, "r": 14, "t": 48, "b": 48},
                       "xaxis": {"title": {"text": row["column"]}}, "yaxis": {"title": {"text": "Count"}}}}})
    ratio = len(matches) / max(1, min(a.shape[1], b.shape[1]))
    combined = {"possible": ratio >= .8 and bool(pairs), "match_ratio": ratio}
    assumptions = ["Columns were matched by name; 'similar name' matches should be verified by you.",
                   "Matched numeric columns are assumed to use the same units and scale in both datasets (this cannot be verified from the data).",
                   "Welch's t-test assumes the two datasets are independent samples; Cohen's d describes the size of the difference.",
                   "A statistical difference between datasets may reflect different populations, time periods or collection methods, not just the factor of interest."]
    if combined["possible"]:
        both = pd.concat([a[[m["a"] for m in matches]].assign(_source=na), b[[m["b"] for m in matches]].rename(columns={m["b"]: m["a"] for m in matches}).assign(_source=nb)], ignore_index=True)
        combined.update(rows=len(both), note="Schemas align closely, so a combined dataset is possible. It would stack rows and add a source column; assumes the datasets describe the same kind of records.")
        assumptions.append(f"Combined analysis is feasible ({combined['rows']} rows) but was not merged automatically; confirm the datasets describe the same kind of records.")
    else:
        combined["note"] = "Datasets were not combined: fewer than 80% of columns match, so merging would require assumptions that could mislead."
    summary = (f"{na} has {len(a)} rows x {a.shape[1]} columns; {nb} has {len(b)} rows x {b.shape[1]} columns. {len(matches)} column(s) matched. "
               f"{sum(r['significant'] for r in numeric)} of {len(numeric)} matched numeric column(s) differ significantly.")
    return _py({"names": [na, nb], "summary": summary, "matches": matches, "only_a": only_a, "only_b": only_b, "numeric": numeric,
                "categorical": categorical, "trends": trends, "relationships": rel[:5], "findings": findings, "charts": charts,
                "combined": combined, "assumptions": assumptions})
