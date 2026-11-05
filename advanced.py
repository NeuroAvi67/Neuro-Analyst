"""Advanced analysis: outliers, anomalies, clustering, distributions, time series, trends, forecast.
Pure numpy/scipy (no heavy ML dependencies). All outputs are computed from the data."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from scipy.cluster.vq import kmeans2
from .stats import numeric_columns, _py
from .inspector import infer_type

def outliers(df, cols):
    res = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if len(x) < 8: continue
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_idx = x[(x < lo) | (x > hi)]
        sd = x.std()
        z = ((x - x.mean()) / sd) if sd > 0 else x * 0
        z_idx = x[z.abs() > 3]
        if len(iqr_idx) or len(z_idx):
            res[c] = {"iqr_bounds": [lo, hi], "iqr_outliers": len(iqr_idx), "zscore_outliers": len(z_idx),
                      "pct_of_data": round(len(iqr_idx) / len(x) * 100, 2),
                      "examples": [{"row": int(i), "value": v} for i, v in iqr_idx.abs().sort_values(ascending=False).head(5).items()
                                   for v in [float(x[i])]]}
    return res

def anomalies(df, cols, alpha=0.001):
    """Multivariate anomalies via Mahalanobis distance (chi-square threshold)."""
    d = df[cols].dropna().astype(float)
    if len(cols) < 2 or len(d) < 20: return {"skipped": "Needs at least 2 numeric columns and 20 complete rows."}
    X = d.values - d.values.mean(0)
    cov = np.cov(X, rowvar=False)
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return {"skipped": "Columns are perfectly collinear; Mahalanobis distance undefined."}
    m2 = np.einsum("ij,jk,ik->i", X, inv, X)
    thr = sp.chi2.ppf(1 - alpha, df=len(cols))
    idx = np.where(m2 > thr)[0]
    top = idx[np.argsort(-m2[idx])][:10]
    return {"method": "Mahalanobis distance", "threshold": thr, "anomaly_count": len(idx),
            "pct": round(len(idx) / len(d) * 100, 2),
            "top": [{"row": int(d.index[i]), "distance_sq": m2[i]} for i in top]}

def _silhouette(X, labels):
    n = len(X); D = np.linalg.norm(X[:, None] - X[None], axis=2)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        if same.sum() < 2: continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == k].mean() for k in set(labels) if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())

def clusters(df, cols, max_k=6, seed=0):
    d = df[cols].dropna().astype(float)
    if len(cols) < 2 or len(d) < 30: return {"skipped": "Needs at least 2 numeric columns and 30 complete rows."}
    sub = d.sample(1500, random_state=seed) if len(d) > 1500 else d
    Z = (sub - sub.mean()) / sub.std().replace(0, 1)
    best = None; scores = {}
    for k in range(2, min(max_k, len(sub) // 10) + 1):
        np.random.seed(seed)
        _, lab = kmeans2(Z.values, k, minit="++", seed=seed)
        if len(set(lab)) < k: continue
        sc = _silhouette(Z.values[:600], lab[:600]) if len(set(lab[:600])) == k else -1
        scores[k] = sc
        if best is None or sc > best[1]: best = (k, sc, lab)
    if not best: return {"skipped": "Could not form stable clusters."}
    k, sc, lab = best
    prof = []
    for j in range(k):
        m = sub[lab == j]
        prof.append({"cluster": j, "size": len(m), "share_pct": round(len(m) / len(sub) * 100, 1),
                     "centroid": m.mean().to_dict()})
    quality = "strong" if sc > .5 else "reasonable" if sc > .25 else "weak"
    return {"method": "k-means (standardised)", "k": k, "silhouette": sc, "silhouette_by_k": scores,
            "structure": quality, "profiles": prof, "rows_used": len(sub),
            "note": "Weak silhouette means the data may not contain distinct groups." if quality == "weak" else None}

def distributions(df, cols):
    res = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if len(x) < 20 or x.nunique() < 5: continue
        fits = []
        for name, dist in [("normal", sp.norm), ("lognormal", sp.lognorm), ("exponential", sp.expon)]:
            if name != "normal" and (x <= 0).any(): continue
            try:
                p = dist.fit(x); ks = sp.kstest(x, dist.cdf, args=p)
                fits.append({"distribution": name, "ks_statistic": ks.statistic, "p_value": ks.pvalue})
            except Exception: pass
        if fits:
            fits.sort(key=lambda f: f["ks_statistic"])
            res[c] = {"best_fit": fits[0]["distribution"], "fits": fits,
                      "skew_direction": "right-skewed" if sp.skew(x) > .5 else "left-skewed" if sp.skew(x) < -.5 else "roughly symmetric"}
    return res

def find_time_column(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]): return c, df[c]
    for c in df.columns:
        if infer_type(df[c]) == "datetime (stored as text)":
            return c, pd.to_datetime(df[c], errors="coerce", format="mixed")
    return None, None

def time_series(df, cols, horizon=5):
    tc, t = find_time_column(df)
    if tc is None: return {"skipped": "No date/time column found."}
    out = {"time_column": tc, "series": {}}
    for c in cols:
        d = pd.DataFrame({"t": t, "v": pd.to_numeric(df[c], errors="coerce")}).dropna()
        d = d.groupby("t")["v"].mean().sort_index()
        if len(d) < 8: continue
        y = d.values; x = np.arange(len(y))
        lr = sp.linregress(x, y); tau, p_tau = sp.kendalltau(x, y)
        ac1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if y.std() > 0 else None
        trend = "no significant trend" if p_tau >= .05 else ("increasing" if tau > 0 else "decreasing")
        steps = pd.Series(d.index).diff().dropna()
        step = steps.median() if len(steps) else None
        resid = y - (lr.intercept + lr.slope * x); s = resid.std(ddof=2)
        fc = [{"step": int(len(y) + i), "forecast": lr.intercept + lr.slope * (len(y) + i),
               "approx_95_range": [lr.intercept + lr.slope * (len(y) + i) - 1.96 * s,
                                   lr.intercept + lr.slope * (len(y) + i) + 1.96 * s]} for i in range(horizon)]
        out["series"][c] = {"points": len(y), "start": str(d.index[0]), "end": str(d.index[-1]),
                            "typical_interval": str(step), "trend": trend, "slope_per_step": lr.slope,
                            "r_squared": lr.rvalue ** 2, "kendall_tau": tau, "kendall_p": p_tau,
                            "autocorrelation_lag1": ac1,
                            "change_pct_first_to_last": float((y[-1] - y[0]) / abs(y[0]) * 100) if y[0] else None,
                            "forecast": fc,
                            "forecast_note": "Simple linear-trend extrapolation; not reliable for seasonal or irregular data."}
    return out if out["series"] else {"skipped": "Time column found but too few points (need 8+)."}

def advanced_analysis(df):
    nums = numeric_columns(df)
    return _py({"columns_used": nums, "outliers": outliers(df, nums), "anomalies": anomalies(df, nums),
                "clusters": clusters(df, nums), "distributions": distributions(df, nums),
                "time_series": time_series(df, nums),
                "notes": ["Outliers are flagged for review, not removed. They may be genuine.",
                          "Clusters are exploratory; they describe patterns, not causes."]})
