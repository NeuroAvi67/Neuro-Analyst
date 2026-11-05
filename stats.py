"""Statistical engine. Everything is computed with pandas/scipy; nothing is estimated or invented."""
import numpy as np
import pandas as pd
from scipy import stats as sp
from .inspector import infer_type

ALPHA = 0.05

def _py(x):
    if isinstance(x, dict): return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_py(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (pd.Timestamp,)): return str(x)
    return x

def numeric_columns(df):
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.nunique() == len(s) and (s.dropna() % 1 == 0).all() and s.is_monotonic_increasing:
                continue  # looks like an ID column
            if s.nunique() > 1:
                cols.append(c)
    return cols

def categorical_columns(df, max_levels=20):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 2 <= df[c].nunique() <= max_levels]

def describe(df, cols):
    res = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if x.empty: continue
        mode = x.mode()
        res[c] = {"count": len(x), "mean": x.mean(), "median": x.median(),
                  "mode": mode.iloc[0] if not mode.empty else None,
                  "variance": x.var(), "std": x.std(), "min": x.min(), "max": x.max(),
                  "range": x.max() - x.min(),
                  "percentiles": {p: x.quantile(p / 100) for p in (1, 5, 25, 50, 75, 95, 99)},
                  "skewness": sp.skew(x) if len(x) > 2 else None,
                  "kurtosis": sp.kurtosis(x) if len(x) > 3 else None}
    return res

def frequencies(df, cols, top=10):
    res = {}
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        total = int(vc.sum())
        res[c] = [{"value": str(k), "count": int(v), "pct": round(v / total * 100, 2)} for k, v in vc.head(top).items()]
    return res

def _strength(r):
    a = abs(r)
    return "very strong" if a >= .8 else "strong" if a >= .6 else "moderate" if a >= .4 else "weak" if a >= .2 else "negligible"

def correlations(df, cols, min_n=5):
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = df[[a, b]].dropna()
            if len(d) < min_n or d[a].nunique() < 2 or d[b].nunique() < 2: continue
            r, p = sp.pearsonr(d[a], d[b])
            rho, p_s = sp.spearmanr(d[a], d[b])
            pairs.append({"x": a, "y": b, "n": len(d), "pearson_r": r, "pearson_p": p,
                          "spearman_rho": rho, "spearman_p": p_s, "strength": _strength(r),
                          "direction": "positive" if r > 0 else "negative",
                          "significant": bool(p < ALPHA)})
    pairs.sort(key=lambda d: -abs(d["pearson_r"]))
    return pairs

def regression(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 3 or d[x].nunique() < 2:
        return {"error": "Not enough data for regression."}
    r = sp.linregress(d[x], d[y])
    return {"x": x, "y": y, "n": len(d), "slope": r.slope, "intercept": r.intercept,
            "r_squared": r.rvalue ** 2, "p_value": r.pvalue, "std_err": r.stderr,
            "significant": bool(r.pvalue < ALPHA)}

def normality(df, cols):
    res = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if len(x) < 3 or x.nunique() < 3: continue
        s = x.sample(5000, random_state=0) if len(x) > 5000 else x
        stat, p = sp.shapiro(s)
        res[c] = {"test": "Shapiro-Wilk", "statistic": stat, "p_value": p, "n": len(s),
                  "looks_normal": bool(p >= ALPHA)}
    return res

def group_tests(df, num_cols, cat_cols, max_cats=3, max_nums=5):
    res = []
    for cat in cat_cols[:max_cats]:
        for num in num_cols[:max_nums]:
            d = df[[cat, num]].dropna()
            groups = [g[num].values for _, g in d.groupby(cat) if len(g) >= 2]
            if len(groups) < 2: continue
            if len(groups) == 2:
                stat, p = sp.ttest_ind(groups[0], groups[1], equal_var=False)
                test = "Welch t-test"
            else:
                stat, p = sp.f_oneway(*groups)
                test = "One-way ANOVA"
            res.append({"group_column": cat, "value_column": num, "test": test, "groups": len(groups),
                        "statistic": stat, "p_value": p, "significant": bool(p < ALPHA),
                        "group_means": {str(k): float(g[num].mean()) for k, g in d.groupby(cat)}})
    return res

def chi_square(df, cat_cols, max_cats=4):
    res = []
    cols = cat_cols[:max_cats]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ct = pd.crosstab(df[a], df[b])
            if ct.shape[0] < 2 or ct.shape[1] < 2: continue
            chi, p, dof, _ = sp.chi2_contingency(ct)
            res.append({"x": a, "y": b, "chi2": chi, "dof": dof, "p_value": p, "significant": bool(p < ALPHA),
                        "note": "Expected counts may be too small for reliable results." if (_ < 5).any() else None})
    return res

def full_analysis(df: pd.DataFrame) -> dict:
    nums, cats = numeric_columns(df), categorical_columns(df)
    corr = correlations(df, nums)
    out = {
        "columns_used": {"numeric": nums, "categorical": cats},
        "descriptive": describe(df, nums),
        "frequencies": frequencies(df, cats),
        "correlations": corr,
        "covariance": df[nums].cov().to_dict() if len(nums) >= 2 else {},
        "regressions": [regression(df, c["x"], c["y"]) for c in corr[:3]],
        "normality": normality(df, nums),
        "group_tests": group_tests(df, nums, cats),
        "chi_square": chi_square(df, cats),
        "notes": ["Correlation does not imply causation.",
                  f"Significance threshold: alpha = {ALPHA}. With many tests, some 'significant' results can occur by chance."],
    }
    return _py(out)
