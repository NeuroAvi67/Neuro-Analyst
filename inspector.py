"""Dataset inspection: types, missing values, duplicates, format issues, suspicious values.
Every number returned is computed from the data."""
import numpy as np
import pandas as pd

def infer_type(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    as_str = s.astype(str).str.strip()
    num = pd.to_numeric(as_str.str.replace(",", ""), errors="coerce")
    if num.notna().mean() > 0.9:
        return "numeric (stored as text)"
    dt = pd.to_datetime(as_str.head(500), errors="coerce", format="mixed")
    if dt.notna().mean() > 0.9:
        return "datetime (stored as text)"
    if as_str.nunique() / max(len(as_str), 1) < 0.5 or as_str.nunique() <= 20:
        return "categorical"
    return "text"

def inspect(df: pd.DataFrame) -> dict:
    n = len(df)
    columns, issues = [], []
    for c in df.columns:
        s = df[c]
        t = infer_type(s)
        miss = int(s.isna().sum())
        col = {"name": c, "type": t, "missing": miss,
               "missing_pct": round(miss / n * 100, 2) if n else 0,
               "unique": int(s.nunique(dropna=True))}
        if miss:
            issues.append({"column": c, "issue": "missing_values", "detail": f"{miss} missing ({col['missing_pct']}%)"})
        if t.endswith("(stored as text)"):
            issues.append({"column": c, "issue": "wrong_type", "detail": f"Looks like {t.split(' ')[0]} but stored as text"})
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            st = s.dropna().astype(str)
            if (st != st.str.strip()).any():
                issues.append({"column": c, "issue": "whitespace", "detail": "Leading/trailing spaces found"})
            if t == "categorical":
                stripped = st.str.strip()
                if stripped.str.lower().nunique() < stripped.nunique():
                    issues.append({"column": c, "issue": "inconsistent_case", "detail": "Same values differ only by letter case"})
        if t == "numeric":
            x = s.dropna().astype(float)
            q1, q3 = x.quantile([.25, .75])
            iqr = q3 - q1
            if iqr > 0:
                out = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
                if out:
                    issues.append({"column": c, "issue": "extreme_values", "detail": f"{out} extreme value(s) beyond 3×IQR - verify they are correct"})
            if np.isinf(x).any():
                issues.append({"column": c, "issue": "infinite_values", "detail": "Infinite values present"})
        columns.append(col)
    dups = int(df.duplicated().sum())
    if dups:
        issues.append({"column": None, "issue": "duplicates", "detail": f"{dups} fully duplicated rows"})
    return {
        "rows": n, "columns": df.shape[1],
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "duplicate_rows": dups,
        "total_missing_cells": int(df.isna().sum().sum()),
        "column_info": columns, "issues": issues,
        "preview": df.head(10).astype(object).where(df.head(10).notna(), None).to_dict(orient="records"),
    }
