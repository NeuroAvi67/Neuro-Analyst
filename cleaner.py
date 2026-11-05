"""Non-destructive cleaning. Works on a copy; every change is logged with counts."""
import numpy as np
import pandas as pd
from .inspector import infer_type

DEFAULTS = {"strip_whitespace": True, "normalize_case": True, "convert_types": True,
            "drop_duplicates": True, "missing": "fill"}  # fill | drop_rows | none
MAX_FILL_MISSING_PCT = 50

def _is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))

def _summary(df):
    return {"rows": len(df), "columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())}

def clean(df: pd.DataFrame, opts: dict | None = None):
    o = {**DEFAULTS, **(opts or {})}
    out, log = df.copy(), []
    before = _summary(df)

    if o["strip_whitespace"]:
        for c in out.columns:
            if _is_text(out[c]):
                s = out[c]
                m = s.notna()
                stripped = s[m].astype(str).str.strip()
                n = int((s[m].astype(str) != stripped).sum())
                if n:
                    out.loc[m, c] = stripped
                    log.append({"action": "strip_whitespace", "column": c, "changed": n})

    if o["convert_types"]:
        for c in out.columns:
            t = infer_type(out[c])
            if t == "numeric (stored as text)":
                raw = out[c]
                conv = pd.to_numeric(raw.astype(str).str.replace(",", "", regex=False), errors="coerce")
                conv[raw.isna()] = np.nan
                invalid = int(raw.notna().sum() - conv.notna().sum())
                out[c] = conv
                log.append({"action": "convert_to_numeric", "column": c, "invalid_set_to_missing": invalid})
            elif t == "datetime (stored as text)":
                raw = out[c]
                conv = pd.to_datetime(raw, errors="coerce", format="mixed")
                invalid = int(raw.notna().sum() - conv.notna().sum())
                out[c] = conv
                log.append({"action": "convert_to_datetime", "column": c, "invalid_set_to_missing": invalid})

    if o["normalize_case"]:
        for c in out.columns:
            if _is_text(out[c]) and infer_type(out[c]) == "categorical":
                s = out[c].dropna().astype(str)
                key = s.str.lower()
                if key.nunique() < s.nunique():
                    canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
                    mapped = out[c].map(lambda v: canon.get(str(v).lower(), v) if pd.notna(v) else v)
                    n = int((mapped.astype(str) != out[c].astype(str)).sum())
                    out[c] = mapped
                    log.append({"action": "normalize_case", "column": c, "changed": n})

    if o["drop_duplicates"]:
        n = int(out.duplicated().sum())
        if n:
            out = out.drop_duplicates().reset_index(drop=True)
            log.append({"action": "drop_duplicates", "rows_removed": n})

    if o["missing"] == "drop_rows":
        n = int(out.isna().any(axis=1).sum())
        if n:
            out = out.dropna().reset_index(drop=True)
            log.append({"action": "drop_rows_with_missing", "rows_removed": n})
    elif o["missing"] == "fill":
        for c in out.columns:
            miss = int(out[c].isna().sum())
            if not miss:
                continue
            pct = miss / len(out) * 100
            if pct > MAX_FILL_MISSING_PCT or pd.api.types.is_datetime64_any_dtype(out[c]):
                log.append({"action": "skipped_fill", "column": c, "reason": f"{pct:.0f}% missing or datetime - left as is"})
                continue
            if pd.api.types.is_numeric_dtype(out[c]):
                val, how = out[c].median(), "median"
            else:
                mode = out[c].mode(dropna=True)
                if mode.empty:
                    continue
                val, how = mode.iloc[0], "mode"
            out[c] = out[c].fillna(val)
            log.append({"action": f"fill_{how}", "column": c, "filled": miss, "value": None if pd.isna(val) else str(val)})

    return out, {"options": o, "log": log, "before": before, "after": _summary(out)}
