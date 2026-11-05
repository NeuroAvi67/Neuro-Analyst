"""Chart builder. Emits plain Plotly figure JSON ({data, layout}); the browser renders it with plotly.js.
No server-side plotly dependency is needed."""
import numpy as np
import pandas as pd
from .stats import numeric_columns, categorical_columns
from .advanced import find_time_column

KINDS = ["bar", "line", "pie", "scatter", "histogram", "box", "heatmap", "correlation", "timeseries"]
PALETTE = ["#6c8cff", "#9b7bff", "#35d0ba", "#ffb454", "#ff6b8b", "#7ad1ff"]

def L(a):
    out = []
    for v in (a.tolist() if hasattr(a, "tolist") else list(a)):
        if isinstance(v, float) and not np.isfinite(v): v = None
        out.append(str(v) if isinstance(v, pd.Timestamp) else v)
    return out

def _lbl(col, units):
    u = (units or {}).get(col)
    return f"{col} ({u})" if u else str(col)

def _fig(data, title, x=None, y=None, **layout):
    lay = {"title": {"text": title, "font": {"size": 14}}, "colorway": PALETTE, "paper_bgcolor": "rgba(0,0,0,0)",
           "plot_bgcolor": "rgba(0,0,0,0)", "font": {"size": 11, "color": "#cfd6ee"}, "autosize": True,
           "margin": {"l": 48, "r": 14, "t": 48, "b": 48}, "legend": {"orientation": "h", "y": -0.28},
           "xaxis": {"title": {"text": x or ""}, "gridcolor": "rgba(255,255,255,.08)"},
           "yaxis": {"title": {"text": y or ""}, "gridcolor": "rgba(255,255,255,.08)"}}
    lay.update(layout)
    return {"data": data, "layout": lay}

def _need(df, *cols):
    for c in cols:
        if c is not None and c not in df.columns: raise ValueError(f"Column '{c}' not found.")

def make_chart(df, kind, x=None, y=None, color=None, units=None):
    if kind not in KINDS: raise ValueError(f"Unknown chart type '{kind}'.")
    _need(df, x, y, color)
    if df.empty: raise ValueError("Dataset is empty.")
    if kind == "correlation":
        cols = numeric_columns(df)
        if len(cols) < 2: raise ValueError("Need at least 2 numeric columns.")
        m = df[cols].corr().round(2)
        return _fig([{"type": "heatmap", "z": L(m.values), "x": cols, "y": cols, "zmin": -1, "zmax": 1,
                      "colorscale": "RdBu", "reversescale": True, "text": L(m.values), "texttemplate": "%{text}",
                      "colorbar": {"title": {"text": "r"}}}], "Correlation matrix (Pearson r)")
    if kind == "histogram":
        if not x: raise ValueError("x column required.")
        return _fig([{"type": "histogram", "x": L(df[x].dropna()), "nbinsx": 30, "name": x}], f"Distribution of {x}", _lbl(x, units), "Count")
    if kind == "box":
        if not y: raise ValueError("y (numeric) column required.")
        if x: data = [{"type": "box", "y": L(d[y].dropna()), "name": str(g)} for g, d in df.groupby(x)]
        else: data = [{"type": "box", "y": L(df[y].dropna()), "name": y}]
        return _fig(data, f"{y}" + (f" by {x}" if x else ""), x, _lbl(y, units))
    if kind == "pie":
        if not x: raise ValueError("x (category) column required.")
        vc = df[x].value_counts().head(8)
        return _fig([{"type": "pie", "labels": L(vc.index.astype(str)), "values": L(vc.values), "hole": .35}], f"Share of {x} (count)")
    if kind == "bar":
        if not x: raise ValueError("x column required.")
        if y:
            g = df.groupby(x)[y].mean().sort_values(ascending=False).head(15); yl, t = f"Mean {_lbl(y, units)}", f"Mean {y} by {x}"
        else:
            g = df[x].value_counts().head(15); yl, t = "Count", f"Frequency of {x}"
        return _fig([{"type": "bar", "x": L(g.index.astype(str)), "y": L(g.values)}], t, x, yl)
    if kind == "scatter":
        if not (x and y): raise ValueError("x and y columns required.")
        d = df[[x, y] + ([color] if color else [])].dropna()
        d = d.sample(3000, random_state=0) if len(d) > 3000 else d
        data = []
        if color: data += [{"type": "scatter", "mode": "markers", "x": L(dd[x]), "y": L(dd[y]), "name": str(g)} for g, dd in d.groupby(color)]
        else: data.append({"type": "scatter", "mode": "markers", "x": L(d[x]), "y": L(d[y]), "name": "data", "marker": {"opacity": .7}})
        if len(d) > 2 and d[x].nunique() > 1 and pd.api.types.is_numeric_dtype(d[x]) and pd.api.types.is_numeric_dtype(d[y]):
            k, b = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
            xs = [float(d[x].min()), float(d[x].max())]
            data.append({"type": "scatter", "mode": "lines", "x": xs, "y": [k * v + b for v in xs], "name": "linear fit"})
        return _fig(data, f"{y} vs {x}", _lbl(x, units), _lbl(y, units))
    if kind in ("line", "timeseries"):
        if not (x and y): raise ValueError("x and y columns required.")
        d = df[[x, y]].dropna()
        xv = pd.to_datetime(d[x], errors="coerce", format="mixed") if kind == "timeseries" else d[x]
        d = pd.DataFrame({x: xv, y: d[y]}).dropna().sort_values(x)
        return _fig([{"type": "scatter", "mode": "lines", "x": L(d[x]), "y": L(d[y]), "name": y}], f"{y} over {x}", x, _lbl(y, units))
    if kind == "heatmap":
        if not (x and y): raise ValueError("x and y category columns required.")
        ct = pd.crosstab(df[y], df[x])
        return _fig([{"type": "heatmap", "z": L(ct.values), "x": L(ct.columns.astype(str)), "y": L(ct.index.astype(str)),
                      "colorscale": "Viridis", "colorbar": {"title": {"text": "Count"}}}], f"{y} × {x} (counts)", x, y)

def auto_charts(df, st, adv, limit=8):
    nums, cats = numeric_columns(df), categorical_columns(df)
    out = []
    def add(kind, title, **kw):
        try: out.append({"id": f"C{len(out)+1}", "kind": kind, "title": title, "figure": make_chart(df, kind, **kw)})
        except Exception: pass
    tc, _ = find_time_column(df)
    if len(nums) >= 3: add("correlation", "Correlation matrix")
    for c in (st.get("correlations") or [])[:1]: add("scatter", f"{c['y']} vs {c['x']}", x=c["x"], y=c["y"])
    for c in nums[:2]: add("histogram", f"Distribution of {c}", x=c)
    if cats and nums: add("box", f"{nums[0]} by {cats[0]}", x=cats[0], y=nums[0])
    if cats: add("bar", f"Frequency of {cats[0]}", x=cats[0])
    if tc:
        for c in nums[:2]: add("timeseries", f"{c} over time", x=tc, y=c)
    return out[:limit]
