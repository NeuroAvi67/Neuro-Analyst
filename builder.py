"""Report builder. Produces a structured report (sections of blocks) from analysis results,
then renders it to Markdown, HTML or PDF. Only relevant sections are included."""
import re, io, html, json, time
from ..ai import interpreter as I

INCLUDE = {
    "scientific": ["summary", "objective", "dataset", "method", "quality", "stats", "visuals", "insights", "interpretation", "limitations", "recommendations", "conclusion"],
    "executive": ["summary", "insights", "visuals", "recommendations", "limitations", "conclusion"],
    "business": ["summary", "dataset", "quality", "insights", "visuals", "recommendations", "limitations", "conclusion"],
    "statistical": ["summary", "dataset", "method", "quality", "stats", "visuals", "insights", "interpretation", "limitations", "conclusion"],
    "top_findings": ["summary", "insights", "limitations"],
    "general": ["summary", "objective", "dataset", "quality", "stats", "visuals", "insights", "interpretation", "limitations", "recommendations", "conclusion"],
}
TITLES = {"summary": "Executive Summary", "objective": "Research Objective", "dataset": "Dataset Description",
          "method": "Methodology", "quality": "Data Quality Assessment", "stats": "Statistical Findings",
          "visuals": "Visualisations", "insights": "Important Insights", "interpretation": "Interpretation",
          "limitations": "Limitations", "recommendations": "Recommendations", "conclusion": "Conclusion"}

def md_to_blocks(md):
    blocks, para, lst, kind = [], [], [], None
    def flush():
        nonlocal para, lst, kind
        if para: blocks.append(("p", " ".join(para))); para = []
        if lst: blocks.append((kind, lst)); lst = []; kind = None
    for line in md.splitlines():
        s = line.strip()
        if not s: flush(); continue
        if s.startswith("#"): flush(); continue
        m_ul, m_ol = re.match(r"^[-*]\s+(.*)", s), re.match(r"^\d+\.\s+(.*)", s)
        if m_ul or m_ol:
            k = "ul" if m_ul else "ol"
            if para: flush()
            if kind and kind != k: flush()
            kind = k; lst.append((m_ul or m_ol).group(1)); continue
        if lst: flush()
        para.append(s)
    flush(); return blocks

def _section(md, name):
    m = re.search(rf"##\s*{name}\s*\n(.*?)(?=\n##\s|\Z)", md, re.S | re.I)
    return m.group(1).strip() if m else ""

def _fmt(v, d=3):
    return "" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))

def _pf(p):
    return "" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")

def build_report(result, rtype=None, title=None):
    rtype = rtype if rtype in INCLUDE else result["plan"]["report_type"]
    info, st, adv, facts = result["inspection"], result["stats"], result["advanced"], result["facts"]
    cr, plan = result["cleaning"], result["plan"]
    ranked = [f for f in I.rank_facts(facts, plan) if f["category"] != "dataset"]
    top_n = plan.get("top_n", 10)
    name = result.get("dataset_name", "Dataset")
    sections = []
    def add(key, blocks):
        if key in INCLUDE[rtype] and blocks: sections.append({"id": key, "title": TITLES[key], "blocks": blocks})
    ds_line = f"The dataset '{name}' contains {info['rows']} rows and {info['columns']} columns."
    summ = [("p", ds_line + (" The most notable results are:" if ranked else " No statistically notable patterns were found."))]
    if ranked: summ.append(("ul", [f["text"] for f in ranked[:3]]))
    add("summary", summ)
    if (result.get("instruction") or "").strip():
        add("objective", [("p", f"The analysis was requested as: \"{result['instruction'].strip()}\".")])
    rows = [["Column", "Type", "Missing %", "Unique"]] + [[c["name"], c["type"], _fmt(c["missing_pct"]), str(c["unique"])] for c in info["column_info"][:25]]
    ds = [("p", ds_line), ("table", rows)]
    if len(info["column_info"]) > 25: ds.append(("p", f"Showing 25 of {len(info['column_info'])} columns."))
    add("dataset", ds)
    meth = ["Data was loaded and inspected; the original file was preserved and all cleaning was applied to a copy.",
            "Descriptive statistics: mean, median, mode, variance, standard deviation, range, percentiles, skewness, kurtosis."]
    if st.get("correlations"): meth.append("Pearson and Spearman correlation with significance tests; covariance; simple linear regression on the strongest pairs.")
    if st.get("normality"): meth.append("Shapiro-Wilk normality tests.")
    if st.get("group_tests") or st.get("chi_square"): meth.append("Group comparison with Welch t-test or one-way ANOVA; chi-square tests of association (alpha = 0.05).")
    if adv.get("outliers") is not None: meth.append("Outliers by the 1.5x IQR rule and |z| > 3; multivariate anomalies by Mahalanobis distance.")
    if (adv.get("clusters") or {}).get("k"): meth.append("k-means clustering on standardised variables; number of clusters chosen by silhouette score.")
    if (adv.get("time_series") or {}).get("series"): meth.append("Time-series trend tested with Kendall's tau; simple linear-trend forecast.")
    meth.append("All calculations were performed in Python (pandas, NumPy, SciPy). " + (f"Wording was produced by {result['mode']} and every number was checked against the calculations." if result["mode"].startswith("AI") else "Wording was generated by rules from the calculated results (offline mode)."))
    add("method", [("ul", meth)])
    qb = [("p", f"Before cleaning: {cr['before']['rows']} rows, {cr['before']['missing_cells']} missing cells, {cr['before']['duplicate_rows']} duplicate rows. After cleaning: {cr['after']['rows']} rows, {cr['after']['missing_cells']} missing cells, {cr['after']['duplicate_rows']} duplicate rows.")]
    if info["issues"]: qb.append(("table", [["Column", "Issue", "Detail"]] + [[i["column"] or "(dataset)", i["issue"], i["detail"]] for i in info["issues"][:15]]))
    if cr["log"]:
        acts = []
        for l in cr["log"]:
            extra = ", ".join(f"{k}={v}" for k, v in l.items() if k not in ("action", "column"))
            acts.append(f"{l['action'].replace('_', ' ')}" + (f" on {l['column']}" if l.get("column") else "") + (f" ({extra})" if extra else ""))
        qb += [("p", "Cleaning actions applied to the working copy:"), ("ul", acts)]
    add("quality", qb if (info["issues"] or cr["log"]) else [])
    sb = []
    d = st.get("descriptive") or {}
    if d: sb.append(("table", [["Variable", "Mean", "Median", "Std dev", "Min", "Max"]] + [[c, _fmt(v["mean"]), _fmt(v["median"]), _fmt(v["std"]), _fmt(v["min"]), _fmt(v["max"])] for c, v in list(d.items())[:10]]))
    cs = (st.get("correlations") or [])[:10]
    if cs: sb += [("p", "Strongest pairwise correlations (correlation does not imply causation):"),
                  ("table", [["Pair", "Pearson r", "p-value", "n", "Strength"]] + [[f"{c['x']} - {c['y']}", _fmt(c["pearson_r"]), _pf(c["pearson_p"]), str(c["n"]), c["strength"]] for c in cs])]
    gt = st.get("group_tests") or []
    if gt: sb += [("p", "Group comparisons:"), ("table", [["Variable", "Grouped by", "Test", "p-value", "Significant"]] + [[g["value_column"], g["group_column"], g["test"], _pf(g["p_value"]), "yes" if g["significant"] else "no"] for g in gt])]
    add("stats", sb)
    add("visuals", [("p", "The following charts were generated from the analysed data.")] + [("chart", c["id"], c["title"]) for c in result["charts"]] if result["charts"] else [])
    add("insights", [("ol", [f["text"] for f in ranked[:top_n]])] if ranked else [])
    interp = md_to_blocks(_section(result["narrative"], "Interpretation")) if result["mode"].startswith("AI") else []
    if not interp and ranked:
        cats = {f["category"] for f in ranked}
        t = []
        if "correlation" in cats or "regression" in cats: t.append("Variables that move together may be linked, but these results alone cannot show that one causes the other.")
        if "group_test" in cats or "association" in cats: t.append("Differences between groups are statistically detectable; whether they are practically important depends on their size and context.")
        if "trend" in cats: t.append("Trends describe the direction of change over the observed period and should not be assumed to continue.")
        if "outliers" in cats or "anomalies" in cats: t.append("Flagged unusual values should be checked at the source before being excluded or acted on.")
        interp = [("ul", t)] if t else []
    add("interpretation", interp)
    add("limitations", [("ul", I.limitations(info, st, adv))])
    add("recommendations", [("ul", I.next_steps(facts))])
    lead = ranked[0]["text"] if ranked else "no statistically notable patterns"
    add("conclusion", [("p", f"The analysis of {info['rows']} rows highlighted {len(ranked)} notable finding(s), led by: {lead} These results describe patterns in this dataset and should be validated before decisions are made.")])
    if rtype == "top_findings" and sections:
        pass
    return {"title": title or f"Analysis Report: {name}", "type": rtype, "generated": time.strftime("%Y-%m-%d %H:%M"),
            "mode": result["mode"], "sections": sections}

def build_comparison_report(cmp):
    secs = [{"id": "summary", "title": "Executive Summary", "blocks": [("p", cmp["summary"])] + ([("ul", cmp["findings"][:5])] if cmp["findings"] else [])},
            {"id": "match", "title": "Column Matching", "blocks": [("table", [["Dataset A", "Dataset B", "Match"]] + [[m["a"], m["b"], m["how"]] for m in cmp["matches"]] )] if cmp["matches"] else [("p", "No columns matched.")]}]
    if cmp["only_a"] or cmp["only_b"]:
        secs.append({"id": "diff", "title": "Differences in Structure", "blocks": [("ul", [f"Only in {cmp['names'][0]}: {', '.join(cmp['only_a']) or 'none'}", f"Only in {cmp['names'][1]}: {', '.join(cmp['only_b']) or 'none'}"])]})
    if cmp["numeric"]:
        secs.append({"id": "numeric", "title": "Statistical Comparison", "blocks": [("table", [["Column", "Mean A", "Mean B", "Cohen's d", "Welch p", "KS p", "Different?"]] + [[n["column"], _fmt(n["mean_a"]), _fmt(n["mean_b"]), _fmt(n["cohens_d"]), _pf(n["welch_p"]), _pf(n["ks_p"]), "yes" if n["significant"] else "no"] for n in cmp["numeric"]])]})
    if cmp["trends"]:
        secs.append({"id": "trends", "title": "Trend Comparison", "blocks": [("ul", cmp["trends"])]})
    if cmp["relationships"]:
        secs.append({"id": "rel", "title": "Relationship Comparison", "blocks": [("ul", cmp["relationships"])]})
    secs.append({"id": "assume", "title": "Assumptions and Limitations", "blocks": [("ul", cmp["assumptions"])]})
    return {"title": f"Comparison: {cmp['names'][0]} vs {cmp['names'][1]}", "type": "comparison", "generated": time.strftime("%Y-%m-%d %H:%M"), "mode": "statistical", "sections": secs}

# ---------- renderers ----------
def to_markdown(rep, charts=None):
    out = [f"# {rep['title']}", f"*Generated {rep['generated']} - {rep['mode']}*", ""]
    for n, s in enumerate(rep["sections"], 1):
        out += [f"## {n}. {s['title']}", ""]
        for b in s["blocks"]:
            if b[0] == "p": out += [b[1], ""]
            elif b[0] == "ul": out += [f"- {x}" for x in b[1]] + [""]
            elif b[0] == "ol": out += [f"{i}. {x}" for i, x in enumerate(b[1], 1)] + [""]
            elif b[0] == "table":
                r = b[1]; out += ["| " + " | ".join(r[0]) + " |", "|" + "---|" * len(r[0])] + ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in r[1:]] + [""]
            elif b[0] == "chart": out += [f"- Chart: {b[2]} (interactive in the HTML export)"]
        out.append("")
    return "\n".join(out)

CSS = """:root{--bg:#0b1020;--fg:#e6e9f5;--mut:#9aa3c4;--card:#131a33;--ac:#6c8cff;--bd:#25305a}
@media (prefers-color-scheme: light){:root{--bg:#fff;--fg:#14182b;--mut:#555d7a;--card:#f4f6fc;--ac:#3b5bdb;--bd:#d8def0}}
*{box-sizing:border-box}body{font:16px/1.6 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:20px;max-width:860px;margin:auto}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.15rem;color:var(--ac);margin-top:1.6em;border-bottom:1px solid var(--bd);padding-bottom:.2em}
.meta{color:var(--mut);font-size:.85rem}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{border:1px solid var(--bd);padding:6px 8px;text-align:left}
th{background:var(--card)}.scroll{overflow-x:auto}.chart{height:340px;margin:12px 0}"""

def to_html(rep, charts=None):
    e = html.escape
    parts, cmap, used = [f"<h1>{e(rep['title'])}</h1><div class='meta'>Generated {e(rep['generated'])} · {e(rep['mode'])}</div>"], {c["id"]: c for c in (charts or [])}, []
    for n, s in enumerate(rep["sections"], 1):
        parts.append(f"<h2>{n}. {e(s['title'])}</h2>")
        for b in s["blocks"]:
            if b[0] == "p": parts.append(f"<p>{e(b[1])}</p>")
            elif b[0] in ("ul", "ol"): parts.append(f"<{b[0]}>" + "".join(f"<li>{e(x)}</li>" for x in b[1]) + f"</{b[0]}>")
            elif b[0] == "table":
                r = b[1]; parts.append("<div class='scroll'><table><tr>" + "".join(f"<th>{e(c)}</th>" for c in r[0]) + "</tr>" + "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in row) + "</tr>" for row in r[1:]) + "</table></div>")
            elif b[0] == "chart" and b[1] in cmap:
                used.append(b[1]); parts.append(f"<div class='chart' id='ch_{b[1]}'></div>")
    script = ""
    if used:
        figs = {k: cmap[k]["figure"] for k in used}
        data = json.dumps(figs).replace("</", "<\\/")
        script = ("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script><script>const F=" + data +
                  ";for(const k in F){Plotly.newPlot('ch_'+k,F[k].data,F[k].layout,{responsive:true,displaylogo:false})}</script>")
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{e(rep['title'])}</title><style>{CSS}</style></head><body>{''.join(parts)}<p class='meta'>Generated by NEURO ANALYST. Charts need an internet connection to load plotly.js.</p>{script}</body></html>"

def _pdf_safe(t):
    return str(t).encode("cp1252", "replace").decode("cp1252").replace("?", "?") if t is not None else ""

def figure_png(fig, w=6.4, h=3.4):
    """Render a Plotly-JSON figure to PNG with matplotlib (optional dependency; returns None if unavailable)."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return None
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    lay, tr = fig["layout"], fig["data"]
    ttl = lay.get("title", {}).get("text", "")
    try:
        for t in tr:
            ty = t.get("type")
            if ty == "histogram": ax.hist([v for v in t["x"] if v is not None], bins=t.get("nbinsx", 30), color="#6c8cff")
            elif ty == "scatter":
                if t.get("mode") == "lines": ax.plot(t["x"], t["y"], lw=1.4, label=t.get("name"))
                else: ax.scatter(t["x"], t["y"], s=8, alpha=.7, label=t.get("name"))
            elif ty == "bar": ax.bar(t["x"], t["y"], color="#6c8cff"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            elif ty == "box": ax.boxplot([v for v in t["y"] if v is not None], positions=[len(ax.get_xticklabels()) + 1], widths=.5); ax.set_xticks(range(1, len(ax.get_xticks()) + 1))
            elif ty == "pie": ax.pie(t["values"], labels=t["labels"], autopct="%1.0f%%"); ax.axis("equal")
            elif ty == "heatmap":
                im = ax.imshow(t["z"], cmap="RdBu_r" if t.get("zmin") == -1 else "viridis", vmin=t.get("zmin"), vmax=t.get("zmax"), aspect="auto")
                ax.set_xticks(range(len(t["x"]))); ax.set_xticklabels(t["x"], rotation=45, ha="right"); ax.set_yticks(range(len(t["y"]))); ax.set_yticklabels(t["y"]); f.colorbar(im)
        if any(t.get("type") == "box" for t in tr):
            ax.set_xticklabels([t.get("name") for t in tr if t.get("type") == "box"])
        ax.set_title(ttl, fontsize=10)
        xl, yl = lay.get("xaxis", {}).get("title", {}).get("text", ""), lay.get("yaxis", {}).get("title", {}).get("text", "")
        if xl and tr[0].get("type") not in ("pie", "heatmap"): ax.set_xlabel(xl, fontsize=8)
        if yl and tr[0].get("type") not in ("pie", "heatmap"): ax.set_ylabel(yl, fontsize=8)
        if len([t for t in tr if t.get("name")]) > 1 and tr[0].get("type") in ("scatter",): ax.legend(fontsize=7)
        f.tight_layout(); buf = io.BytesIO(); f.savefig(buf, format="png"); return buf.getvalue()
    except Exception:
        return None
    finally:
        plt.close(f)

def to_pdf(rep, charts=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#1b2559"))
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#3b5bdb"), spaceBefore=12)
    P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=10, leading=14)
    SM = ParagraphStyle("sm", parent=P, fontSize=8, textColor=colors.grey)
    CELL = ParagraphStyle("cell", parent=P, fontSize=7.5, leading=9)
    esc = lambda t: html.escape(_pdf_safe(t))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm, title=_pdf_safe(rep["title"]), author="NEURO ANALYST")
    cmap, story = {c["id"]: c for c in (charts or [])}, [Paragraph(esc(rep["title"]), H1), Paragraph(esc(f"Generated {rep['generated']} - {rep['mode']}"), SM), Spacer(1, 8)]
    for n, s in enumerate(rep["sections"], 1):
        story.append(Paragraph(esc(f"{n}. {s['title']}"), H2))
        for b in s["blocks"]:
            if b[0] == "p": story.append(Paragraph(esc(b[1]), P))
            elif b[0] in ("ul", "ol"):
                for i, x in enumerate(b[1], 1): story.append(Paragraph(esc(x), P, bulletText="•" if b[0] == "ul" else f"{i}."))
            elif b[0] == "table":
                r = b[1]; data = [[Paragraph(f"<b>{esc(c)}</b>" if k == 0 else esc(c), CELL) for c in row] for k, row in enumerate(r)]
                t = Table(data, repeatRows=1, colWidths=[17*cm / len(r[0])] * len(r[0]))
                t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.lightgrey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1fb")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story += [Spacer(1, 4), t, Spacer(1, 6)]
            elif b[0] == "chart" and b[1] in cmap:
                png = figure_png(cmap[b[1]]["figure"])
                if png: story += [KeepTogether([Image(io.BytesIO(png), width=15*cm, height=15*cm*3.4/6.4)]), Spacer(1, 6)]
                else: story.append(Paragraph(esc(f"[Chart: {b[2]} - install matplotlib to include charts in PDF]"), SM))
    story += [Spacer(1, 14), Paragraph("Generated by NEURO ANALYST. Numbers come from Python calculations; see Methodology and Limitations.", SM)]
    doc.build(story); return buf.getvalue()
