# NEURO ANALYST — Turn Raw Data Into Intelligence.

Mobile-first AI data analysis platform: upload data → describe what you want → verified statistics, charts and a professional report (PDF / HTML / Markdown).

## Run locally (₹0)
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m backend.app          # open http://localhost:5000 on your computer
```
To use it on your phone on the same Wi-Fi, open `http://<your-computer-IP>:5000`.
(Installing as a PWA on a phone needs HTTPS, which hosted deployments provide.)

Run the tests: `python -m unittest discover tests`

## How it works
Upload → inspect → clean (on a copy) → plan → statistics → advanced analysis → charts → interpretation → **verification** → report.
Every number comes from Python (pandas/SciPy). The wording layer (rules, or an AI model) may only explain computed "facts";
any number in AI text that is not in the facts causes that text to be rejected and the verified offline summary is shown instead.

## AI providers (optional — offline statistics always works)
Set environment variables on the server (see `.env.example`): `NEURO_AI_PROVIDER` = offline | groq | gemini | openrouter | ollama,
plus the matching `*_API_KEY`. Ollama runs models locally for free (needs a capable computer). Keys never reach the browser.
A chatbot subscription is not API access; free API tiers have rate limits and can change — check each provider's current terms.
Default model names are in `backend/ai/providers.py` and can be overridden with `*_MODEL` variables.

## Deploy on a free tier
Files included: `Procfile`, `render.yaml`, `Dockerfile`.
1. Push this folder to a GitHub repository.
2. Create a web service on a free host (e.g. Render: "New → Blueprint" using `render.yaml`; or a Docker-based host using the `Dockerfile`).
3. Add any AI keys as environment variables in the host dashboard (never in code).
Keep `--workers 1` (analysis jobs are tracked in memory). Free tiers typically sleep when idle, have limited RAM/CPU, and often use
temporary disk that is wiped on restart — **download your reports**. Verify each host's current free-tier terms before relying on it.

## Limits and safety
25 MB per file · 500,000 rows · 60 POST requests/min per IP · files are parsed as data only (never executed) · ids validated against path tricks ·
security headers + CSP · originals preserved · delete any dataset or all data from Settings.
Data is stored unencrypted on the server; whoever runs the server can read it. In AI mode only your instruction, column names and computed summaries are sent to the provider (never raw rows).

## Known limitations
- Charts use plotly.js loaded from cdn.plot.ly (version pinned, cached by the service worker after first load). For fully offline use, download that file into `frontend/` and change the `<script>` tag (and the CSP in `backend/app.py`).
- PDF charts are drawn with matplotlib (included in requirements); HTML exports load plotly.js from the CDN.
- Forecasts are simple linear extrapolations. Clustering and anomaly detection are exploratory.
- No user accounts: anyone who can open the app can see its library. Do not host sensitive data publicly without adding authentication.
