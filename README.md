# Krishi.AI

**An AI-powered digital assistant for farmers.**

Ask a farming question in your own language, find the right crop for your soil,
work out fertilizer doses and irrigation schedules, check the rainfall outlook
for your state, and photograph a leaf to check it for disease — all from one
place.

---

## What it does

| Tool | What it answers | Model |
|---|---|---|
| **Ask AI** | Any farming question, in 13 Indian languages | Curated FAQ, then Google Gemini |
| **Crop Recommendation** | Which crop suits my soil and climate? | RandomForest, 26 crops, 99.8% held-out accuracy |
| **Plant Disease Detection** | What is wrong with this leaf? | Xception fine-tune, 38 PlantVillage classes, 99.7% test accuracy |
| **Fertilizer Plan** | How much N, P and K, and when? | Gradient boosting + classifier, ICAR-based |
| **Water Management** | How much water, how often? | RandomForest, FAO-56 based |
| **Rainfall Forecast** | How much rain next month? | Gradient boosting on IMD normals |
| **Market Prices** | What is my crop selling for, and what is my harvest worth? | Live Agmarknet data via data.gov.in |

Crop recommendations show **which of your inputs drove the answer**, disease
results include a **heatmap of what the model looked at**, and every tool is
annotated with the **cropping season** for your region.

Every prediction comes from a real trained model and every price is a real
reported figure. When something is unavailable the app says so — it never
invents a result.

## Architecture

```
Browser  ── static HTML/CSS/ES modules, no build step
   │ JWT bearer token
Backend  ── FastAPI :8000   auth · chat · history · uploads · serves the frontend
   │ HTTP
ML service ─ FastAPI :8100  owns every model file, loads each once at startup
   │
Database ── SQLite by default, PostgreSQL via DATABASE_URL
```

Two services because the ML side pins `scikit-learn==1.6.1` (needed to load the
trained models) and optionally TensorFlow, which the backend has no use for.
The split also means one broken model cannot take the site down.

```
krishi-ai/
├── backend/          FastAPI app + seed_faq.py
│   └── app/          config, db, models, security, routers/, services/
├── ml-service/       FastAPI app
│   ├── app/          registry, knowledge, routers/
│   ├── artifacts/    the .pkl model files (committed); the 245 MB .h5 is not
│   └── training/     the original training scripts and notebook
├── frontend/         10 pages, one stylesheet, shared ES modules
├── docs/             API reference, model notes, progress log
├── .env.example
└── README.md
```

---

## Quick start

**Requires Python 3.11 or newer.** No Node.js, no build step.

### 1. Configuration

```bash
cd krishi-ai
cp .env.example .env
```

It runs as-is. To enable the AI assistant, put a
[Gemini API key](https://aistudio.google.com/apikey) in `.env`:

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
```

Without a key the chatbot still answers curated FAQ questions and says plainly
that the AI is not configured.

### Market prices (optional)

Real mandi rates come from the Government of India's open data platform. Get a
free key at <https://data.gov.in/apis> and add it to `.env`:

```
DATA_GOV_API_KEY=your-key-here
```

Without it the Market page reports itself unavailable and everything else keeps
working. Prices are always the government's own reported figures — Krishi.AI
never estimates or fills in a missing price.

**Register your own key.** The sample key in data.gov.in's documentation is
shared by everyone, so it is frequently rate-limited and caps every response at
**10 rows** regardless of the `limit` asked for. A registered key returns full
pages (106 commodities across 20 states, vs 9 and 2 on the sample key) and is
not contended.

Note that the upstream API's filtering is unreliable above ~300 rows per page —
it starts returning other states alongside the one asked for. Krishi.AI caps the
page size and re-checks every row against the requested filter, so this cannot
surface as wrong prices; see `MAX_UPSTREAM_PAGE` in `backend/app/services/market.py`.

### Gemini model

If you get a 404 naming the model, your key does not have access to it. List
what it can use:

```bash
python -c "from google import genai; print([m.name for m in genai.Client(api_key='YOUR_KEY').models.list()])"
```

Pin a specific model rather than an alias like `gemini-flash-latest` — an alias
can start resolving to a slower or overloaded model without warning.

**Free-tier quota is small.** Google's free tier allows roughly 20 requests per
day *per model*, which is fine for trying the app out but not for real use. When
it runs out the chatbot returns a clear "reached today's usage limit" message and
every prediction tool keeps working. Enable billing on your Google Cloud project
to lift it.

### 2. ML service

```bash
python -m venv .venv-ml
.venv-ml/Scripts/activate          # Windows
# source .venv-ml/bin/activate     # macOS / Linux
pip install -r ml-service/requirements.txt

cd ml-service
python -m uvicorn app.main:app --port 8100
```

Check it: <http://localhost:8100/health> should show all five models
`ready: true`. If `disease` is `false`, its weights or TensorFlow are missing —
see [docs/plant-disease-model.md](docs/plant-disease-model.md).

### 3. Backend

In a second terminal:

```bash
cd krishi-ai
python -m venv .venv-api
.venv-api/Scripts/activate         # Windows
pip install -r backend/requirements.txt

cd backend
python seed_faq.py                 # optional: curated chatbot answers
python -m uvicorn app.main:app --port 8000
```

### 4. Open it

<http://localhost:8000>

There is no sign-up wall — the first visit provisions a guest account
automatically. Create a real account from **Profile** to keep your history
across devices.

API docs (development only): <http://localhost:8000/api/docs>

---

## Languages

The whole interface is available in 13 languages: English, Hindi, Bengali,
Marathi, Telugu, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese
and Urdu (right-to-left). The picker is in the header on every page.

Two catalogues, both static JSON under `frontend/i18n/`:

| File | What it holds |
|---|---|
| `en.json` + `<lang>.json` | 283 interface strings - labels, buttons, states, errors |
| `content.en.json` + `content.<lang>.json` | 135 agronomic advisories returned by the ML service - crop tips, irrigation notes, fertilizer schedules, rainfall advice |

The second one matters: without it the interface would be translated but the
advice a farmer actually acts on would still be English.

The browser never calls a translation API. To regenerate after changing an
English string:

```bash
python tools/extract_content.py          # rebuild content.en.json from the ML tables
python tools/translate_ui.py             # interface strings
python tools/translate_ui.py --content   # agronomic advice
python tools/translate_ui.py --repair    # retry anything that came back in English
python tools/annotate_i18n.py            # tag new markup with data-i18n
```

Needs `GEMINI_API_KEY`. Any key missing from a translation falls back to
English at runtime, so a partial catalogue degrades to a mixed-language page
rather than a broken one.

The language choice is also saved to the user's profile, which is what the
chatbot reads to decide which language to answer in.

---

## Configuration

Full annotated list in [`.env.example`](.env.example). The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` hides API docs and requires `JWT_SECRET` |
| `JWT_SECRET` | *(random per process)* | **Required in production.** `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `DATABASE_URL` | `sqlite:///./krishi.db` | `postgresql+psycopg://…` for production |
| `ML_SERVICE_URL` | `http://127.0.0.1:8100` | Keep on a private network |
| `GEMINI_API_KEY` | *(empty)* | Enables the AI assistant |
| `DATA_GOV_API_KEY` | *(empty)* | Enables live mandi prices |
| `CORS_ORIGINS` | *(empty)* | Only needed if the frontend is on another domain |
| `MAX_UPLOAD_BYTES` | `8388608` | 8 MB photo limit |
| `DISEASE_MODEL_PATH` | auto-detected | Where the Keras weights live; prefers `.keras` over `.h5` |

No URL is hardcoded. The frontend calls `/api` on its own origin by default; set
`window.KRISHI_API_BASE` before the module scripts to point it elsewhere.

---

## Deployment

### Frontend
Served by the backend at `/`. Nothing to build. To host it separately (CDN,
Netlify, S3), upload `frontend/` as static files, set `window.KRISHI_API_BASE`
to your API URL, and add that origin to `CORS_ORIGINS` on the backend.

### Backend

```bash
ENVIRONMENT=production JWT_SECRET=<generated> \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Put a reverse proxy in front for TLS. Behind a proxy, add `--proxy-headers
--forwarded-allow-ips=<proxy-ip>`.

### ML service

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8100 --workers 2
```

Each worker loads its own copy of the models — the tabular ones total about
150 MB, so size workers against available memory. **Never expose this service
publicly**: it has no authentication by design, because the backend is its trust
boundary.

### Database
SQLite is fine for a single instance. For more than one backend worker or any
real traffic, use PostgreSQL: uncomment `psycopg` in `backend/requirements.txt`
and set `DATABASE_URL`. Tables are created at startup; once the schema starts
changing in production, put Alembic in front of `init_db()`.

---

## Security

- Passwords are bcrypt-hashed. Login failures give one message for both a wrong
  username and a wrong password, so accounts cannot be enumerated.
- Every chat and prediction is scoped to its owner — asking for another user's
  conversation returns 404.
- Uploads are checked for declared size, content type, actual decodability and
  real format before reaching the model.
- The Gemini key is used server-side only and never reaches the browser.
- CORS is empty by default rather than `*`, and the API is token-authenticated.
- Internal errors are logged in full and returned as a generic message.

**If you are migrating from the original projects:** their Django settings
contained committed database passwords and a committed `SECRET_KEY`. Those
credentials should be rotated — see
[docs/PROGRESS.md](docs/PROGRESS.md#6-security-issues-found-in-the-originals).

---

## Where this came from

Five separate projects were merged into this one:

| Source | What was kept |
|---|---|
| `Krishi.AI-main` | Chat data model, the visual language, 13-language support |
| `farmer Chatbot` (Shoora) | Gemini integration, FAQ-before-LLM answering |
| `ml-models` | All four trained models and their agronomy reference tables |
| `Plant-Disease-Prediction-main` | Class list, preprocessing, inference pipeline |
| `public/` | Image assets |

The disease model's weights were not in any of those folders and the author's
download link is dead; they were recovered from their published Docker image and
verified against the original app's own screenshot. See
[docs/plant-disease-model.md](docs/plant-disease-model.md).

Discarded: the placeholder keyword "chatbot", two conflicting Django setups,
four separate Streamlit UIs, and a duplicated dataset. [docs/PROGRESS.md](docs/PROGRESS.md)
records what each project actually contained and why each decision was made.

---

## Retraining

The original training scripts are in `ml-service/training/`:

```bash
cd ml-service/training
python train_crop.py          # needs data/Crop_recommendation.csv
python train_fertilizer.py    # generates its own ICAR-based dataset
python train_water.py         # generates its own FAO-56-based dataset
python train_rainfall.py      # generates its own IMD-normals dataset
```

They write `.pkl` files next to themselves — copy the outputs into the matching
`ml-service/artifacts/<tool>/` folder and restart the service. Keep
`scikit-learn` at the version used for training; a mismatch can break loading.
