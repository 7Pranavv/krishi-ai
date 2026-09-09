# Krishi.AI integration — build log

**Last updated:** 2026-09-09 (audit + full i18n)
**Status:** Integration complete and tested. All five models serving real
predictions. Nothing was deleted from the original project folders.

---

## 1. What the source projects actually are

Findings from a full read of every file. Several differ from the original brief.

| Folder | What it really is | Verdict |
|---|---|---|
| `Krishi.AI-main/` | **Not** a website. A Django 4.2 + FastAPI chatbot with one 465-line `index.html`. `chat/chatbot_handler.py` is a **hardcoded keyword placeholder** — no AI at all. Postgres creds committed in `config/settings.py`. | Source of truth for the **chat data model** (Conversation/Message/UserProfile) and the **frontend look + 13-language i18n**. Its bot logic was discarded. |
| `farmer Chatbot/Shoora Chatbot/` | Django 5.2 + DRF + **Gemini 1.5 Flash**. The only real chatbot. Bugs: `llm_service.get_db_answer` reads `obj.bot_answer` but the field is `answer` (AttributeError); `ChatHistory.__str__` is broken. Prompt says "Farmer safety & tracking platform" (wrong product). Postgres creds committed. `GEMINI_API_KEY = ""` (no key leaked). | Source of truth for **AI answering** and the FAQ-before-LLM pattern. |
| `ml-models/` | 4 Streamlit apps, each with real trained `.pkl` files (**scikit-learn 1.6.1**). All load and predict correctly. | Source of truth for **all tabular ML** and the agronomy reference tables. |
| `Plant-Disease-Prediction-main/` | Streamlit + Xception, 38 PlantVillage classes. **The `.h5` weights file is MISSING** — only `class_indices.json` and the training notebook. Also references a missing `config.json` holding `GOOGLE_API_KEY`. | Preprocessing + class list recovered; weights must be supplied. |
| `public/` | Loose assets (`Crop.png`, `Hero.png`, `vite.svg`, farm photos) from an abandoned Vite frontend. **No `package.json` anywhere in the repo.** | Assets reused. |
| `Crop_recommendation.csv` (root) | Byte-identical duplicate of `ml-models/crop-recommendation/Crop_recommendation.csv` (md5 `fa837397…`). | Deduplicated. |

**No Node/React/Vite project exists anywhere.** The whole stack is Python, so the
integration stays Python — no framework rewrite.

### Exact model contracts (read from `train.py`, not assumed)

| Model | Algorithm | Feature order | Output |
|---|---|---|---|
| crop | RandomForestClassifier | `N, P, K, temperature, humidity, ph, rainfall` | 26 crops + `predict_proba` (retrained — see section 7) |
| fertilizer | 3× GradientBoostingRegressor + RandomForestClassifier | `crop_enc, soil_enc, irr_enc, prev_enc, soil_N, soil_P, soil_K, soil_pH, organic_matter, target_yield` | N/P/K kg/ha + product (8 classes) |
| water | 2× RandomForestRegressor | `crop_enc, soil_enc, stage_enc, temperature, humidity, wind_speed, sunshine_hours, rainfall` | mm/day + interval days |
| rainfall | GradientBoostingRegressor | `state_enc, month, season_enc, temperature, humidity, prev_month_rain, avg_normal_rain` | mm + IMD departure category |
| disease | Xception fine-tune | 224×224 RGB, **rescale `/255`** (not `preprocess_input`) | 38-class softmax |

Encoder vocabularies verified at runtime: 22 crops (crop); 24 crops / 7 soils /
3 irrigation / 5 prev-crop / 8 fertilizers; 15 crops / 5 soils / 4 stages (water);
20 states / 4 seasons (rainfall); 38 disease classes.

---

## 2. Architecture

```
Browser (static HTML/CSS/ES modules, no build step)
    |  JWT bearer
Backend  FastAPI :8000  — auth, chat, history, upload limits, serves the frontend
    |  httpx
ML service FastAPI :8100 — owns every model file, loads once at startup
    |
SQLite (default) / Postgres via DATABASE_URL
```

Decisions and why:

- **Django dropped, SQLModel adopted.** Both chat projects ran Django only for
  its ORM, and one called `django.setup()` inside FastAPI. The two Django
  projects also had incompatible versions (4.2 vs 5.2), separate Postgres
  databases and separate history schemas — they could not coexist. One ORM, one
  config, SQLite default so a fresh clone runs with no external services.
- **Two services, two venvs.** ML pins `scikit-learn==1.6.1` (required to
  unpickle) and optionally TensorFlow; the backend needs neither. The split also
  gives the graceful degradation Phase 14 asks for.
- **Guest auth.** Every API call is authenticated, but the browser silently
  provisions a guest account on first visit so no farmer hits a signup wall.
  Registering upgrades the account.
- **No fake predictions.** Every endpoint that cannot answer honestly returns
  503 with an explanation and the UI shows a banner, rather than guessing. This
  covers the AI assistant without a key, market prices without a key, and any
  model that fails to load.

---

## 3. What was built

### ML service — `ml-service/`
`registry.py` (loads each bundle once; a failed bundle marks only itself
unavailable), `knowledge.py` (agronomy tables lifted verbatim from the Streamlit
apps), `routers/{crop,fertilizer,water,rainfall,disease}.py` with `/predict` and
`/options`, `main.py` (lifespan loading, `{error, detail}` envelope, `/health`).
Plus `season.py`, `explain.py` (SHAP) and `gradcam.py` — see section 6.

### Backend — `backend/`
`config.py` (refuses to start in production without `JWT_SECRET`), `db.py`,
`models.py` (User / Conversation / Message / Prediction / FaqEntry),
`security.py` (bcrypt + PyJWT), `schemas.py`, `services/ml_client.py` (pooled
httpx with timeout/503/502 translation), `services/assistant.py` (FAQ → Gemini,
farming system prompt, disease explanation), `services/market.py` (live
Agmarknet prices), `routers/{auth,chat,ml,dashboard,market}.py`, `main.py`.
Plus `seed_faq.py` and `test_api.py`.

### Frontend — `frontend/`
`css/app.css` (one design system, light/dark, responsive), `js/{api,i18n,ui,
chat-core,widget}.js`, and 10 pages: `index`, `chat`, `crop`, `disease`,
`fertilizer`, `water`, `rainfall`, `market`, `profile`, `404`.

### Docs
`README.md`, `.env.example`, `.gitignore`, `docs/api.md`,
`docs/plant-disease-model.md`, this file.

---

## 4. Test results

`backend/test_api.py` — checks against running servers, real models, no mocks:

```
53 passed, 0 failed, 2 skipped
  skipped: AI-unavailable path (GEMINI_API_KEY is set)
  skipped: live market price call (DATA_GOV_API_KEY is not set)
```

Each skip is a path that only runs when the corresponding key is absent or
present, so the suite covers both configurations across two runs.

The disease checks assert against published ground truth: the original
project's own screenshot shows a specific grape leaf detected as
`Grape___Black_rot` at 100.00%, and `backend/testdata/grape_black_rot.png` is
that leaf. Matching it verifies the weights, the class mapping and the
preprocessing together - something a synthetic image cannot do.

Live AI verified separately: English / Hindi / Tamil / Marathi replies in the
right language, conversation memory across turns, an off-topic question
declined, an unknown MSP answered with "I do not know" plus who to ask, and a
direct prompt-extraction attempt refused.

Covers: guest provisioning, register/login/duplicate/weak-password, token
rejection, all four models' `/options` and `/predict`, out-of-range and
unknown-category rejection, auth on predictions, chat threading and history,
cross-user isolation, deletion, the honest 503 when the AI is unconfigured, and
dashboard/history filtering.

Verified in the browser at 1280×900 and 375×812 (mobile): every page loads,
navigation and the mobile menu work, all four tools return real predictions,
the chat page and floating widget both work, forms validate, and there is no
horizontal overflow.

File-upload validation, checked directly: oversize → 413, wrong type → 415,
empty → 400, missing → 422, valid image → the honest 503 for missing weights.

Path traversal probes (`/../README.md`, `/..%2F..%2F.env`, `/backend/app/config.py`,
`/krishi.db`, …) all return 404 with no file contents leaked.

### Bugs found and fixed during integration

| Bug | Fix |
|---|---|
| `from __future__ import annotations` in `models.py` broke SQLModel relationship mapping at runtime | Removed, with a comment saying why it must not come back |
| A failed chat still created an empty conversation, littering the sidebar | Answer first, persist the thread only on success |
| `[hidden]` was overridden by `.chat-panel { display: flex }`, so the widget could not close | Global `[hidden] { display: none !important }` |
| Re-raising from the HTTP exception handler turned every unmatched multi-segment URL into a 500 | Every branch now returns a response; page 404s serve `404.html` |
| Fertilizer `target_yield` had `min="0.1" step="0.5" value="3"` — an invalid default, so the form silently refused to submit | `step="0.1"`, with a comment |
| Chat welcome panel stayed on screen when a message was typed rather than clicked | `data-welcome` nodes cleared on first bubble |
| "PER DAY · 1 ACRES" | Singularised |
| `google-generativeai` hung indefinitely against a live key — the package is end of life and stated so at import | Migrated to the supported `google-genai` SDK; added an explicit request timeout so a stalled provider returns 504 instead of holding the request open |
| `gemini-1.5-flash` is retired; `gemini-flash-latest` returned 504 DEADLINE_EXCEEDED | Pinned `gemini-2.5-flash`, verified working at ~1.4 s |
| The assistant answered **English** questions in Hindi | Reply language now comes from the user's stored profile language, stated explicitly per request; the header selector writes it to the profile |
| Model replies are Markdown, so `**bold**` showed as literal asterisks | `formatReply()` in `ui.js` — escapes first, then re-introduces bold and bullets, so no model output can inject markup |
| A returning browser kept a stale `ui.js` while `chat-core.js` was fresh; the missing export failed the whole ES-module graph and the page rendered **nothing** — no header, no tools | `Cache-Control: no-cache` on `/js/*`, `/css/*` and HTML, so code assets always revalidate (304 with the existing ETag). Images still cache normally. Without a build step there are no content-hashed URLs, which makes this failure mode reachable on every deploy |
| `HEAD` on any page returned 405 — FastAPI does not add HEAD to `@app.get` the way Starlette does | Pages now serve `GET, HEAD`, so uptime monitors and link checkers work |
| The distributed `.h5` (Keras 3.5.0) loads in neither Keras 3.15 (`Cannot convert '((None, 2048),)' to a shape`) nor Keras 2 (`Unrecognized keyword arguments: ['batch_shape']`) | Rebuild the architecture from the notebook with layer names pinned to the file's, load weights by name, and guard against a silent no-match by comparing the output layer before and after. Cached as native `.keras` afterwards |
| Gemini free tier is ~20 requests/day *per model*; hitting it surfaced as a generic "could not answer" | 429 now maps to a clear "reached today's usage limit" message, and the frontend treats it as a retryable amber state rather than a red error |
| AI replies use Markdown headings (`### Cause`) which rendered as literal hashes; a naive italic rule then turned `2 * 3 * 4` into italics | `formatReply()` handles headings, bold, italics and bullets, escaping first; the italic rule requires non-space at both edges |
| `sqlite:///./krishi.db` resolves against the working directory, so launching uvicorn from `backend/` used a *different, unseeded* database than launching from the repo root | Relative SQLite paths are now anchored to the repo root in `config.py` |
| A system-prompt rule leaked into an answer as advice to the farmer | Prompt reworded so the rules read as instructions to the assistant, with an explicit "never quote these" clause; verified against a direct prompt-extraction attempt |

---

## 5. Recovering the disease model

The `.h5` weights were in none of the source folders, and the author's Google
Drive link (named in their README) now returns **404**. The upstream GitHub repo
has no model file and no releases.

The surviving copy was inside the author's published Docker image. Rather than
pull 1.9 GB, the manifest was inspected and only the 237 MB `COPY . /app` layer
fetched, with its SHA-256 verified against the registry digest. That layer
contained `app/trained_model/plant_disease_prediction.h5` (245 MB). Only the
model was extracted - the layer also holds the author's `config.json` with their
API key, which was left alone.

The file turned out to be unloadable by either current Keras major version, so
`ml-service/app/disease_model.py` rebuilds the architecture from the notebook
and loads weights by name instead, then caches a native `.keras` copy. Full
detail in [plant-disease-model.md](plant-disease-model.md).

Deliberately not done at any point: no stand-in model, no heuristic, no cached
sample results. A wrong disease call costs a farmer a crop.

---

## 6. Security issues in the originals

Fixed in the new tree; still present in the old folders:

- Postgres passwords committed in `Krishi.AI-main/.../config/settings.py`
  (`chatbot_0`) and `ShooraChatbot/settings.py` (`Shoora@123`) —
  **these credentials should be rotated.**
- Django `SECRET_KEY` committed in `ShooraChatbot/settings.py`.
- `CORS_ALLOW_ALL_ORIGINS = True` together with `allow_credentials=True` in both.
- `ALLOWED_HOSTS = ['*']`, `DEBUG = True`.
- The old chat API trusted a client-supplied `user_id` — anyone could read
  anyone else's history.
- No upload validation anywhere in the disease app.
- A committed `venv/` and `env/` inside the source folders.

---

## 7. Remaining work

1. Add Alembic once the schema starts changing in production.
2b. The Gemini key was pasted into a chat transcript during setup — rotate it.
3. Decide what to do with the five original folders now the new tree is verified.
4. Cosmetic: Starlette deprecates `HTTP_422_UNPROCESSABLE_ENTITY` in favour of
   `HTTP_422_UNPROCESSABLE_CONTENT` — a warning, not a failure.
5. The 13-language selector translates the shell strings that the original
   project had human-authored translations for; newer UI strings fall back to
   English per key. Filling those in needs a native speaker, not a machine.


---

## 6. Features ported from AICropRecommendation

Source: [AhqafCoder/AICropRecommendation](https://github.com/AhqafCoder/AICropRecommendation)
(MIT). Reviewed the whole repo before taking anything.

**Kept ours instead:** their disease model is ResNet50 over 15 classes at 90.09%
and its `.pth` files are 133-byte Git LFS pointers with no weights behind them;
ours is Xception over 38 classes at 99.7% and works. Their crop model is the same
RandomForest on the same dataset.

**Ported:**

| Feature | Where | Notes |
|---|---|---|
| Season detection | `ml-service/app/season.py` | Region-aware Kharif/Rabi/Zaid, with the state-to-region map added so it works from the states our rainfall model already knows. Returned with every crop recommendation, plus `GET /crop/season`. |
| SHAP explainability | `ml-service/app/explain.py` | Per-prediction feature contributions on the crop model. Explainer built once at first use — constructing a `TreeExplainer` per request took 1.3 s; warm calls are 0.05 s. |
| Grad-CAM | `ml-service/app/gradcam.py` | Heatmap over the uploaded leaf. Adds ~2 s to a disease prediction. |
| Market prices | `backend/app/services/market.py` | **Not** their version — see below. |

**Deliberately not ported:** their marketplace, analytics dashboard and "market
price integration" all read from `backend/config/mockDatabase.js` — hardcoded
products, invented prices, fake ratings and review counts. Showing a farmer a
fabricated mandi price could cost them money on a real sale. Built against the
Government of India's live Agmarknet feed instead, which returns nothing when a
mandi has not reported rather than filling the gap.

Their Android app was out of scope; our frontend is already responsive.

### Bugs found while porting

| Bug | Fix |
|---|---|
| Grad-CAM produced a blank heatmap on every image | The model outputs an exact `1.0` for confident predictions, where softmax has zero gradient. Differentiate the **pre-softmax logit** instead. |
| The heatmap tinted the whole photo evenly and showed nothing useful | 7×7 feature map upsampled linearly; added a gamma curve so only the hottest region is tinted, and alpha-blended per pixel so cool areas stay as the original photo. |
| A 275 KB base64 heatmap was being written into every history row | Stripped before persisting, with a `heatmap_returned` flag kept instead. History rows are 1.6 KB. |
| **data.gov.in silently black-holes requests whose User-Agent is the httpx default.** The connection opened, nothing ever came back, and the request died on the read timeout — while `curl` to the same URL answered in under a second. | Send an explicit `User-Agent`. Any recognisable agent gets an immediate response. Marked do-not-remove in `market.py`. |
| A 429 rate-limit was being reported to the farmer as "could not reach the service" | Handle 429 explicitly, with one short retry, and say plainly that the key is being throttled |
| The crop/state dropdowns offered only 9 crops and 2 states | The shared demo key truncates every response to 10 rows, so option discovery saw almost nothing. Fields are now typeable `<input list=…>` with the discovered values as suggestions, which is more robust on any key tier. |
| **data.gov.in's own filtering breaks above ~300 rows per page.** Asking for `limit=500` with `filters[state]=Madhya Pradesh` returned 305 MP rows *plus 195 Uttar Pradesh rows*, while still reporting the filtered total — so a farmer could have been shown another state's rates as their own. Verified 2026-09-09: 250 and 300 clean, 350 leaks 45, 500 leaks 195. | Cap the upstream page at `MAX_UPSTREAM_PAGE = 300`, **and** drop any row that does not match the requested crop/state after parsing. Regression test asserts a `limit=500` state-filtered query returns exactly one state. |
| A single median price was quoted for crops whose mandi rates ranged 5x (onion ₹800–9,000) | `wide_spread` flag when max/min > 1.5; the UI switches to a warning telling the farmer to check their own mandi rather than trust the median |
| The market page reported "not configured" twice | Check `/health` before fetching options, and skip the redundant call. |


---

## 7. Crop model retrained to 26 crops

Asked to adopt the model from
[KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning](https://github.com/KRUTHIKTR/Crop-Recommendation-System-Using-Machine-Learning)
(MIT). Inspected before swapping anything.

**Their published artefact was not adopted.** Measured on the same data and the
same 80/20 split:

| Model | Held-out accuracy |
|---|---|
| RandomForest (ours) | **99.55%** |
| GaussianNB (their README's stated best) | 99.55% |
| SVC (what their pickle actually contains) | 98.41% |

Their file is an `SVC`, not the GaussianNB the README claims, ships no scaler,
and has `probability=False` — so it cannot produce the confidence score or the
top-3 alternatives the UI is built around. Their `crop_data1.csv` is
byte-identical to the dataset we already train on.

**Their second dataset is largely corrupt.** 1603 of its 1796 rows already
exist in dataset 1, and 547 of those carry a different crop label on identical
soil and climate values:

| Real label (dataset 1) | Relabelled as | Rows |
|---|---|---|
| kidneybeans | beans | 125 |
| mungbean | cowpeas | 122 |
| chickpea | Soyabeans | 100 |
| pigeonpeas | peas | 100 |
| mothbeans | groundnuts | 100 |

Five of their nine "extra" crops are therefore existing pulses renamed. Their
31-class model is trained on that, so it splits each pulse's evidence across two
labels and will answer "soyabeans" for chickpea conditions. It also carries both
`Groundnut` and `groundnuts` as separate classes for one crop.

**What was actually gained.** Four crops appear nowhere in dataset 1 and have
genuine data: arecanut (24 rows), jackfruit (25), sugarcane (25), groundnut
(25), plus 64 extra samples for rice/maize/banana. `training/train_crop.py`
merges only those, lowercases labels, and excludes the relabelled names by
name. Result: **26 crops at 99.79% held-out accuracy**, up from 22 at 99.55%,
keeping RandomForest and `predict_proba`.

The four new crops have only ~5 test samples each, so their accuracy figures
carry much less weight than the established 22. Previous artefacts are kept
beside the new ones as `.pkl.bak`.


---

## 8. Audit, bug fixes and full multilingual support

### Bugs found by audit and fixed

| Bug | Why it mattered | Fix |
|---|---|---|
| **bcrypt takes 72 BYTES, the schema capped 72 CHARACTERS.** A 70-character Devanagari password is 210 bytes, so bcrypt raised `ValueError` and registration returned **500**. | Directly in the way of the multilingual goal - non-ASCII passwords are expected here | Validate byte length in `RegisterRequest` with a message that explains non-English letters take more space |
| `MAX_PASSWORD_BYTES` / `MIN_PASSWORD_LENGTH` were declared with a comment about rejecting long passwords, but nothing enforced them | Dead constants that read as if the check existed | Now actually used by the validator |
| Disease conditions rendered inconsistently cased - "Early blight" beside "healthy" | The condition is the page's main heading | Normalise the first letter in `prettify_disease_label` |
| The mandi feed spells states differently from the rainfall model: `Keralam`, `Chattisgarh`, `NCT of Delhi`, `Manipur` | All four silently fell back to the generic season calendar. Kerala's Kharif runs a month longer than the default, so a Kerala farmer got the wrong season | Aliases added, plus the remaining states and UTs; `Delhi` and `Jammu and Kashmir` were also duplicated in the dict |
| `season.py` claimed in a comment that its crop lists were filtered to crops the model can return - they were not (wheat, potato, tomato…) | A comment asserting something false, next to a UI that implied the recommender could suggest those crops | Comment corrected, and the UI section relabelled "Typical crops for this season" with a note that some are not covered by the recommender |
| Dead code: `ApiError`, `ErrorResponse`, `showLoading`, `languageName` | — | Removed |
| `chat.html` imported `t` from two modules | **Fatal** `SyntaxError` - the whole chat page rendered nothing | Duplicate import removed; every page's inline script is now syntax-checked |

`ruff --select F,E9` reports zero errors across `backend/`, `ml-service/` and
`tools/`. The 37 `B008` hits are FastAPI's `Depends()`-in-defaults idiom and are
not defects.

### Multilingual

Before: 11 keys, and only English and Hindi had navigation labels - the other
11 languages fell back to English for almost everything.

After: **13 languages, two catalogues, no runtime translation API.**

- `frontend/i18n/en.json` — 283 interface strings
- `frontend/i18n/content.en.json` — 135 agronomic advisories extracted from the
  ML service's own tables by `tools/extract_content.py`

The second catalogue is the point. The interface being translated while
"Soil moisture is adequate. Monitor and irrigate only if rainfall drops."
stayed English would leave the app still requiring English to use.

Tooling (all reproducible, all under `tools/`):

| Script | Job |
|---|---|
| `extract_content.py` | Pull advisory strings out of `knowledge.py`, `season.py` and the water router |
| `translate_ui.py` | Generate the 12 translations via Gemini. `--content` for the advice catalogue, `--repair` for keys that came back in English |
| `annotate_i18n.py` | Tag markup with `data-i18n`, matching whole element text |
| `localise_scripts.py` | Replace English literals in page scripts with `t()` |

### Problems hit while building it

| Problem | Fix |
|---|---|
| Free tier allows **20 requests/day/model**; 283 keys × 12 languages in 45-key batches needed ~80 | One request per language (the cap is on requests, not tokens), plus rotation across 8 models, each with its own quota |
| `gemini-2.5-pro` answers 404 "no longer available to new users", and rotation **stuck** there - every language after it failed | Permanent failures mark a model dead and move on; transient 503s retry elsewhere |
| Several models read "keep N, P, K unchanged" as licence to leave "Nitrogen (N)" wholly in English | Keep-list narrowed to genuine symbols, with an explicit rule to translate the words around them. `--repair` catches any value identical to its English source - it fixed 36 keys across 6 languages |
| Wrapped `<p>` text never matched its one-line catalogue entry | Whitespace-normalised matching |
| Busy labels written `'Checking...'` did not match `"Checking…"` | `...` and `…` treated as equal |
| 77 `data-i18n` attributes ended up **inside script template literals**, where `applyTranslations()` never reaches them | A `MutationObserver` translates any inserted node carrying the attribute - fewer call sites than translating after every render, and impossible to forget for a new panel |
| CSS used physical properties (`right:`, `text-align: right`), so Urdu's `dir="rtl"` did not mirror | Converted to logical properties (`inset-inline-end`, `text-align: end`, `border-end-end-radius`) |
| `DAYS` became a function but one call site still indexed it as an array | Caught by per-page inline-script syntax checks, now part of the routine |
