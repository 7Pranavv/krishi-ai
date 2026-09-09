# Krishi.AI API reference

Two HTTP surfaces:

- **Backend** — `http://localhost:8000/api`. What the browser talks to. Authenticated.
- **ML service** — `http://localhost:8100`. Internal only. The backend is its only client.

Interactive docs are generated from the code and served at `/api/docs` when
`ENVIRONMENT=development`.

---

## Conventions

**Authentication.** Every endpoint except `/api/health` and the three under
`/api/auth` requires a bearer token:

```
Authorization: Bearer <access_token>
```

A token comes from `POST /api/auth/guest`, `/register` or `/login`. Tokens are
HS256 JWTs valid for `ACCESS_TOKEN_DAYS` (default 30).

**Errors.** Every failure returns the same shape, with a message written for a
farmer to read:

```json
{ "error": "Please check the values you entered.", "detail": "N: Input should be less than or equal to 140" }
```

| Status | Meaning |
|---|---|
| 400 | The request body or file was malformed |
| 401 | Missing, invalid or expired token |
| 404 | Not found, or belongs to another user |
| 409 | Username already taken |
| 413 | Upload above `MAX_UPLOAD_BYTES` |
| 415 | Unsupported file type |
| 422 | Validation failed — `detail` names the field |
| 502 | The AI or ML service answered with an error |
| 503 | A model or the AI assistant is not available |
| 504 | The prediction timed out |

`503` and `504` mean *try again later*; the rest of the app keeps working.

---

## Authentication

### `POST /api/auth/guest`
No body, no auth. Provisions a throwaway account so a farmer can use every tool
without signing up. The browser calls this once and stores the token.

**201** → `{ "access_token": "...", "token_type": "bearer", "user": { ... } }`

### `POST /api/auth/register`
```json
{ "username": "ramesh_k", "password": "at least 8 chars", "full_name": "Ramesh Kumar" }
```
`username` is 3–64 characters of `a-z A-Z 0-9 . _ -`. `password` is 8–72 characters
(bcrypt's limit; longer is rejected rather than silently truncated).

**201** → same shape as `/guest`. **409** if the username exists.

### `POST /api/auth/login`
```json
{ "username": "ramesh_k", "password": "..." }
```
**200** → token. **401** for a wrong username *or* password — the message is
identical either way so the endpoint cannot be used to discover who has an account.

### `GET /api/auth/me` → `UserOut`
### `PATCH /api/auth/me`
Any subset of `full_name`, `pincode` (exactly 6 digits or null), `language`.

**`UserOut`**
```json
{ "id": 1, "username": "guest_4ec7439d", "is_guest": true, "full_name": null,
  "pincode": null, "language": "en",
  "created_at": "2026-09-08T13:42:12", "last_active": "2026-09-08T13:42:12" }
```

---

## Chat

### `POST /api/chat`
```json
{ "message": "My rice leaves are turning yellow", "session_id": "optional-uuid" }
```
Omit `session_id` to start a thread; pass the one you got back to continue it.
An unrecognised id starts a new thread rather than failing mid-conversation.

Answers are resolved in order: curated FAQ table first, then Gemini with the last
`CHAT_HISTORY_TURNS` turns as context.

**200**
```json
{ "reply": "...", "session_id": "91e4b7d1-...", "source": "faq", "response_time_ms": 12.4 }
```
`source` is `"faq"` for a curated answer, `"ai"` for a generated one.

**422** empty or whitespace-only message, or over `CHAT_MAX_CHARS`.
**503** no `GEMINI_API_KEY` configured and no FAQ match.
**502** the model failed or returned nothing. No thread is created when the
answer fails, so the sidebar does not fill with empty conversations.

### `GET /api/chat/conversations?limit=20`
Your threads, newest first: `session_id`, `title`, `created_at`, `updated_at`,
`message_count`.

### `GET /api/chat/conversations/{session_id}`
Adds `messages[]` of `{ id, role, content, created_at }` where `role` is
`"user"` or `"bot"`. **404** if the thread is not yours.

### `DELETE /api/chat/conversations/{session_id}` → **204**
Deletes the thread and its messages.

---

## ML tools

Uniform for every tool:

```
GET  /api/ml/{tool}/options    dropdown values, read off the trained encoders
POST /api/ml/{tool}/predict    run inference, save it to your history
```

`tool` is one of `crop`, `fertilizer`, `water`, `rainfall`, `disease`.

Always populate dropdowns from `/options` rather than hardcoding. The values come
straight from the label encoders the models were trained with, so they cannot
drift out of sync.

```json
{ "crop": { "values": ["rice", "wheat"], "labels": { "rice": "Rice", "wheat": "Wheat" } } }
```

Numeric bounds below are the ranges each model was trained on. Values outside
them are rejected rather than silently extrapolated.

### `POST /api/ml/crop/predict`
```json
{ "N": 90, "P": 42, "K": 43, "temperature": 20.9,
  "humidity": 82, "ph": 6.5, "rainfall": 202 }
```
`N` 0–140, `P` 5–145, `K` 5–205 (kg/ha) · `temperature` 8–44 °C · `humidity`
14–100 % · `ph` 3.5–10 · `rainfall` 20–300 mm.

**200**
```json
{ "crop": "rice", "confidence": 0.935, "low_confidence": false,
  "season": "Kharif", "tip": "Needs flooded fields and high humidity.", "emoji": "🌾",
  "top_predictions": [ { "crop": "rice", "probability": 0.935, "emoji": "🌾" } ] }
```
`low_confidence` is true below `ML_LOW_CONFIDENCE` (0.55). Show the result, but
tell the user it is uncertain. `top_predictions` includes the winner first.

### `POST /api/ml/fertilizer/predict`
```json
{ "crop": "wheat", "soil_type": "loamy", "irrigation": "irrigated",
  "prev_crop": "legume", "soil_N": 250, "soil_P": 10, "soil_K": 150,
  "soil_pH": 6.5, "organic_matter": 0.8, "target_yield": 3.0 }
```
`soil_N` 0–900, `soil_P` 0–45, `soil_K` 0–500 kg/ha · `soil_pH` 4–9 ·
`organic_matter` 0.1–3 % · `target_yield` 0.1–100 t/ha.

**200** returns `fertilizer`, `n_dose_kg_ha` / `p_dose_kg_ha` / `k_dose_kg_ha`,
their commercial equivalents `urea_kg_ha` / `dap_kg_ha` / `mop_kg_ha`,
`soil_status` (`Low`/`Medium`/`High` per nutrient, ICAR bands),
`schedule[]` of `{ stage, action }`, and a crop `tip`.

### `POST /api/ml/water/predict`
```json
{ "crop": "rice", "soil_type": "clay", "growth_stage": "mid",
  "temperature": 32, "humidity": 70, "wind_speed": 2, "sunshine_hours": 9,
  "rainfall": 0, "area": 2, "area_unit": "acre" }
```
`temperature` 10–45 °C · `humidity` 10–100 % · `wind_speed` 0–8 m/s ·
`sunshine_hours` 2–14 · `rainfall` 0–30 mm/day · `area` 0.1–500 ·
`area_unit` one of `hectare` | `acre` | `bigha`.

**200** returns `water_req_mm_day`, `irrigation_interval_days`, `area_hectares`,
`litres_per_day`, `litres_per_week`, `stress_risk` (`Low`/`Moderate`/`High`),
`risk_note`, `tip`, and `schedule` — 7 booleans, one per day from today.

### `POST /api/ml/rainfall/predict`
```json
{ "state": "Punjab", "month": 7, "temperature": 33, "humidity": 70,
  "prev_month_rain": 55 }
```
`month` 1–12 · `temperature` 5–45 °C · `humidity` 10–100 % ·
`prev_month_rain` 0–800 mm, **optional** — omit it and the IMD normal for the
previous month is used.

**200** returns `predicted_mm`, `normal_mm`, `departure_pct`, `category`
(`Deficient` / `Normal` / `Excess`, the IMD ±20 % classification), `season`,
`advice`, and `monthly_normals` — 12 values for charting.

### `POST /api/ml/disease/predict`
`multipart/form-data`:

| Field | Type | |
|---|---|---|
| `file` | JPG, PNG or WEBP, ≤ `MAX_UPLOAD_BYTES` | required |
| `language` | e.g. `Hindi` — language for the guidance text | default `English` |
| `explain` | ask the AI for cause/prevention/treatment | default `true` |

**200**
```json
{ "label": "Tomato___Late_blight", "crop": "Tomato", "condition": "Late blight",
  "healthy": false, "confidence": 0.91, "low_confidence": false,
  "top_predictions": [ ... ],
  "guidance": "Cause: ...\nPrevention: ...\nTreatment: ..." }
```
`guidance` is AI-generated and is `null` when the assistant is unavailable — the
detection is still returned. `crop` and `condition` are split from the model's
own label; nothing is inferred beyond that.

**503** when the trained weights are not installed — see
[plant-disease-model.md](plant-disease-model.md).

### `GET /api/ml/disease/classes`
The 38 labels the model can produce. Works even when the weights are missing.

---

## Dashboard and history

### `GET /api/dashboard`
Everything the home page needs in one call: `user`, `total_predictions`,
`total_conversations`, `recent_predictions[]` (5), `recent_conversations[]` (5),
and `services`:

```json
{ "ml": { "status": "ok", "models": { "crop": { "ready": true, "error": null },
                                      "disease": { "ready": false, "error": "model weights not found..." } } },
  "assistant": { "ready": false } }
```
Use `services` to flag unavailable tools. It never fails the request — an
unreachable ML service reports `"status": "unreachable"`.

### `GET /api/predictions?tool=crop&limit=20`
Your saved results, newest first. `tool` is optional; an unknown value is a 422.
Each row has `id`, `tool`, `summary`, `inputs`, `result`, `created_at`.

### `DELETE /api/predictions/{id}` → **204**

### `GET /api/health`
No auth. Reports backend status, assistant readiness, and per-model ML status.

---

## ML service (internal)

Same paths without the `/api/ml` prefix and without authentication:
`/crop/predict`, `/crop/options`, `/fertilizer/…`, `/water/…`, `/rainfall/…`,
`/disease/predict`, `/disease/classes`, `/health`.

Request and response bodies are identical, except that `guidance` on a disease
result is added by the backend, not the ML service.

**Do not expose this service to the internet.** It has no authentication by
design — the backend is its trust boundary. Bind it to localhost or a private
network.


## Market prices

Requires `DATA_GOV_API_KEY`. Every endpoint returns 503 with an explanation when
it is unset - never fabricated prices.

### `GET /api/market/options`
Commodities and states present in the current government feed. Auth: none.

### `GET /api/market/prices`
Query: `commodity`, `state` (both optional), `limit` (1-500, default 50).
Auth: bearer token.

```json
{
  "count": 12,
  "source": "Agmarknet via data.gov.in",
  "records": [{
    "commodity": "Wheat", "variety": "Dara", "state": "Punjab",
    "district": "Amritsar", "market": "Amritsar",
    "arrival_date": "08/09/2026",
    "min_price": 2400.0, "max_price": 2500.0, "modal_price": 2450.0,
    "unit": "INR per quintal"
  }]
}
```

Rows whose modal price was not reported are omitted rather than zero-filled.

### `POST /api/market/harvest-value`
Body: `commodity`, `state` (optional), `yield_tonnes_per_hectare`, `area_hectares`.
Auth: bearer token.

Returns the gross value at the **median** modal price across reporting mandis,
with the price range and the reference market. Returns 404 when no mandi has
reported that commodity - it does not estimate a price.

Errors: 404 no reported price, 422 invalid input, 503 not configured or upstream
unreachable.
