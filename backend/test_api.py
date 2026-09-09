"""End-to-end check of the Krishi.AI API.

Hits a running server over HTTP - no mocks, no fixtures, real models. Run it
after any change to the backend or the ML service.

    # terminal 1
    cd ml-service && python -m uvicorn app.main:app --port 8100
    # terminal 2
    cd backend && python -m uvicorn app.main:app --port 8000
    # terminal 3
    cd backend && python test_api.py

Exits non-zero if anything fails, so CI can use it as-is.

Two checks depend on server configuration and are skipped otherwise:
  * the FAQ answers need `python seed_faq.py` to have been run;
  * "AI unavailable" only applies when GEMINI_API_KEY is unset.
"""
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = os.getenv("KRISHI_API_BASE", "http://127.0.0.1:8000/api")

_token: str | None = None
_passed = 0
_failed: list[str] = []
_skipped: list[str] = []


def call(path, body=None, method=None, raw=None, token=True):
    headers = {"Content-Type": "application/json"}
    if _token and token:
        headers["Authorization"] = f"Bearer {_token}"
    data = raw if raw is not None else (
        json.dumps(body).encode() if body is not None else None
    )
    request = urllib.request.Request(
        BASE + path, data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, (None if response.status == 204 else json.load(response))
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, None


def upload_image(path, content_type, data=None):
    """POST a multipart image to the disease endpoint."""
    payload = data if data is not None else path.read_bytes()
    name = path.name if path is not None else "upload.bin"
    boundary = uuid.uuid4().hex
    body = bytearray()
    for field, value in (("language", "English"), ("explain", "false")):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{field}"\r\n\r\n{value}\r\n').encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
             f'filename="{name}"\r\nContent-Type: {content_type}\r\n\r\n').encode()
    body += payload + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        BASE + "/ml/disease/predict", data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {_token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, None


def check(name, got, want):
    global _passed
    if got == want:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}: got {got!r}, expected {want!r}")


def section(title):
    print(f"\n{title}")


def main() -> int:
    global _token

    try:
        status, health = call("/health", token=False)
    except OSError as exc:
        print(f"Cannot reach {BASE} - is the backend running?  ({exc})")
        return 2

    ai_enabled = health["assistant"]["ready"]
    models = health.get("ml", {}).get("models", {})
    print(f"Testing {BASE}")
    print(f"  assistant: {'configured' if ai_enabled else 'not configured'}")
    print("  models: " + ", ".join(f"{k}={'ok' if v['ready'] else 'unavailable'}"
                                    for k, v in sorted(models.items())))

    section("Authentication")
    status, guest = call("/auth/guest", method="POST", raw=b"")
    check("guest provisioning", status, 201)
    if status != 201:
        print("\nCannot continue without a token.")
        return 1
    _token = guest["access_token"]

    check("guest is flagged as guest", guest["user"]["is_guest"], True)
    check("current user", call("/auth/me")[0], 200)
    check("request without a token is rejected", call("/auth/me", token=False)[0], 401)

    saved, _token = _token, "not-a-real-token"
    check("invalid token is rejected", call("/auth/me")[0], 401)
    _token = saved

    # Unique so the suite can be re-run against the same database.
    username = f"test_{guest['user']['id']}_{os.getpid()}"
    check("register", call("/auth/register",
                           {"username": username, "password": "tractor12345"})[0], 201)
    check("duplicate username", call("/auth/register",
                                     {"username": username, "password": "tractor12345"})[0], 409)
    check("password under 8 characters", call("/auth/register",
                                              {"username": f"{username}x", "password": "short"})[0], 422)
    check("username with spaces", call("/auth/register",
                                       {"username": "bad name!", "password": "tractor12345"})[0], 422)
    check("login", call("/auth/login",
                        {"username": username, "password": "tractor12345"})[0], 200)
    check("wrong password", call("/auth/login",
                                 {"username": username, "password": "wrongpassword"})[0], 401)
    check("profile update", call("/auth/me", {"pincode": "110001"}, method="PATCH")[0], 200)
    check("malformed pincode", call("/auth/me", {"pincode": "12"}, method="PATCH")[0], 422)

    section("ML tools")
    payloads = {
        "crop": {"N": 90, "P": 42, "K": 43, "temperature": 20.9,
                 "humidity": 82, "ph": 6.5, "rainfall": 202},
        "fertilizer": {"crop": "wheat", "soil_type": "loamy", "irrigation": "irrigated",
                       "prev_crop": "legume", "soil_N": 250, "soil_P": 10, "soil_K": 150,
                       "soil_pH": 6.5, "organic_matter": 0.8, "target_yield": 3.0},
        "water": {"crop": "rice", "soil_type": "clay", "growth_stage": "mid",
                  "temperature": 32, "humidity": 70, "wind_speed": 2,
                  "sunshine_hours": 9, "rainfall": 0},
        "rainfall": {"state": "Punjab", "month": 7, "temperature": 33, "humidity": 70},
    }
    for tool, payload in payloads.items():
        if not models.get(tool, {}).get("ready", True):
            _skipped.append(f"{tool} (model not loaded on the server)")
            continue
        check(f"{tool} options", call(f"/ml/{tool}/options")[0], 200)
        check(f"{tool} prediction", call(f"/ml/{tool}/predict", payload)[0], 200)

    if models.get("crop", {}).get("ready"):
        check("value above the trained range",
              call("/ml/crop/predict", {**payloads["crop"], "N": 999})[0], 422)
        check("missing required field", call("/ml/crop/predict", {"N": 90})[0], 422)
        check("prediction requires a token",
              call("/ml/crop/predict", payloads["crop"], token=False)[0], 401)
    if models.get("water", {}).get("ready"):
        check("crop the model has never seen",
              call("/ml/water/predict", {**payloads["water"], "crop": "quinoa"})[0], 422)
    if models.get("rainfall", {}).get("ready"):
        check("optional previous rainfall",
              call("/ml/rainfall/predict", {**payloads["rainfall"], "prev_month_rain": 25})[0], 200)

    # The class list must work even without the weights file.
    check("disease class list", call("/ml/disease/classes")[0], 200)

    if models.get("disease", {}).get("ready"):
        fixture = Path(__file__).parent / "testdata" / "grape_black_rot.png"
        if fixture.is_file():
            status, result = upload_image(fixture, "image/png")
            check("disease prediction", status, 200)
            if status == 200:
                # Ground truth: the original project's own screenshot shows
                # this leaf detected as Grape___Black_rot. Same label here
                # means the weights, class mapping and preprocessing all match.
                check("disease label matches published ground truth",
                      result.get("label"), "Grape___Black_rot")
                check("crop parsed from label", result.get("crop"), "Grape")
                check("not flagged healthy", result.get("healthy"), False)
        else:
            _skipped.append("disease prediction (testdata/grape_black_rot.png missing)")
        check("rejects a non-image", upload_image(None, "text/plain", b"not an image")[0], 415)
        check("rejects an empty file", upload_image(None, "image/png", b"")[0], 400)
    else:
        _skipped.append("disease prediction (model not loaded on the server)")

    section("Chat")
    check("empty message", call("/chat", {"message": "   "})[0], 422)

    status, first = call("/chat", {"message": "What is Krishi.AI"})
    if status == 503 and not ai_enabled:
        _skipped.append("chat conversation (run seed_faq.py, or set GEMINI_API_KEY)")
    else:
        check("first message", status, 200)
        session = first["session_id"]
        check("second message stays in the thread",
              call("/chat", {"message": "What can you do", "session_id": session})[1]["session_id"],
              session)
        check("both turns stored",
              call(f"/chat/conversations/{session}")[1]["message_count"], 4)

        saved, _token = _token, call("/auth/guest", method="POST", raw=b"")[1]["access_token"]
        check("another user cannot read the thread",
              call(f"/chat/conversations/{session}")[0], 404)
        _token = saved

        check("delete the thread", call(f"/chat/conversations/{session}", method="DELETE")[0], 204)
        check("deleted thread is gone", call(f"/chat/conversations/{session}")[0], 404)

    if not ai_enabled:
        check("unanswerable question reports 503, never a fake answer",
              call("/chat", {"message": "a question no FAQ entry covers"})[0], 503)
    else:
        _skipped.append("AI-unavailable path (GEMINI_API_KEY is set)")

    check("no empty threads are left behind",
          all(c["message_count"] > 0 for c in call("/chat/conversations")[1]), True)

    section("Market prices")
    # The upstream parsing and arithmetic are tested directly - they are the
    # parts that would quietly produce a wrong rupee figure, and they need no
    # network or API key.
    from app.services import market as market_service

    check("a row with no reported price is dropped",
          market_service._normalise({"commodity": "Wheat", "modal_price": "0"}), None)
    check("Agmarknet's -1 placeholder is dropped",
          market_service._normalise({"commodity": "Wheat", "modal_price": "-1"}), None)
    check("alternate field casing is read",
          (market_service._normalise({"Commodity": "Wheat", "Modal_Price": "2450"}) or {})
          .get("modal_price"), 2450.0)
    check("prices are quoted per quintal",
          (market_service._normalise({"commodity": "Wheat", "modal_price": "2450"}) or {})
          .get("unit"), "INR per quintal")
    # 3 t/ha over 2 ha = 60 quintal; at Rs 2450 that is Rs 147,000.
    value = market_service.estimate_value(2450.0, 3.0, 2.0)
    check("harvest value arithmetic", (value["quintals"], value["gross_value"]),
          (60.0, 147000.0))

    market_ready = call("/health", token=False)[1].get("market", {}).get("ready")
    if market_ready:
        status, body = call("/market/prices?commodity=Wheat")
        check("live prices respond", status, 200)
        check("every returned row carries a price and a date",
              all(r.get("modal_price") and r.get("arrival_date")
                  for r in body.get("records", [])), True)

        # Regression guard. data.gov.in's own filtering breaks above ~300 rows
        # per page and starts returning other states alongside the requested
        # one, so a farmer could be shown another state's rates. Ask for the
        # largest page the API accepts and assert nothing foreign came back.
        status, body = call("/market/prices?commodity=Wheat&state=Madhya+Pradesh&limit=500")
        rows = body.get("records", []) if status == 200 else []
        if rows:
            check("a state-filtered query returns only that state",
                  {r["state"] for r in rows}, {"Madhya Pradesh"})
            check("a crop-filtered query returns only that crop",
                  {r["commodity"] for r in rows}, {"Wheat"})
        else:
            _skipped.append("state-filter regression check (no wheat reported today)")
    else:
        check("prices report 503 when no API key is set, never fabricated data",
              call("/market/prices?commodity=Wheat")[0], 503)
        check("harvest value reports 503 too",
              call("/market/harvest-value",
                   {"commodity": "Wheat", "yield_tonnes_per_hectare": 3,
                    "area_hectares": 1})[0], 503)
        _skipped.append("live market price call (DATA_GOV_API_KEY is not set)")

    check("market requires authentication",
          call("/market/prices", token=False)[0], 401)

    section("Dashboard and history")
    check("dashboard", call("/dashboard")[0], 200)
    check("prediction history", call("/predictions")[0], 200)
    check("filtered history", call("/predictions?tool=crop")[0], 200)
    check("unknown tool filter", call("/predictions?tool=nope")[0], 422)

    print(f"\n{_passed} passed, {len(_failed)} failed, {len(_skipped)} skipped")
    for name in _skipped:
        print(f"  skipped: {name}")
    for name in _failed:
        print(f"  failed:  {name}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
