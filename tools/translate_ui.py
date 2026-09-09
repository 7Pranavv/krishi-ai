"""Generate the interface translations from frontend/i18n/en.json.

Run this when English strings change:

    python tools/translate_ui.py            # only missing keys
    python tools/translate_ui.py --all      # retranslate everything
    python tools/translate_ui.py --lang hi  # one language

Output is static JSON under frontend/i18n/, so the browser never calls a
translation API - the cost is paid once here, and the files are reviewable and
diffable. English is the source of truth; any key missing from a translation
falls back to English at runtime rather than showing a key name.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
I18N_DIR = ROOT / "frontend" / "i18n"
SOURCE = I18N_DIR / "en.json"

# The second catalogue: agronomic advice returned inside API responses. Same
# machinery, different source file, produced by tools/extract_content.py.
CONTENT_SOURCE = I18N_DIR / "content.en.json"

# Must match LANGUAGES in frontend/js/i18n.js.
LANGUAGES = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "te": "Telugu",
    "ta": "Tamil", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi (Gurmukhi script)", "or": "Odia", "as": "Assamese",
    "ur": "Urdu (Nastaliq script)",
}

# Terms that must survive untranslated: the brand, agronomic abbreviations,
# file formats and institution names farmers recognise in Latin script.
# Only genuine proper nouns, symbols and unit codes. Ordinary words are NOT
# listed here: an earlier version included "N, P, K, pH" and several models
# read that as licence to leave the whole label "Nitrogen (N)" in English.
KEEP_AS_IS = (
    "Krishi.AI, data.gov.in, ICAR, IMD, KVK, "
    "the letters N, P, K and NPK when used as chemical symbols, "
    "DAP, MOP, pH, JPG, PNG, WEBP, MB, mm, kg/ha, t/ha"
)

PROMPT = """You are translating the interface of Krishi.AI, an agricultural
advisory app used by farmers in India. Translate the JSON values below into
{language}.

Rules:
- Return ONLY a JSON object. Same keys, translated values. No commentary.
- Translate for a farmer with limited schooling. Use plain, everyday words -
  the words a farmer would actually say, not formal or literary register.
- Keep these EXACTLY as they appear, untranslated: {keep}
- Keep any punctuation that carries meaning: the middot separators, arrows,
  curly quotes and the ellipsis character.
- Keep it short. These are buttons, labels and headings in a layout - a
  translation twice the length of the English will break the design. (For
  advisory sentences, accuracy matters more than length - never drop a
  quantity, timing or safety warning to save space.)
- "Modal" in a price table means the most common price (modal value), not a
  dialog box.
- "mandi" is the local market; use the word farmers use in {language}.
- Translate the WORDS even where a symbol stays. "Nitrogen (N)" becomes the
  {language} word for nitrogen followed by "(N)". "Soil pH" becomes the
  {language} words for soil, then "pH". Never return a value unchanged from
  the English unless it is a proper noun in the keep-list.

JSON to translate:
{payload}"""

# The whole catalogue in one request. The free tier caps REQUESTS per day, not
# tokens, so many small batches is the expensive shape - 297 short strings fit
# comfortably inside one reply.
BATCH_SIZE = 400

# Filename prefix, so the advice catalogue writes content.<lang>.json.
PREFIX = ""

# Each model carries its own daily quota, so rotating on 429 multiplies what a
# free key can do. Ordered by preference; all were confirmed available.
# gemini-2.5-pro is deliberately absent: it answers 404 "no longer available
# to new users" on a free key, and a model that always 404s poisons the
# rotation for every language that follows.
MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-pro-latest",
]
_model_index = 0
_dead: set[str] = set()


def client():
    key = os.environ.get("GEMINI_API_KEY") or _key_from_env_file()
    if not key:
        sys.exit("GEMINI_API_KEY is not set (checked the environment and .env)")
    from google import genai
    from google.genai import types

    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=180_000))


def _key_from_env_file() -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def current_model() -> str:
    return MODELS[_model_index]


def next_model(reason: str, permanent: bool = False) -> bool:
    """Advance to the next usable model. False when none are left.

    A permanently broken model (404) is remembered, otherwise one bad model
    would be retried for every remaining language.
    """
    global _model_index
    if permanent:
        _dead.add(MODELS[_model_index])
    for index in range(_model_index + 1, len(MODELS)):
        if MODELS[index] not in _dead:
            _model_index = index
            print(f"    {reason}; switching to {MODELS[index]}")
            return True
    # Wrap around in case an earlier model's quota window has reopened.
    for index in range(0, _model_index):
        if MODELS[index] not in _dead:
            _model_index = index
            print(f"    {reason}; retrying with {MODELS[index]}")
            return True
    return False


def translate_batch(api, batch: dict[str, str], language: str) -> dict[str, str]:
    from google.genai import types

    prompt = PROMPT.format(
        language=language, keep=KEEP_AS_IS,
        payload=json.dumps(batch, ensure_ascii=False, indent=1),
    )
    response = api.models.generate_content(
        model=current_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            # Asking for JSON explicitly avoids the model wrapping it in
            # markdown fences, which then need stripping.
            response_mime_type="application/json",
            max_output_tokens=32768,
            temperature=0.3,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("empty response")
    result = json.loads(text)
    if not isinstance(result, dict):
        raise RuntimeError(f"expected an object, got {type(result).__name__}")
    return result


def translate_language(api, code: str, language: str, source: dict, retranslate: bool) -> None:
    target_path = I18N_DIR / f"{PREFIX}{code}.json"
    existing = {}
    if target_path.exists() and not retranslate:
        existing = json.loads(target_path.read_text(encoding="utf-8"))

    todo = {k: v for k, v in source.items() if k not in existing}
    if not todo:
        print(f"  {code} ({language}): already complete ({len(existing)} keys)")
        return

    print(f"  {code} ({language}): {len(todo)} keys to translate")
    keys = list(todo)
    for start in range(0, len(keys), BATCH_SIZE):
        chunk = {k: todo[k] for k in keys[start:start + BATCH_SIZE]}
        for attempt in range(3):
            try:
                got = translate_batch(api, chunk, language)
                missing = set(chunk) - set(got)
                if missing:
                    raise RuntimeError(f"{len(missing)} keys missing from the reply")
                # Only keep keys we asked for, so a hallucinated extra key
                # cannot enter the catalogue.
                existing.update({k: str(got[k]).strip() for k in chunk})
                print(f"    {start + len(chunk):>3}/{len(keys)}")
                break
            except Exception as exc:  # noqa: BLE001
                text = str(exc)
                if "404" in text or "no longer available" in text:
                    # This model will never work with this key.
                    if next_model("model unavailable", permanent=True):
                        continue
                elif "RESOURCE_EXHAUSTED" in text or "429" in text:
                    if next_model("quota reached"):
                        continue
                elif "503" in text or "UNAVAILABLE" in text:
                    # Transient overload: wait, then try another model.
                    time.sleep(4)
                    if next_model("model busy"):
                        continue
                if attempt == 2:
                    print(f"    FAILED batch at {start}: {text[:140]}")
                else:
                    time.sleep(3 * (attempt + 1))

    # Preserve the English key order so the files diff cleanly.
    ordered = {k: existing[k] for k in source if k in existing}
    target_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = len(ordered)
    print(f"  {code}: wrote {done}/{len(source)} keys"
          + ("" if done == len(source) else "  (rest falls back to English)"))


def untranslated(source: dict, translated: dict) -> list[str]:
    """Keys whose value came back byte-identical to the English.

    Usually a model deciding a technical label should stay in English. The
    brand name is the one legitimate case.
    """
    allowed = {"Krishi.AI"}
    return [k for k, v in translated.items()
            if v.strip() == source.get(k, "").strip() and v.strip() not in allowed]


def repair(api, code: str, language: str, source: dict) -> None:
    """Retranslate only the keys that came back in English."""
    path = I18N_DIR / f"{PREFIX}{code}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    stuck = untranslated(source, data)
    if not stuck:
        print(f"  {code} ({language}): nothing left in English")
        return

    print(f"  {code} ({language}): repairing {len(stuck)} untranslated key(s)")
    chunk = {k: source[k] for k in stuck}
    for attempt in range(3):
        try:
            got = translate_batch(api, chunk, language)
            fixed = {k: str(got[k]).strip() for k in chunk if k in got}
            still = [k for k, v in fixed.items() if v == source[k].strip()]
            data.update(fixed)
            path.write_text(
                json.dumps({k: data[k] for k in source if k in data},
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"    fixed {len(fixed) - len(still)}/{len(chunk)}"
                  + (f", {len(still)} still English" if still else ""))
            return
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if ("404" in text and next_model("model unavailable", permanent=True)) or                (("429" in text or "RESOURCE_EXHAUSTED" in text) and next_model("quota reached")) or                (("503" in text or "UNAVAILABLE" in text) and next_model("model busy")):
                continue
            if attempt == 2:
                print(f"    repair FAILED: {text[:120]}")
            else:
                time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", help="only this language code")
    parser.add_argument("--all", action="store_true",
                        help="retranslate existing keys instead of only missing ones")
    parser.add_argument("--repair", action="store_true",
                        help="retranslate only keys that came back in English")
    parser.add_argument("--content", action="store_true",
                        help="translate the agronomic advice catalogue instead")
    args = parser.parse_args()

    global SOURCE, PREFIX
    if args.content:
        SOURCE = CONTENT_SOURCE
        PREFIX = "content."
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    print(f"source: {len(source)} keys from {SOURCE.relative_to(ROOT)}")

    targets = {args.lang: LANGUAGES[args.lang]} if args.lang else LANGUAGES
    api = client()

    if args.repair:
        for code, language in targets.items():
            repair(api, code, language, source)
        return

    for attempt in (1, 2):
        remaining = {}
        for code, language in targets.items():
            path = I18N_DIR / f"{PREFIX}{code}.json"
            have = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0
            if have < len(source) or (args.all and attempt == 1):
                remaining[code] = language
        if not remaining:
            break
        if attempt == 2:
            print(f"\nsecond pass for {len(remaining)} incomplete language(s)")
        for code, language in remaining.items():
            translate_language(api, code, language, source, args.all and attempt == 1)


if __name__ == "__main__":
    main()
