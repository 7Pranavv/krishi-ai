"""Add data-i18n attributes to the HTML pages from frontend/i18n/en.json.

Matches only the COMPLETE text content of an element against an English value,
so a key is applied to a whole label or heading and never to a fragment of a
sentence. Idempotent: an element that already carries data-i18n is left alone.

    python tools/annotate_i18n.py --check   # report what is untagged
    python tools/annotate_i18n.py           # apply
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

# Elements whose text is a translatable unit on its own.
TEXT_ELEMENTS = ("h1", "h2", "h3", "h4", "label", "button", "strong", "span",
                 "p", "option", "a", "div", "td", "th")


def normalise(text: str) -> str:
    """Collapse runs of whitespace so wrapped HTML matches one-line JSON.

    Without this, a paragraph broken across source lines never matched its
    catalogue entry and silently stayed in English.
    """
    return " ".join(text.replace("...", "…").split())


def reverse_map(source: dict[str, str]) -> dict[str, str]:
    """English text -> key. First key wins when two keys share a value."""
    out: dict[str, str] = {}
    for key, value in source.items():
        out.setdefault(normalise(value), key)
    return out


def annotate(html: str, lookup: dict[str, str]) -> tuple[str, int, list[str]]:
    added = 0

    def tag_element(match: re.Match) -> str:
        nonlocal added
        open_tag, attrs, text, close = match.groups()
        if "data-i18n" in attrs:
            return match.group(0)
        key = lookup.get(normalise(text))
        if not key:
            return match.group(0)
        added += 1
        return f"<{open_tag}{attrs} data-i18n=\"{key}\">{text}</{close}>"

    pattern = re.compile(
        r"<(" + "|".join(TEXT_ELEMENTS) + r")([^>]*)>([^<>{}]+?)</(\1)>")
    html = pattern.sub(tag_element, html)

    def tag_placeholder(match: re.Match) -> str:
        nonlocal added
        whole, text = match.group(0), match.group(1)
        key = lookup.get(normalise(text))
        if not key or "data-i18n-placeholder" in whole:
            return whole
        added += 1
        return f'placeholder="{text}" data-i18n-placeholder="{key}"'

    html = re.sub(r'placeholder="([^"]+)"', tag_placeholder, html)

    # Anything left that looks like prose a farmer would read.
    leftover = [
        t.strip() for t in re.findall(
            r"<(?:h1|h2|h3|label|button|strong)[^>]*>([^<>{}]{4,})</", html)
        if t.strip() and normalise(t) not in lookup and not t.strip().startswith("$")
    ]
    return html, added, leftover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args()

    source = json.loads((FRONTEND / "i18n" / "en.json").read_text(encoding="utf-8"))
    lookup = reverse_map(source)

    total = 0
    for page in sorted(FRONTEND.glob("*.html")):
        original = page.read_text(encoding="utf-8")
        updated, added, leftover = annotate(original, lookup)
        total += added
        status = f"  {page.name:<18} +{added:>3} tagged"
        if leftover:
            status += f"   still untagged: {len(leftover)}"
        print(status)
        for text in leftover[:6]:
            print(f'        "{text[:70]}"')
        if not args.check and updated != original:
            page.write_text(updated, encoding="utf-8")

    print(f"\n{'would tag' if args.check else 'tagged'} {total} elements")


if __name__ == "__main__":
    main()
