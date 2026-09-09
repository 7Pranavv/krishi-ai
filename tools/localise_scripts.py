"""Replace English string literals in the page <script> blocks with t() calls.

The HTML markup is handled by annotate_i18n.py via data-i18n attributes. This
covers the other half: strings that only exist inside JavaScript, such as empty
states, button busy labels and validation messages.

Only whole single-quoted literals that exactly match an en.json value are
touched, so no sentence is half-replaced.

    python tools/localise_scripts.py --check
    python tools/localise_scripts.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def normalise(text: str) -> str:
    """Collapse whitespace and treat "..." as the ellipsis character."""
    return " ".join(text.replace("...", "…").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = json.loads((FRONTEND / "i18n" / "en.json").read_text(encoding="utf-8"))
    by_text: dict[str, str] = {}
    for key, value in source.items():
        by_text.setdefault(normalise(value), key)

    total = 0
    for page in sorted(FRONTEND.glob("*.html")):
        html = page.read_text(encoding="utf-8")
        original = html
        replaced = 0

        def swap(match: re.Match) -> str:
            nonlocal replaced
            text = match.group(1)
            key = by_text.get(normalise(text))
            if not key:
                return match.group(0)
            replaced += 1
            return f"t('{key}')"

        # Single-quoted literals only. Template literals carry interpolation and
        # are left to be handled by hand where it matters.
        html = re.sub(r"'([^'\\\n]{4,})'", swap, html)

        # renderShell now loads the language catalogue, so it must be awaited
        # before the page paints its own strings.
        html = re.sub(r"(?<!await )renderShell\(", "await renderShell(", html)

        # Make t() available wherever it is now used.
        if "t('" in html and "renderShell" in html:
            html = re.sub(
                r"(import \{[^}]*\} from '/js/ui\.js';)",
                lambda m: m.group(1) if " t," in m.group(1) or " t " in m.group(1)
                else m.group(1).replace("} from '/js/ui.js';", ", t } from '/js/ui.js';"),
                html, count=1)

        print(f"  {page.name:<18} {replaced:>3} literals -> t()")
        total += replaced
        if not args.check and html != original:
            page.write_text(html, encoding="utf-8")

    print(f"\n{'would replace' if args.check else 'replaced'} {total} literals")


if __name__ == "__main__":
    main()
