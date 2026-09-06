#!/usr/bin/env python3
"""Assemble the three homepages from src/home/chrome.html and src/home/<lang>.html.

Each language file is a block of `key: value` lines, a `---` line, then the article body.
The chrome fills `{{key}}` slots from those lines and derives three more: the language
switcher, the "(EN)" suffix on English-only pages in the sidebar, and the table of contents
(every <h2>, plus any <h3> whose heading is a link, i.e. the package entries).

    python3 scripts/build.py          # write index.html, pl/index.html, cs/index.html
    python3 scripts/build.py --check  # exit 1 if any output is stale
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "home"
LANGS = [("en", "/", "EN"), ("pl", "/pl/", "PL"), ("cs", "/cs/", "CS")]
OUT = {"en": ROOT / "index.html", "pl": ROOT / "pl" / "index.html", "cs": ROOT / "cs" / "index.html"}


def parse(path: Path) -> tuple[dict[str, str], str]:
    front, body = path.read_text(encoding="utf-8").split("\n---\n", 1)
    meta = dict(line.split(": ", 1) for line in front.splitlines() if line.strip())
    return meta, body.strip("\n")


def langs_nav(current: str) -> str:
    items = []
    for code, home, label in LANGS:
        if code == current:
            items.append(f'        <li><span aria-current="true">{label}</span></li>')
        else:
            items.append(f'        <li><a href="{home}" hreflang="{code}">{label}</a></li>')
    return "\n".join(items)


def toc(body: str) -> str:
    sections: list[tuple[str, str, list[tuple[str, str]]]] = []
    for level, anchor, heading in re.findall(r'<h([23]) id="([^"]+)">(.*?)<a class="headerlink"', body):
        label = html.unescape(re.sub(r"<[^>]+>", "", heading)).replace(" ", " ")
        if level == "2":
            sections.append((anchor, label, []))
        elif heading.lstrip().startswith("<a href") and sections:
            sections[-1][2].append((anchor, label))
    lines = []
    for anchor, label, children in sections:
        link = f'<a href="#{anchor}" class="md-nav__link">{label}</a>'
        if not children:
            lines.append(f'        <li class="md-nav__item">{link}</li>')
            continue
        lines.append(f'        <li class="md-nav__item">{link}')
        lines.append('          <ul class="md-nav__list">')
        lines += [f'            <li class="md-nav__item"><a href="#{a}" class="md-nav__link">{t}</a></li>' for a, t in children]
        lines.append("          </ul>")
        lines.append("        </li>")
    return "\n".join(lines)


def render(lang: str) -> str:
    chrome = (SRC / "chrome.html").read_text(encoding="utf-8")
    meta, body = parse(SRC / f"{lang}.html")
    meta.update(lang=lang, langs=langs_nav(lang), toc=toc(body), body=body,
                en_suffix="" if lang == "en" else " <small>(EN)</small>")
    missing = set(re.findall(r"{{(\w+)}}", chrome)) - set(meta)
    if missing:
        sys.exit(f"{lang}: no value for {sorted(missing)}")
    return re.sub(r"{{(\w+)}}", lambda m: meta[m.group(1)], chrome)


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for lang, _, _ in LANGS:
        out = render(lang)
        target = OUT[lang]
        if check:
            if target.read_text(encoding="utf-8") != out:
                stale.append(str(target.relative_to(ROOT)))
        else:
            target.write_text(out, encoding="utf-8")
            print("wrote", target.relative_to(ROOT))
    if stale:
        print("stale:", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
