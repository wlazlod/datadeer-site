# datadeer.pl

Personal site for Daniel Wlazło. Static HTML/CSS served by GitHub Pages with
`datadeer.pl` as the custom domain. Trilingual homepage (EN default, PL, CS);
notes and course pages are English only.

## Local preview

```bash
python3 -m http.server 8000
# http://localhost:8000        English
# http://localhost:8000/pl/    Polish
# http://localhost:8000/cs/    Czech
```

## Files

```
.
├── index.html, pl/, cs/      # generated homepages — edit src/home/, then build
├── src/home/
│   ├── chrome.html           # shared header, sidebars, footer with {{slots}}
│   └── en.html, pl.html, cs.html   # per-language strings + article body
├── scripts/
│   ├── build.py              # assembles the homepages; --check flags stale output
│   └── figures/              # generators for the note figures (not referenced by pages)
├── notes/                    # standalone notes, hand-edited
├── didactics/                # course pages, hand-edited
├── 404.html                  # served by GitHub Pages for missing paths
├── assets/
│   ├── style.css             # one stylesheet for every page
│   ├── app.js                # palette toggle, TOC highlighting
│   ├── fonts/                # self-hosted Roboto + Roboto Mono (latin, latin-ext)
│   ├── img/                  # headshot, Open Graph card, note figures
│   └── favicon.svg
├── sitemap.xml, robots.txt
├── CNAME                     # custom-domain marker for GitHub Pages
└── .nojekyll                 # serve files as-is
```

Everything committed is served. Source material that must not ship (PDFs) is
excluded through `.gitignore`.

## Editing the homepage

The three homepages are generated. Edit `src/home/<lang>.html`, then:

```bash
python3 scripts/build.py          # rewrites index.html, pl/index.html, cs/index.html
python3 scripts/build.py --check  # exits 1 if the generated files are stale
```

Each language file starts with `key: value` lines (page metadata and the
translated chrome strings), a `---` line, and the article body. The table of
contents, the language switcher, and the "(EN)" markers in the sidebar are
derived by the script, so they cannot drift between languages. Chrome changes
(header, footer, sidebar) go in `chrome.html` once.

Package versions in the Open source section are static text. Bump them in all
three language files when probcal or treecf releases.

## Adding a note

1. Create `notes/<slug>.html` from an existing note (they share the chrome by copy).
2. Add it to the Notes list in the three `src/home/*.html` bodies (newest first),
   to the sidebar in `chrome.html`, and to the sidebar of the other notes.
3. If it is the newest note, point `next_href` / `next_label` in the three
   language files at it.
4. Add a `<url>` entry to `sitemap.xml`.
5. Rebuild.

## Palette and fonts

The colour scheme is chosen by an inline script at the top of `<body>`: the
stored choice wins, otherwise `prefers-color-scheme`. Fonts are self-hosted
under `assets/fonts/`; no request leaves the domain.

## Deploy

Push to `main`. GitHub Pages serves the root of the branch. DNS for the apex
points at the GitHub Pages IPs; `www` is a CNAME to `wlazlod.github.io`.

## License

Site code: MIT. Content: © Daniel Wlazło.
