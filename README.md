# datadeer.pl

Personal site for Daniel Wlazło. Static HTML/CSS, no build step, deployed
via GitHub Pages with `datadeer.pl` as the custom domain. Trilingual: EN
(default), PL, CS.

## Local preview

```bash
cd /home/danielwlazlo/Code/datadeer.pl
python3 -m http.server 8000
# open http://localhost:8000        (English)
# open http://localhost:8000/pl/    (Polish)
# open http://localhost:8000/cs/    (Czech)
```

## Files

```
.
├── index.html            # English (root)
├── pl/index.html         # Polish
├── cs/index.html         # Czech
├── assets/
│   ├── style.css         # one stylesheet, all three pages
│   └── favicon.svg
├── CNAME                 # custom-domain marker for GitHub Pages
├── .nojekyll             # tell Pages: skip Jekyll, serve files as-is
├── .gitignore            # excludes Profile.pdf etc. from deploys
├── robots.txt
└── README.md
```

`Profile.pdf` is intentionally not committed — it's source material, not a
deployable asset. Edit `.gitignore` if that ever needs to change.

## Editing copy

Each language is its own HTML file. There's no shared layout file because
there isn't a build step. When you change a section heading or paragraph,
update all three. The structure is identical across languages, so a copy/
diff/translate workflow works fine.

To add the next language, copy `pl/index.html`, change `<html lang="…">`,
the `hreflang` tags, the language switcher's `current` marker, and the
content. Add a row to the language switcher in all existing pages.

## Deploy to GitHub Pages

1. Create the repo on GitHub: `wlazlod/datadeer-site` (public).
2. Push this directory:

   ```bash
   cd /home/danielwlazlo/Code/datadeer.pl
   git init
   git add -A
   git commit -m "feat: initial trilingual site"
   git branch -M main
   git remote add origin git@github.com:wlazlod/datadeer-site.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages**.
   - Source: *Deploy from a branch*
   - Branch: `main` / `(root)`
   - Custom domain: `datadeer.pl`
   - Tick **Enforce HTTPS** once the cert is provisioned (a few minutes
     after DNS lands).

## DNS — switch `datadeer.pl` from WordPress to GitHub Pages

At your registrar (the one currently pointing `datadeer.pl` at the WP host),
replace the existing A records for the apex with the four GitHub Pages IPs,
and point `www` at your `github.io` URL.

**Apex `datadeer.pl` — A records (IPv4):**

```
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

**Apex `datadeer.pl` — AAAA records (IPv6, optional but recommended):**

```
2606:50c0:8000::153
2606:50c0:8001::153
2606:50c0:8002::153
2606:50c0:8003::153
```

**Subdomain `www` — CNAME:**

```
www  →  wlazlod.github.io.
```

(Reference: GitHub docs on [apex domain configuration](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site#configuring-an-apex-domain).)

### Cutover plan (no downtime risk)

1. Push the site, configure custom domain in Pages → wait for the green
   check under *Custom domain*.
2. Lower the TTL on the existing WordPress DNS records to 300 s, **the day
   before** you flip.
3. Replace the records as above.
4. Once `https://datadeer.pl/` serves the new site, decommission the
   WordPress hosting at your leisure. Keep a backup of the old site if it
   has anything you might want to reuse.

## Adding writing later

When you have your first essay:

1. Create `notes/<slug>.html` (copy `index.html` as a starting layout, swap
   the body for the essay content).
2. Replace each language's *Notes / Notatki / Poznámky* placeholder with a
   list linking to the post (translate the title in each language file).

When the essay count justifies it, switch to a static-site generator
(11ty / Astro / Hugo) without changing the URL structure.

## License

Site code: MIT. Content: © Daniel Wlazło.
