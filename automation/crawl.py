"""Crawl the authenticated competition pages and save text + links to disk."""
import os, json, re
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

OUT = os.path.join(os.path.dirname(__file__), "..", "crawl")
os.makedirs(OUT, exist_ok=True)
COMP = f"{BASE}/competitions/3/tutoring-outcomes/"

# Pages to visit by nav-link text (we resolve hrefs from the competition page nav)
WANT = [
    "Home", "Problem description", "Code submission format", "Official rules",
    "Data download", "Submissions", "Code jobs", "About", "Leaderboard",
]

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

with sync_playwright() as pw:
    browser, ctx = connect(pw)
    page = get_page(ctx)
    page.goto(COMP, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)

    # Collect all links on the competition landing page
    links = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => ({text: (e.innerText||'').trim(), href: e.href}))",
    )
    # De-dup and keep same-site
    seen = {}
    for l in links:
        h = l["href"]
        if "k12-ai-infrastructure.org" in h and h not in seen:
            seen[h] = l["text"]
    with open(os.path.join(OUT, "_all_links.json"), "w") as f:
        json.dump([{"text": t, "href": h} for h, t in seen.items()], f, indent=2)
    print("Total same-site links:", len(seen))

    # Build a map from nav text -> href
    navmap = {}
    for h, t in seen.items():
        if t and t not in navmap:
            navmap[t] = h
    print("Nav map keys sample:", [k for k in navmap.keys()][:40])

    results = {}
    for name in WANT:
        href = navmap.get(name)
        if not href:
            # fuzzy
            for t, h in navmap.items():
                if name.lower() in t.lower():
                    href = h; break
        if not href:
            print(f"[skip] no link for {name!r}")
            continue
        try:
            page.goto(href, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(1500)
            text = page.inner_text("body")
            plinks = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => ({text:(e.innerText||'').trim(), href:e.href})).filter(x=>x.href)",
            )
            # capture download-ish links & buttons
            dl = [x for x in plinks if re.search(r"download|\.csv|\.zip|\.tar|s3|blob|storage|media|/data", x["href"], re.I)]
            buttons = page.eval_on_selector_all(
                "button, input[type=submit], input[type=file], form",
                "els => els.map(e => ({tag:e.tagName, type:e.getAttribute('type'), name:e.getAttribute('name'), text:(e.innerText||e.value||'').trim().slice(0,80), action:e.getAttribute('action'), accept:e.getAttribute('accept')}))",
            )
            fn = os.path.join(OUT, f"{slug(name)}.txt")
            with open(fn, "w") as f:
                f.write(f"# {name}\nURL: {href}\n\n=== TEXT ===\n{text}\n\n=== DOWNLOAD-ISH LINKS ===\n")
                for x in dl:
                    f.write(f"{x['text']!r} -> {x['href']}\n")
                f.write("\n=== FORMS/BUTTONS/INPUTS ===\n")
                for b in buttons:
                    f.write(json.dumps(b) + "\n")
            results[name] = {"href": href, "n_download_links": len(dl), "n_form_els": len(buttons), "chars": len(text)}
            print(f"[ok] {name}: {href} | dl_links={len(dl)} form_els={len(buttons)} chars={len(text)}")
        except Exception as e:
            print(f"[err] {name}: {e}")

    with open(os.path.join(OUT, "_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved crawl to", os.path.abspath(OUT))
