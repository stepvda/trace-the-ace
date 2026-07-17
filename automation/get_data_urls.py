"""Navigate to the data-download page and emit fresh signed URLs for each file."""
import json, os
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

DATA_PAGE = f"{BASE}/competitions/3/tutoring-outcomes/data/"
LABELS = ["Train features", "Train transcripts", "Train labels",
          "Test submission format", "Smoke test submission format"]

with sync_playwright() as pw:
    browser, ctx = connect(pw)
    page = get_page(ctx)
    page.goto(DATA_PAGE, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)
    links = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => ({text:(e.innerText||'').trim(), href:e.href}))",
    )
    out = {}
    for lab in LABELS:
        for l in links:
            if l["text"] == lab and "s3" in l["href"].lower():
                out[lab] = l["href"]
                break
    print(json.dumps(out, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "_urls.json"), "w") as f:
        json.dump(out, f)
