"""Cancel the current (in-progress) code job and confirm the submission count."""
import re, time
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"
SUBS = f"{BASE}/competitions/3/tutoring-outcomes/submissions/"

with sync_playwright() as pw:
    b, ctx = connect(pw); page = get_page(ctx)
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(2000)
    btn = page.query_selector("button:has-text('Cancel current code job'), a:has-text('Cancel current code job')")
    if not btn:
        print("No 'Cancel current code job' button found (maybe already gone).")
    else:
        # capture confirm dialog if any
        page.on("dialog", lambda d: d.accept())
        btn.click()
        page.wait_for_timeout(1500)
        # a confirmation modal may appear
        for sel in ["button:has-text('Cancel job')", "button:has-text('Confirm')",
                    "button:has-text('Yes')", "button:has-text('Cancel submission')"]:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click(); print("clicked confirm:", sel); break
        page.wait_for_timeout(2500)
        print("cancel clicked.")
    # report
    page.goto(URL, wait_until="domcontentloaded", timeout=45000); page.wait_for_timeout(2000)
    body = page.inner_text("body")
    m = re.search(r"Code jobs\s*\n(.*?)Stay informed", body, re.S)
    print("=== code jobs after cancel ===")
    print(" | ".join([l.strip() for l in (m.group(1) if m else body).splitlines() if l.strip()][:16]))
    page.goto(SUBS, wait_until="domcontentloaded", timeout=45000); page.wait_for_timeout(2000)
    b2 = page.inner_text("body")
    for l in b2.splitlines():
        if re.search(r"of 3|submissions left", l, re.I) and l.strip():
            print("COUNTER:", l.strip())
