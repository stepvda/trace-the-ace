"""Open the Code jobs page, reveal the upload form, and dump its exact structure."""
import json
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"

with sync_playwright() as pw:
    browser, ctx = connect(pw)
    page = get_page(ctx)
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)

    # Try to click "New code submission" to reveal the form/modal
    for sel in ["text=New code submission", "button:has-text('New code submission')"]:
        try:
            el = page.query_selector(sel)
            if el:
                el.click()
                page.wait_for_timeout(1200)
                print("clicked:", sel)
                break
        except Exception as e:
            print("click err", sel, e)

    page.wait_for_timeout(800)
    # Dump all form-related elements with attributes
    els = page.eval_on_selector_all(
        "form, input, button, label, [role=radio], [role=switch], select, textarea",
        """els => els.map(e => ({
            tag: e.tagName,
            type: e.getAttribute('type'),
            name: e.getAttribute('name'),
            id: e.id || null,
            value: e.getAttribute('value'),
            role: e.getAttribute('role'),
            accept: e.getAttribute('accept'),
            classes: (e.className && e.className.baseVal!==undefined)? e.className.baseVal : (e.className||''),
            text: (e.innerText||e.value||'').trim().slice(0,60),
            visible: !!(e.offsetParent!==null || e.getClientRects().length)
        }))"""
    )
    print(json.dumps(els, indent=1))
