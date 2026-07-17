"""Shared helper: connect to the visible Edge instance over CDP.

The Edge window was launched with --remote-debugging-port=9333 on a dedicated
automation profile, so we attach to the already-running, user-visible browser
(never headless) and reuse whatever the user is logged into.
"""
import sys
from playwright.sync_api import sync_playwright

CDP_URL = "http://127.0.0.1:9333"
BASE = "https://platform.k12-ai-infrastructure.org"


def connect(pw):
    """Return (browser, context) attached to the running Edge via CDP."""
    browser = pw.chromium.connect_over_cdp(CDP_URL)
    context = browser.contexts[0]
    return browser, context


def get_page(context, prefer_host="platform.k12-ai-infrastructure.org"):
    """Pick an existing page (prefer one already on the platform host), else open one."""
    pages = [p for p in context.pages]
    for p in pages:
        try:
            if prefer_host in (p.url or ""):
                return p
        except Exception:
            pass
    if pages:
        return pages[0]
    return context.new_page()


if __name__ == "__main__":
    with sync_playwright() as pw:
        browser, ctx = connect(pw)
        print("Connected. Contexts:", len(browser.contexts))
        for p in ctx.pages:
            print(" tab:", (p.url or "")[:100])
