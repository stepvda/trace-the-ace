"""Read the Code jobs page and print the status of recent jobs."""
import sys, json
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"
SUBS = f"{BASE}/competitions/3/tutoring-outcomes/submissions/"

with sync_playwright() as pw:
    browser, ctx = connect(pw)
    page = get_page(ctx)
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)
    print("=== CODE JOBS ===")
    print(page.inner_text("body"))
    # also the scored submissions page
    page.goto(SUBS, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)
    print("\n=== SUBMISSIONS ===")
    txt = page.inner_text("body")
    # print the section around submissions
    print(txt)
