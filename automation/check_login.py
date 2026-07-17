"""Check whether the user is logged in on the competition site.

Prints a compact status line. Exit code 0 = logged in, 2 = not logged in.
"""
import sys
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

COMP = f"{BASE}/competitions/3/tutoring-outcomes/"

with sync_playwright() as pw:
    browser, ctx = connect(pw)
    page = get_page(ctx)
    try:
        page.goto(COMP, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print("NAV_ERR", e)
    page.wait_for_timeout(1200)
    html = page.content()
    url = page.url

    # Heuristics for auth state
    body_text = page.inner_text("body")[:6000] if page.query_selector("body") else ""
    has_login_link = ("/accounts/login" in html) or ("Log in" in body_text) or ("Login" in body_text and "Logout" not in body_text)
    has_logout = ("/accounts/logout" in html) or ("Log out" in body_text) or ("Logout" in body_text) or ("Sign out" in body_text)
    # DrivenData-style: when logged in you often see your username / "My submissions" / avatar
    signed_in_markers = [m for m in ["Logout", "Log out", "Sign out", "My Submissions", "My submissions", "Submissions", "Account", "Dashboard"] if m in body_text]

    logged_in = has_logout or (not has_login_link and len(signed_in_markers) > 0)

    print("URL:", url)
    print("has_login_link:", has_login_link, "| has_logout:", has_logout)
    print("signed_in_markers:", signed_in_markers)
    print("LOGGED_IN:" , logged_in)
    # Dump the nav/menu area text for debugging
    print("--- body head ---")
    print(body_text[:800].replace("\n\n", "\n"))
    sys.exit(0 if logged_in else 2)
