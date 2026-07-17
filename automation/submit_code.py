"""Upload a code submission (submission.zip) to the Code jobs page.

Usage: python submit_code.py /path/to/submission.zip [smoke|normal] ["private note"]

Selects the Smoke test or Normal submission toggle, uploads the zip, and clicks
Submit. Prints the resulting job state. Does NOT poll long-running jobs; use
poll_codejobs.py for that.
"""
import sys, time, json
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"


def main(zip_path, mode="smoke", note="automated: sklearn tfidf+lr/hgb blend"):
    assert mode in ("smoke", "normal")
    with sync_playwright() as pw:
        browser, ctx = connect(pw)
        page = get_page(ctx)
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1200)

        # open the form
        btn = page.query_selector("button:has-text('New code submission')")
        if btn:
            btn.click(); page.wait_for_timeout(1000)

        form = page.query_selector("#id_form")
        if not form:
            print("ERROR: upload form not found"); return 1

        # set the file
        file_input = page.query_selector("#id_form input[type=file]") or page.query_selector("input[type=file]")
        file_input.set_input_files(zip_path)
        page.wait_for_timeout(500)

        # email notification -> Finished (so the user gets notified when it completes)
        try:
            page.select_option("#id_email_on", label="Finished")
        except Exception:
            pass

        # private note
        try:
            page.fill("#id_private_note", note)
        except Exception:
            pass

        # choose mode toggle
        target_text = "Smoke test" if mode == "smoke" else "Normal submission"
        toggle = page.query_selector(f"#id_form button:has-text('{target_text}')")
        if toggle:
            toggle.click(); page.wait_for_timeout(400)
        # report active state of both toggles for confirmation
        toggles = page.eval_on_selector_all(
            "#id_form button",
            "els=>els.filter(e=>/Smoke test|Normal submission/.test(e.innerText)).map(e=>({t:e.innerText.trim(), cls:e.className, active:/active|selected|btn-primary/.test(e.className)}))",
        )
        print("toggle states:", json.dumps(toggles))

        # submit
        submit = page.query_selector("#id_form button[type=submit]")
        print(f"submitting {mode} job with zip {zip_path} ...")
        submit.click()
        try:
            page.wait_for_load_state("networkidle", timeout=120000)
        except Exception:
            pass

        # verify the job registered by reloading the code jobs page until a job row appears
        import re
        registered = False
        for i in range(12):
            page.goto(URL, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)
            body = page.inner_text("body")
            if re.search(r"Cancel current code job|\b(Pending|Starting|Running|Queued|Scoring|Completed)\b", body):
                registered = True
                m = re.search(r"Code jobs\s*\n(.*?)Stay informed", body, re.S)
                sec = (m.group(1) if m else body).strip()
                print(f"JOB REGISTERED (attempt {i+1}). Current code-jobs section:")
                print(" | ".join([l.strip() for l in sec.splitlines() if l.strip()][:16]))
                break
        if not registered:
            print("WARNING: could not confirm job registration; check the browser.")
            print(page.inner_text("body")[:1200])
        return 0 if registered else 3


if __name__ == "__main__":
    zip_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "smoke"
    note = sys.argv[3] if len(sys.argv) > 3 else "automated: sklearn tfidf+lr/hgb blend"
    sys.exit(main(zip_path, mode, note))
