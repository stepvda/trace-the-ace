"""Robust code submission: after clicking Submit, NEVER navigate the upload tab
(the 28 MB upload is async and dies if the tab navigates). Monitor progress from
a SEPARATE tab until the new job leaves the 'Uploading' state.

Usage: python submit_code_v2.py <zip> [smoke|normal] ["note"]
"""
import sys, re, time
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"


def newest_status(mon):
    mon.goto(URL, wait_until="domcontentloaded", timeout=45000)
    mon.wait_for_timeout(1500)
    body = mon.inner_text("body")
    top = body.split("id-")[0] if "id-" in body else body  # text before first job id ~ newest row status
    # newest job id
    ids = re.findall(r"id-(\d+)", body)
    newest_id = ids[0] if ids else None
    status = "?"
    for w in ["Uploading", "Pending", "Starting", "Running", "Scoring", "Completed", "Failed", "Canceled", "Cancelled", "Errored"]:
        if re.search(r"\b" + w + r"\b", top):
            status = w
    return newest_id, status, body


def main(zip_path, mode="normal", note="automated submission"):
    assert mode in ("smoke", "normal")
    with sync_playwright() as pw:
        b, ctx = connect(pw)
        page = get_page(ctx)

        # baseline: how many job ids exist before we submit
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1500)
        pre_ids = set(re.findall(r"id-(\d+)", page.inner_text("body")))

        # open + fill the form
        btn = page.query_selector("button:has-text('New code submission')")
        if btn:
            btn.click(); page.wait_for_timeout(1000)
        assert page.query_selector("#id_form"), "upload form not found"
        (page.query_selector("#id_form input[type=file]") or page.query_selector("input[type=file]")).set_input_files(zip_path)
        page.wait_for_timeout(500)
        try: page.select_option("#id_email_on", label="Finished")
        except Exception: pass
        try: page.fill("#id_private_note", note)
        except Exception: pass
        target = "Smoke test" if mode == "smoke" else "Normal submission"
        tog = page.query_selector(f"#id_form button:has-text('{target}')")
        if tog: tog.click(); page.wait_for_timeout(400)
        states = page.eval_on_selector_all("#id_form button",
            "els=>els.filter(e=>/Smoke test|Normal submission/.test(e.innerText)).map(e=>({t:e.innerText.trim(),active:/active|btn-primary/.test(e.className)}))")
        print("toggle states:", states, flush=True)

        # SUBMIT — then do NOT touch this tab again
        page.query_selector("#id_form button[type=submit]").click()
        print(f"clicked Submit ({mode}); leaving upload tab alone, monitoring in a separate tab...", flush=True)

        mon = ctx.new_page()
        t0 = time.time(); ok = False; new_id = None
        try:
            while time.time() - t0 < 360:
                mon.wait_for_timeout(6000)
                nid, status, body = newest_status(mon)
                now_ids = set(re.findall(r"id-(\d+)", body))
                fresh = now_ids - pre_ids
                el = int(time.time() - t0)
                print(f"[{el}s] newest_id={nid} status={status} new_ids={sorted(fresh)}", flush=True)
                if status in ("Pending", "Starting", "Running", "Scoring", "Completed"):
                    ok = True; new_id = nid; break
                if status in ("Failed", "Canceled", "Cancelled", "Errored"):
                    new_id = nid; break
        finally:
            mon.close()
        if ok:
            print(f"UPLOAD_COMPLETE job id-{new_id} is now '{status if 'status' in dir() else ''}' running.", flush=True)
            return 0
        print("UPLOAD_NOT_CONFIRMED (still uploading or failed). Check the browser.", flush=True)
        return 3


if __name__ == "__main__":
    zp = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
    note = sys.argv[3] if len(sys.argv) > 3 else "automated submission"
    sys.exit(main(zp, mode, note))
