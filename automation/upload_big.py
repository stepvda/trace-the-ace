"""Upload a LARGE submission (>50MB) that the CDP path can't handle.

Playwright's connect_over_cdp caps file transfers at 50MB ("browser not
co-located"). Instead we LAUNCH the SAME automation profile via
launch_persistent_context (Playwright-owned => no cap), preserving the login.

Usage: python upload_big.py <zip> <smoke|normal> "<note>"
Runs launch -> (login check) -> upload -> monitor to terminal, then closes.
"""
import sys, re, time
from playwright.sync_api import sync_playwright

PROFILE = ("/private/tmp/claude-502/-Users-nstephane-Dev-AI-Data-Science-training-"
           "trace-the-ace/648de05e-289a-4509-a6b1-01c3253db3d6/scratchpad/edge-auto-profile")
BASE = "https://platform.k12-ai-infrastructure.org"
URL = f"{BASE}/competitions/3/submissions/code/"


def newest_status(page):
    body = page.inner_text("body")
    ids = re.findall(r"id-(\d+)", body)
    nid = sorted(ids, key=int)[-1] if ids else None
    if not nid:
        return None, "?", body
    # window around the newest job id (status word precedes the id in the row)
    idx = body.find(f"id-{nid}")
    win = body[max(0, idx - 400):idx + 200]
    if "not yet finished uploading" in win:
        return nid, "Uploading", body
    st = next((w for w in ["Scoring", "Running", "Starting", "Pending", "Completed",
                           "Failed", "Canceled", "Uploading"] if re.search(r"\b" + w + r"\b", win)), "?")
    return nid, st, body


def main(zip_path, mode, note):
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            PROFILE, channel="msedge", headless=False,
            args=["--no-first-run", "--no-default-browser-check"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        body = page.inner_text("body")
        if "Log out" not in body and "Logout" not in body:
            print("NOT_LOGGED_IN — waiting up to 180s for you to log in in the window...", flush=True)
            for _ in range(18):
                page.wait_for_timeout(10000)
                page.goto(URL, wait_until="domcontentloaded", timeout=45000); page.wait_for_timeout(1500)
                if "Log out" in page.inner_text("body"):
                    print("login detected", flush=True); break
            else:
                print("STILL_NOT_LOGGED_IN — aborting"); ctx.close(); return 2
        pre = set(re.findall(r"id-(\d+)", page.inner_text("body")))

        # A stuck / in-progress job blocks new submissions (the platform allows only one
        # active job at a time and shows "Cancel current code job" instead of the
        # "New code submission" button). Cancel it first so the form becomes available.
        try:
            page.on("dialog", lambda d: d.accept())
            for _ in range(3):
                page.goto(URL, wait_until="domcontentloaded", timeout=45000); page.wait_for_timeout(2000)
                cb = page.query_selector("button:has-text('Cancel current code job'), a:has-text('Cancel current code job')")
                if not cb:
                    break
                print("cancelling a stuck in-progress job to free the queue...", flush=True)
                cb.click(); page.wait_for_timeout(1500)
                for sel in ["button:has-text('Cancel job')", "button:has-text('Confirm')",
                            "button:has-text('Yes')", "button:has-text('Cancel submission')"]:
                    el = page.query_selector(sel)
                    if el and el.is_visible():
                        el.click(); break
                page.wait_for_timeout(5000)
        except Exception as e:
            print(f"cancel step note: {str(e)[:70]}", flush=True)

        # Robustly open the upload form — the button can lag after login/navigation,
        # so re-navigate + wait for it to be visible, and retry a few times.
        opened = False
        for attempt in range(4):
            try:
                page.goto(URL, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_selector("button:has-text('New code submission')", state="visible", timeout=30000)
                page.click("button:has-text('New code submission')")
                page.wait_for_selector("#id_form input[type=file]", state="attached", timeout=20000)
                opened = True
                break
            except Exception as e:
                print(f"open-form attempt {attempt+1}/4 failed ({str(e)[:70]}); retrying...", flush=True)
                page.wait_for_timeout(4000)
        if not opened:
            print("COULD_NOT_OPEN_FORM — aborting (check the page manually)"); ctx.close(); return 3
        print("uploading (no 50MB cap in launch mode)...", flush=True)
        t0 = time.time()
        page.query_selector("#id_form input[type=file]").set_input_files(zip_path, timeout=600000)
        print(f"file set in {int(time.time()-t0)}s", flush=True)
        page.wait_for_timeout(1000)
        target = "Smoke test" if mode == "smoke" else "Normal submission"
        tog = page.query_selector(f"#id_form button:has-text('{target}')")
        if tog: tog.click(); page.wait_for_timeout(400)
        try: page.fill("#id_private_note", note)
        except Exception: pass
        try: page.select_option("#id_email_on", label="Finished")
        except Exception: pass
        page.query_selector("#id_form button[type=submit]").click()
        # CRITICAL: the ~700MB upload runs asynchronously on THIS tab. Navigating this
        # tab (page.goto) aborts the in-flight upload -> the job hangs on "Uploading"
        # forever. So we monitor from a SEPARATE tab and never touch the upload tab.
        print("submitted; monitoring from a separate tab (upload tab left untouched)...", flush=True)
        mon = ctx.new_page()

        t0 = time.time(); newid = None; last = None; up_confirmed = False
        terminal = {"Completed", "Failed", "Canceled"}
        while time.time() - t0 < 9000:          # full ModernBERT fine-tune can take a while
            mon.wait_for_timeout(12000)
            try:
                mon.goto(URL, wait_until="domcontentloaded", timeout=45000); mon.wait_for_timeout(1500)
            except Exception:
                continue
            nid, st, body = newest_status(mon)
            newid = nid
            fresh = nid and int(nid) not in {int(x) for x in pre}
            if (nid, st) != last:
                print(f"[{int(time.time()-t0)}s] id-{nid} status={st} fresh={fresh}", flush=True); last = (nid, st)
            if not fresh:
                continue
            if st not in ("Uploading", "?") and not up_confirmed:
                up_confirmed = True
                print(f"UPLOAD_COMPLETE id-{nid} queued (safe on the server now)", flush=True)
            if st in terminal:                 # run to the end and report the score
                idx = body.find(f"id-{nid}"); blk = body[max(0, idx - 300):idx + 30]
                sc = re.search(r"0\.\d{3,4}", blk)
                print(f"TERMINAL id-{nid} status={st} score={sc.group() if sc else None}", flush=True)
                print("CONTEXT:", blk.replace("\n", " | "), flush=True)
                break
        print(f"DONE id-{newid}", flush=True)
        ctx.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "smoke",
         sys.argv[3] if len(sys.argv) > 3 else "container-trainer smoke")
