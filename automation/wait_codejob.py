"""Poll the Code jobs page until the most recent job reaches a terminal state.

Prints a status line each poll; exits when the job succeeds/fails/cancels or on
timeout. Terminal detection is keyword-based and tolerant of wording.
"""
import sys, re, time
from playwright.sync_api import sync_playwright
from browser import connect, get_page, BASE

URL = f"{BASE}/competitions/3/submissions/code/"
# Still RUNNING while the cancel button shows or the status is an active state.
# NB: do NOT include the static "live updates until…" help text (always present).
RUNNING = re.compile(
    r"Cancel current code job|"
    r"\b(pending|queued|running|processing|executing|starting|building|uploading|scoring)\b",
    re.I,
)
# Terminal status words that appear in the job's Status cell when finished:
DONE = re.compile(r"\b(completed|failed|errored|cancell?ed|invalid|rejected)\b", re.I)


def snapshot(page):
    page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(1500)
    txt = page.inner_text("body")
    m = re.search(r"Code jobs\s*\n(.*?)Stay informed", txt, re.S)
    section = m.group(1) if m else txt
    return section.strip()


def main(timeout_s=1200, interval=45):
    t0 = time.time()
    with sync_playwright() as pw:
        browser, ctx = connect(pw)
        page = get_page(ctx)
        while time.time() - t0 < timeout_s:
            sec = snapshot(page)
            line = " | ".join([l.strip() for l in sec.splitlines() if l.strip()][:14])
            running = bool(RUNNING.search(sec))
            done = bool(DONE.search(sec))
            elapsed = int(time.time() - t0)
            print(f"[{elapsed}s] running={running} done={done} :: {line[:260]}", flush=True)
            if done and not running:
                print("JOB_TERMINAL", flush=True)
                print("---- full section ----", flush=True)
                print(sec, flush=True)
                return 0
            time.sleep(interval)
        print("POLL_TIMEOUT", flush=True)
        print(snapshot(page), flush=True)
        return 2


if __name__ == "__main__":
    to = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    sys.exit(main(to))
