"""Resilient poll of a specific code job until terminal, printing its public score.

Robust to the debug-Edge tab being closed mid-poll: it reconnects over CDP and
opens a FRESH page every iteration (never reuses a stale tab), so a closed tab or
transient nav error is retried rather than fatal.

Usage: python poll_job.py <job_id> [timeout_s] [interval_s]
"""
import sys, re, time
from playwright.sync_api import sync_playwright
from browser import connect, BASE

URL = f"{BASE}/competitions/3/submissions/code/"
TERMINAL = re.compile(r"\b(completed|failed|errored|cancell?ed|invalid|rejected)\b", re.I)


def row_for(body, jid):
    """Return the status word + score seen in the block of text for id-<jid>."""
    m = re.search(r"id-" + str(jid), body)
    if not m:
        return None, None, ""
    # the status + score appear BEFORE the id in each row block; take the ~260 chars before it
    block = body[max(0, m.start() - 320):m.start() + 40]
    status = "?"
    for w in ["Uploading", "Pending", "Queued", "Starting", "Running", "Processing",
              "Scoring", "Building", "Completed", "Failed", "Errored", "Canceled",
              "Cancelled", "Invalid", "Rejected"]:
        if re.search(r"\b" + w + r"\b", block, re.I):
            status = w
    sc = re.search(r"\b(0\.\d{3,4})\b", block)
    return status, (sc.group(1) if sc else None), block


def snapshot(jid):
    with sync_playwright() as pw:
        b, ctx = connect(pw)
        p = ctx.new_page()
        try:
            p.goto(URL, wait_until="domcontentloaded", timeout=45000)
            p.wait_for_timeout(2000)
            body = p.inner_text("body")
        finally:
            try: p.close()
            except Exception: pass
    return row_for(body, jid)


def main(jid, timeout_s=2400, interval=40):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        el = int(time.time() - t0)
        try:
            status, score, block = snapshot(jid)
        except Exception as e:
            print(f"[{el}s] transient error, retrying: {str(e)[:80]}", flush=True)
            time.sleep(interval); continue
        print(f"[{el}s] id-{jid} status={status} score={score}", flush=True)
        if status and TERMINAL.search(status):
            print(f"JOB_TERMINAL id-{jid} status={status} score={score}", flush=True)
            print("---- row block ----", flush=True)
            print(block.replace("\t", " ").strip()[:600], flush=True)
            return 0
        time.sleep(interval)
    print(f"POLL_TIMEOUT after {timeout_s}s (job still running server-side)", flush=True)
    return 2


if __name__ == "__main__":
    jid = sys.argv[1]
    to = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    iv = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    sys.exit(main(jid, to, iv))
