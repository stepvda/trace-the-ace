"""Autonomous finalize chain (run after CV starts):
  1. wait for cv_fast RESULT line
  2. parse tuned blend weight + best clip level
  3. fit final pipeline on ALL data with those overrides -> assets/artifacts.pkl
  4. build submission.zip
  5. run the local end-to-end runtime test
Prints clear step markers so progress is visible.
"""
import os, sys, json, time, subprocess, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CVLOG = "/tmp/cvfast.log"
PY = os.path.join(ROOT, ".venv", "bin", "python")
CFG_PATH = os.path.join(ROOT, "solution", "cache", "final_config.json")


def wait_for_result(timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            txt = open(CVLOG).read()
        except FileNotFoundError:
            txt = ""
        if "RESULT " in txt:
            for line in txt.splitlines():
                if line.startswith("RESULT "):
                    return json.loads(line[len("RESULT "):])
        if re.search(r"Traceback|Error|Killed", txt):
            print("CV appears to have errored:\n", txt[-1000:]); sys.exit(1)
        time.sleep(5)
    print("timed out waiting for CV RESULT"); sys.exit(1)


def main():
    print("[finalize] waiting for CV result ...", flush=True)
    res = wait_for_result()
    print("[finalize] CV result:", json.dumps(res), flush=True)

    best_w = res.get("best_w_lr", 0.55)
    # choose clip level minimizing OOF log loss from the clip scan
    clip_keys = {float(k.split("_")[1]): v for k, v in res.items() if k.startswith("clip_")}
    best_clip = min(clip_keys, key=clip_keys.get) if clip_keys else 0.005
    cfg = {"blend_w_lr": float(best_w), "clip": float(best_clip)}
    os.makedirs(os.path.dirname(CFG_PATH), exist_ok=True)
    json.dump(cfg, open(CFG_PATH, "w"))
    print(f"[finalize] tuned config: {cfg}  (best OOF blend logloss={res.get('best_blend_ll'):.5f})", flush=True)

    def run(step, args):
        print(f"\n[finalize] === {step} ===", flush=True)
        r = subprocess.run(args, cwd=ROOT)
        if r.returncode != 0:
            print(f"[finalize] {step} FAILED rc={r.returncode}"); sys.exit(r.returncode)

    run("train_final", [PY, "solution/train_final.py", CFG_PATH])
    run("package", [PY, "automation/package.py"])
    run("local_test", [PY, "automation/test_submission_local.py", "800"])
    print("\n[finalize] ALL DONE — submission.zip built and locally validated.", flush=True)


if __name__ == "__main__":
    main()
