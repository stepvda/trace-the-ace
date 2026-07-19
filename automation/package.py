"""Assemble a submission zip. Two modes:

  python package.py classical   -> submission_classical.zip (~49MB, reliable)
      main.py (classical) + features.py + model.py + assets/artifacts.pkl

  python package.py container   -> submission_container.zip (~692MB, ambitious)
      main_container.py (as main.py) + features.py + model.py + dl_common.py +
      dl_train.py + assets/ (artifacts, train_texts, classical_oof, base_model)

  python package.py             -> defaults to 'container' (back-compat) as submission.zip

main.py must be at the archive ROOT (no wrapping folder), per the rules.
"""
import os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")


def build(mode):
    if mode == "classical":
        out = os.path.join(ROOT, "submission_classical.zip")
        root_files = [(os.path.join(SUB, "main.py"), "main.py"),
                      (os.path.join(ROOT, "solution", "features.py"), "features.py"),
                      (os.path.join(ROOT, "solution", "model.py"), "model.py")]
        asset_files = ["artifacts.pkl"]
        asset_dirs = []
    elif mode == "infer":
        # INFERENCE-ONLY container: bundled classical + pre-fine-tuned ModernBERT seeds.
        # No dl_train.py (no in-container training), no base_model, no train_texts.
        out = os.path.join(ROOT, "submission_infer.zip")
        root_files = [(os.path.join(SUB, "main_infer.py"), "main.py"),
                      (os.path.join(ROOT, "solution", "features.py"), "features.py"),
                      (os.path.join(ROOT, "solution", "model.py"), "model.py"),
                      (os.path.join(ROOT, "solution", "dl_common.py"), "dl_common.py")]
        asset_files = ["artifacts.pkl", "classical_oof.parquet", "blend_config.json"]
        asset_dirs = ["mbert_seeds"]
    else:
        out = os.path.join(ROOT, "submission_container.zip" if mode == "container" else "submission.zip")
        root_files = [(os.path.join(SUB, "main_container.py"), "main.py"),
                      (os.path.join(ROOT, "solution", "features.py"), "features.py"),
                      (os.path.join(ROOT, "solution", "model.py"), "model.py"),
                      (os.path.join(ROOT, "solution", "dl_common.py"), "dl_common.py"),
                      (os.path.join(ROOT, "solution", "dl_train.py"), "dl_train.py")]
        asset_files = ["artifacts.pkl", "train_texts.parquet", "classical_oof.parquet"]
        asset_dirs = ["base_model"]

    for src, _ in root_files:
        assert os.path.exists(src), f"missing {src}"
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in root_files:
            z.write(src, arc)
        for fn in asset_files:
            src = os.path.join(SUB, "assets", fn)
            if os.path.exists(src):
                z.write(src, os.path.join("assets", fn))
        for d in asset_dirs:
            base = os.path.join(SUB, "assets", d)
            for dp, _, files in os.walk(base):
                for fn in files:
                    fp = os.path.join(dp, fn)
                    # relpath is already relative to submission/assets and includes the
                    # dir name (e.g. "base_model/model.safetensors"); joining `d` again
                    # would double-nest to assets/base_model/base_model/... and the
                    # container would not find the model -> silent classical fallback.
                    arc = os.path.join("assets", os.path.relpath(fp, os.path.join(SUB, "assets")))
                    z.write(fp, arc)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "main.py" in names, "main.py not at root!"
    print(f"built {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    for n in names[:12]:
        print("  ", n)
    if len(names) > 12:
        print(f"   ... (+{len(names)-12} more)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "container")
