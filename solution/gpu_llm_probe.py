"""QLoRA Qwen2.5-7B classifier PROBE (Fable Round-3 spec) — fold-0 only, subsampled train, to
decide continue/drop BEFORE the expensive full OOF. Metric log loss; last-token pooling (decoder).

Gate (checked after): continue iff leg AUROC >= 0.640 AND val logloss < 0.63 AND corr(p_7b,p_base) < 0.90.
Saves /workspace/probe_7b_fold0.parquet (response_id, p_7b, y) for the correlation-vs-base check.
Usage: python gpu_llm_probe.py <train_subsample> <epochs>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
import dl_common as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("LLM_DIR", "/workspace/models/Qwen2.5-7B")
TDIR = os.path.join(ROOT, "data", "train_transcripts")
MAX_LEN = 1536


def main():
    subset_train = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    val_subset = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # >0 = subsample val (smoke)
    t0 = time.time()
    f = pd.read_csv(os.path.join(ROOT, "data", "train_features.csv"))
    lab = pd.read_csv(os.path.join(ROOT, "data", "train_labels.csv")).set_index("response_id")
    f["y"] = lab.loc[f.response_id, "is_correct"].values
    y = f.y.to_numpy(int); groups = f.learning_objective_id.astype(str).to_numpy()
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(f, y, groups))  # SAME fold-0 as OOF
    rng = np.random.RandomState(42); tr = rng.permutation(tr)[:subset_train]
    if val_subset > 0:
        va = np.random.RandomState(1).permutation(va)[:val_subset]   # smoke: shrink val
    print(f"probe: train={len(tr)} (subsample of fold-0 train) val={len(va)}", flush=True)

    D.HISTORY_WORDS, D.RELEVANT_WORDS, D.RECENT_WORDS = 0, 300, 550   # focused rep sized for ~1536 tok
    texts = D.build_texts(f, TDIR, n_words=850, centered=True, proxy_tags=True)
    tr_texts = [texts[i] for i in tr]; tr_y = y[tr]
    va_texts = [texts[i] for i in va]; va_y = y[va].astype(int)
    va_ids = f.response_id.astype(str).to_numpy()[va]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LEFT padding: Qwen2 + flash-attn hard-errors on right-padding in eval; Qwen2ForSequenceClassification's
    # last-non-pad-token pooling is correct for left-padding too (argmax(pad)-1 % L -> last real position).
    tok.padding_side = "left"; tok.truncation_side = "left"     # keep the assessment-relevant tail

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, quantization_config=bnb,
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      modules_to_save=["score"])
    m = get_peft_model(m, lora)
    m.print_trainable_parameters()

    class DS(Dataset):
        def __init__(self, texts, ys): self.texts = texts; self.ys = ys
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            enc = tok(self.texts[i], truncation=True, max_length=MAX_LEN)
            enc["labels"] = int(self.ys[i]); return enc

    def compute_metrics(ep):
        logits, labels = ep
        p = torch.softmax(torch.tensor(logits).float(), -1)[:, 1].numpy()
        return {"auc": float(roc_auc_score(labels, p))}

    args = TrainingArguments(
        output_dir="/workspace/llm_out", per_device_train_batch_size=2, per_device_eval_batch_size=8,
        gradient_accumulation_steps=8, learning_rate=1e-4, num_train_epochs=epochs, lr_scheduler_type="cosine",
        warmup_ratio=0.1, weight_decay=0.0, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="paged_adamw_32bit",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="auc", greater_is_better=True, save_total_limit=1,
        logging_steps=25, report_to=[], dataloader_num_workers=2)
    trainer = Trainer(model=m, args=args, train_dataset=DS(tr_texts, tr_y),
                      eval_dataset=DS(va_texts, va_y), data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()

    pred = trainer.predict(DS(va_texts, va_y))
    p = torch.softmax(torch.tensor(pred.predictions).float(), -1)[:, 1].numpy()
    auc = float(roc_auc_score(va_y, p)); ll = float(log_loss(va_y, np.clip(p, 1e-6, 1 - 1e-6)))
    pd.DataFrame({"response_id": va_ids, "p_7b": p, "y": va_y}).to_parquet("/workspace/probe_7b_fold0.parquet")
    print(f"PROBE_RESULT leg_auc={auc:.4f} val_logloss={ll:.5f} n_val={len(va_y)} time={int(time.time()-t0)}s", flush=True)
    print(f"GATE auc>=0.640:{'PASS' if auc >= 0.640 else 'FAIL'} ll<0.63:{'PASS' if ll < 0.63 else 'FAIL'} "
          f"(corr-vs-base checked locally)", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
