"""Shared dataset loaders, scoring helpers, and metric functions."""
import os, glob, json, numpy as np, torch
from tqdm import tqdm
from sklearn import metrics as sk
from datasets import Dataset

# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_ghostbuster(domain, path="/content/ghostbuster_repo"):
    """Load Ghostbuster txt files (ChatGPT only). domain: 'essay' | 'wp' | 'reuter'"""
    def _read(d):
        # Use direct .txt files if they exist (essay/wp are flat).
        # Fall back to one-level-deep glob for reuter (author subdirectories).
        # Never recurse deeper to avoid picking up log-probability sidecar files.
        direct = [f for f in sorted(glob.glob(os.path.join(d, "*.txt")))
                  if os.path.isfile(f)]
        fps = direct if direct else sorted(glob.glob(os.path.join(d, "*", "*.txt")))
        texts = []
        for fp in fps:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read().strip()
                if len(t) > 50: texts.append(t)
        return texts
    h = _read(os.path.join(path, "data", domain, "human"))
    m = _read(os.path.join(path, "data", domain, "gpt"))
    print(f"  {domain}: {len(h)} human, {len(m)} machine")
    return h, m

def load_jsonl(jsonl_path, human_key, machine_key):
    """Load from Binoculars-style JSONL."""
    ds = Dataset.from_json(jsonl_path)
    h = [x for x in ds[human_key] if x and len(x.strip()) > 50]
    m = [x for x in ds[machine_key] if x and len(x.strip()) > 50]
    print(f"  {jsonl_path}: {len(h)} human, {len(m)} machine")
    return h, m

def load_essayforum():
    """Load EssayForum ESL dataset from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("nid989/EssayFroum-Dataset", split="train")
    cols = ds.column_names
    orig_col = next((c for c in ["Cleaned Essay", "Cleaned_Essay", "original_text", "essay", "text"] if c in cols), None)
    corr_col = next((c for c in ["Correct Grammar", "Correct_Grammar", "corrected_text", "corrected"] if c in cols), None)
    if not orig_col or not corr_col:
        print(f"  WARNING: unexpected columns {cols}; returning empty")
        return [], []
    original = [t.strip() for t in ds[orig_col] if t and len(t.strip()) > 50]
    corrected = [t.strip() for t in ds[corr_col] if t and len(t.strip()) > 50]
    print(f"  EssayForum: {len(original)} original, {len(corrected)} corrected")
    return original, corrected

# ── Scoring helpers ──────────────────────────────────────────────────────────

def score_binoculars(bino, texts, batch_size=16):
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Binoculars"):
        b = texts[i:i+batch_size]
        s = bino.compute_score(b)
        scores.extend(s if isinstance(s, list) else [s])
    return np.array(scores)

def score_single(detector, texts, label="Detector"):
    scores = []
    for i in tqdm(range(len(texts)), desc=f"  {label}"):
        try:
            s = detector.compute_score(texts[i])
            scores.append(s if isinstance(s, (float, int)) else s[0])
        except: scores.append(0.0)
    return np.array(scores)

# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_roc(human_scores, machine_scores, higher_human=True, fpr_target=0.0001):
    """Returns dict with tpr_at_fpr, auc, fpr, tpr arrays."""
    labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(machine_scores))])
    if higher_human:
        scores = np.concatenate([-human_scores, -machine_scores])
    else:
        scores = np.concatenate([human_scores, machine_scores])
    fpr, tpr, _ = sk.roc_curve(labels, scores, pos_label=1)
    return {"tpr_at_fpr": float(np.interp(fpr_target, fpr, tpr)),
            "auc": float(sk.auc(fpr, tpr)),
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}

def compute_tpr_at_multiple_fprs(human_scores, machine_scores, higher_human=True):
    """Returns TPR at 0.01%, 0.1%, 1%, 5% FPR + AUC (for Table 4)."""
    labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(machine_scores))])
    scores = np.concatenate([-human_scores, -machine_scores]) if higher_human else np.concatenate([human_scores, machine_scores])
    fpr, tpr, _ = sk.roc_curve(labels, scores, pos_label=1)
    return {"auc": float(sk.auc(fpr, tpr)),
            "tpr@0.01%": float(np.interp(0.0001, fpr, tpr)),
            "tpr@0.1%":  float(np.interp(0.001, fpr, tpr)),
            "tpr@1%":    float(np.interp(0.01, fpr, tpr)),
            "tpr@5%":    float(np.interp(0.05, fpr, tpr))}

# ── I/O ──────────────────────────────────────────────────────────────────────

def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"  Saved: {path}")

def load_json(path):
    with open(path) as f: return json.load(f)
