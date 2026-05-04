"""Section 2.2 — Figure 3: ROC curves on PubMed / CC-News / CNN (LLaMA-2-13B)."""
import argparse, sys, torch
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from detectors import DetectGPT, FastDetectGPT, DNAGPT, GhostbusterOOD
from fig_utils import load_ghostbuster, load_jsonl, score_binoculars, score_single, compute_roc, save_json
import numpy as np

DS = {"PubMed":  ("pubmed/pubmed-llama2_13.jsonl",    "article", "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"),
      "CC News": ("cc_news/cc_news-llama2_13.jsonl",  "text",    "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt"),
      "CNN":     ("cnn/cnn-llama2_13.jsonl",           "article", "meta-llama-Llama-2-13b-hf_generated_text_wo_prompt")}

def main(a):
    # Load all datasets upfront
    all_data = {}
    for name, (path, hk, mk) in DS.items():
        h, m = load_jsonl(f"{a.ds_base}/{path}", hk, mk)
        if a.max_n: h, m = h[:a.max_n], m[:a.max_n]
        all_data[name] = (h, m)

    # Load Ghostbuster data (all domains) for OOD training
    print("\nLoading Ghostbuster data for OOD training...")
    ghost_h_all, ghost_m_all = [], []
    for folder in ["reuter", "wp", "essay"]:
        h_g, m_g = load_ghostbuster(folder, a.ghost_path)
        ghost_h_all.extend(h_g)
        ghost_m_all.extend(m_g)
    print(f"  Ghostbuster OOD train: {len(ghost_h_all)} human, {len(ghost_m_all)} machine")

    results = {name: {} for name in DS}

    # --- Binoculars (load, score all datasets, delete) ---
    print("\nInitializing Binoculars...")
    bino = Binoculars(mode="accuracy", max_token_observed=512)
    for name, (h, m) in all_data.items():
        print(f"  Binoculars on {name}...")
        hs = score_binoculars(bino, h)
        ms = score_binoculars(bino, m)
        r = compute_roc(hs, ms, higher_human=True)
        results[name]["Binoculars (Ours)"] = r
        print(f"    AUC={r['auc']:.4f}  TPR@0.01%FPR={r['tpr_at_fpr']:.4f}")
    del bino; torch.cuda.empty_cache()

    # --- Ghostbuster OOD (train on Ghostbuster domains, test on PubMed/CC-News/CNN) ---
    print("\nInitializing Ghostbuster (OOD)...")
    ghost = GhostbusterOOD()
    ghost.train(ghost_h_all, ghost_m_all)
    for name, (h, m) in all_data.items():
        print(f"  Ghostbuster on {name}...")
        hs_g = np.array(ghost.compute_score(h))
        ms_g = np.array(ghost.compute_score(m))
        r = compute_roc(hs_g, ms_g, higher_human=False)
        results[name]["Ghostbuster (OOD)"] = r
        print(f"    AUC={r['auc']:.4f}  TPR@0.01%FPR={r['tpr_at_fpr']:.4f}")
    del ghost; torch.cuda.empty_cache()

    # --- Fast-DetectGPT (load, score all datasets, delete) ---
    print("\nInitializing Fast-DetectGPT...")
    fdgpt = FastDetectGPT()
    for name, (h, m) in all_data.items():
        print(f"  Fast-DetectGPT on {name}...")
        hs = score_single(fdgpt, h, "Fast-DetectGPT")
        ms = score_single(fdgpt, m, "Fast-DetectGPT")
        r = compute_roc(hs, ms, higher_human=False)
        results[name]["Fast-DetectGPT"] = r
        print(f"    AUC={r['auc']:.4f}  TPR@0.01%FPR={r['tpr_at_fpr']:.4f}")
    del fdgpt; torch.cuda.empty_cache()

    # --- DetectGPT (load, score all datasets, delete) ---
    if not a.skip_detectgpt:
        print("\nInitializing DetectGPT (slow)...")
        dgpt = DetectGPT(n_perturbations=a.n_perturb)
        for name, (h, m) in all_data.items():
            print(f"  DetectGPT on {name}...")
            hs = score_single(dgpt, h, "DetectGPT")
            ms = score_single(dgpt, m, "DetectGPT")
            r = compute_roc(hs, ms, higher_human=False)
            results[name]["DetectGPT"] = r
            print(f"    AUC={r['auc']:.4f}  TPR@0.01%FPR={r['tpr_at_fpr']:.4f}")
        del dgpt; torch.cuda.empty_cache()

    # --- DNA-GPT (load, score all datasets, delete) ---
    if not a.skip_dnagpt:
        print("\nInitializing DNA-GPT (slow)...")
        dna = DNAGPT()
        for name, (h, m) in all_data.items():
            print(f"  DNA-GPT on {name}...")
            hs = score_single(dna, h, "DNA-GPT")
            ms = score_single(dna, m, "DNA-GPT")
            r = compute_roc(hs, ms, higher_human=False)
            results[name]["DNA-GPT"] = r
            print(f"    AUC={r['auc']:.4f}  TPR@0.01%FPR={r['tpr_at_fpr']:.4f}")
        del dna; torch.cuda.empty_cache()

    save_json(results, f"{a.out}/figure3_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ds_base", default="/content/Binoculars/datasets/core")
    p.add_argument("--out", default="results")
    p.add_argument("--skip_detectgpt", action="store_true")
    p.add_argument("--skip_dnagpt", action="store_true")
    p.add_argument("--ghost_path", default="/content/ghostbuster_repo")
    p.add_argument("--n_perturb", type=int, default=10)
    p.add_argument("--max_n", type=int, default=None)
    main(p.parse_args())
