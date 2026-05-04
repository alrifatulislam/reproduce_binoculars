"""Section 2.1 — Figure 1: TPR @ 0.01% FPR on Ghostbuster ChatGPT datasets.
Compares: Binoculars, Ghostbuster (OOD), DetectGPT, Fast-DetectGPT, DNA-GPT
"""
import argparse, sys, torch
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from detectors import DetectGPT, FastDetectGPT, DNAGPT, GhostbusterOOD
from fig_utils import load_ghostbuster, score_binoculars, score_single, compute_roc, save_json
import numpy as np

DOMAINS = {"News": "reuter", "Creative Writing": "wp", "Student Essay": "essay"}

def main(a):
    # Load all domains upfront
    all_domain_data = {}
    for name, folder in DOMAINS.items():
        h, m = load_ghostbuster(folder, a.ghost_path)
        if a.max_n: h, m = h[:a.max_n], m[:a.max_n]
        all_domain_data[name] = (h, m)

    results = {name: {} for name in DOMAINS}

    # --- Binoculars (load, score all domains, delete) ---
    print("\nInitializing Binoculars...")
    bino = Binoculars(mode="accuracy", max_token_observed=512)
    for name in DOMAINS:
        h, m = all_domain_data[name]
        hs = score_binoculars(bino, h)
        ms = score_binoculars(bino, m)
        r = compute_roc(hs, ms, higher_human=True)
        results[name]["Binoculars (Ours)"] = r
        print(f"  {name}: TPR@0.01%FPR={r['tpr_at_fpr']:.4f}  AUC={r['auc']:.4f}")
    del bino; torch.cuda.empty_cache()

    # --- Ghostbuster OOD (CPU-based, safe to run any time) ---
    print("\nInitializing Ghostbuster (OOD)...")
    ghost = GhostbusterOOD()
    for name in DOMAINS:
        h, m = all_domain_data[name]
        ghost.train_ood(all_domain_data, test_domain=name)
        hs_g = np.array(ghost.compute_score(h) if len(h) > 1 else [ghost.compute_score(h[0])])
        ms_g = np.array(ghost.compute_score(m) if len(m) > 1 else [ghost.compute_score(m[0])])
        r = compute_roc(hs_g, ms_g, higher_human=False)
        results[name]["Ghostbuster (OOD)"] = r
        print(f"  {name}: TPR@0.01%FPR={r['tpr_at_fpr']:.4f}  AUC={r['auc']:.4f}")
    del ghost; torch.cuda.empty_cache()

    # --- Fast-DetectGPT (load, score all domains, delete) ---
    print("\nInitializing Fast-DetectGPT...")
    fdgpt = FastDetectGPT()
    for name in DOMAINS:
        h, m = all_domain_data[name]
        hs = score_single(fdgpt, h, "Fast-DetectGPT")
        ms = score_single(fdgpt, m, "Fast-DetectGPT")
        r = compute_roc(hs, ms, higher_human=False)
        results[name]["Fast-DetectGPT"] = r
        print(f"  {name}: TPR@0.01%FPR={r['tpr_at_fpr']:.4f}  AUC={r['auc']:.4f}")
    del fdgpt; torch.cuda.empty_cache()

    # --- DetectGPT (load, score all domains, delete) ---
    if not a.skip_detectgpt:
        print("\nInitializing DetectGPT (slow)...")
        dgpt = DetectGPT(n_perturbations=a.n_perturb)
        for name in DOMAINS:
            h, m = all_domain_data[name]
            hs = score_single(dgpt, h, "DetectGPT")
            ms = score_single(dgpt, m, "DetectGPT")
            r = compute_roc(hs, ms, higher_human=False)
            results[name]["DetectGPT"] = r
            print(f"  {name}: TPR@0.01%FPR={r['tpr_at_fpr']:.4f}  AUC={r['auc']:.4f}")
        del dgpt; torch.cuda.empty_cache()

    # --- DNA-GPT (load, score all domains, delete) ---
    if not a.skip_dnagpt:
        print("\nInitializing DNA-GPT (slow)...")
        dna = DNAGPT()
        for name in DOMAINS:
            h, m = all_domain_data[name]
            hs = score_single(dna, h, "DNA-GPT")
            ms = score_single(dna, m, "DNA-GPT")
            r = compute_roc(hs, ms, higher_human=False)
            results[name]["DNA-GPT"] = r
            print(f"  {name}: TPR@0.01%FPR={r['tpr_at_fpr']:.4f}  AUC={r['auc']:.4f}")
        del dna; torch.cuda.empty_cache()

    save_json(results, f"{a.out}/figure1_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ghost_path", default="/content/ghostbuster_repo")
    p.add_argument("--out", default="results")
    p.add_argument("--skip_detectgpt", action="store_true")
    p.add_argument("--skip_dnagpt", action="store_true")
    p.add_argument("--n_perturb", type=int, default=10)
    p.add_argument("--max_n", type=int, default=None)
    main(p.parse_args())
