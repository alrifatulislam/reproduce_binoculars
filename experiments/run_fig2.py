"""Section 2.3 — Figure 2: TPR @ 0.01% FPR at 128 / 256 / 512 tokens.

Detectors: Binoculars, Fast-DetectGPT, Ghostbuster (in-distribution).
Ghostbuster: trained on 80% of each domain, tested on remaining 20%.
"""
import argparse, sys, torch, numpy as np
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from detectors import FastDetectGPT, GhostbusterOOD
from fig_utils import load_ghostbuster, score_binoculars, score_single, compute_roc, save_json

DOMAINS = {"News": "reuter", "Creative Writing": "wp", "Student Essay": "essay"}
TOKENS = [128, 256, 512]

def main(a):
    # Load all domain data upfront
    all_domain_data = {}
    for name, folder in DOMAINS.items():
        print(f"\nLoading {name}...")
        h, m = load_ghostbuster(folder, a.ghost_path)
        if a.max_n: h, m = h[:a.max_n], m[:a.max_n]
        all_domain_data[name] = (h, m)

    results = {name: {"Binoculars (Ours)": {}, "Fast-DetectGPT": {},
                      "Ghostbuster (ID)": {}}
               for name in DOMAINS}

    # --- Binoculars (load per token length, score all domains, delete) ---
    for nt in TOKENS:
        print(f"\n=== Binoculars {nt} tokens ===")
        bino = Binoculars(mode="accuracy", max_token_observed=nt)
        for name in DOMAINS:
            h, m = all_domain_data[name]
            hs, ms = score_binoculars(bino, h), score_binoculars(bino, m)
            results[name]["Binoculars (Ours)"][str(nt)] = compute_roc(hs, ms)["tpr_at_fpr"]
            print(f"  {name}: TPR@0.01%FPR={results[name]['Binoculars (Ours)'][str(nt)]:.4f}")
        del bino; torch.cuda.empty_cache()

    # --- Fast-DetectGPT (load per token length, score all domains, delete) ---
    for nt in TOKENS:
        print(f"\n=== Fast-DetectGPT {nt} tokens ===")
        fdgpt = FastDetectGPT(max_length=nt)
        for name in DOMAINS:
            h, m = all_domain_data[name]
            hs = score_single(fdgpt, h, "Fast-DetectGPT")
            ms = score_single(fdgpt, m, "Fast-DetectGPT")
            results[name]["Fast-DetectGPT"][str(nt)] = compute_roc(hs, ms, higher_human=False)["tpr_at_fpr"]
            print(f"  {name}: TPR@0.01%FPR={results[name]['Fast-DetectGPT'][str(nt)]:.4f}")
        del fdgpt; torch.cuda.empty_cache()

    # --- Ghostbuster in-distribution (load once, train per domain 80/20 split) ---
    print("\n=== Ghostbuster (in-distribution) ===")
    ghost = GhostbusterOOD()
    for name in DOMAINS:
        h, m = all_domain_data[name]
        n_h_tr, n_m_tr = int(len(h) * 0.8), int(len(m) * 0.8)
        h_train, h_test = h[:n_h_tr], h[n_h_tr:]
        m_train, m_test = m[:n_m_tr], m[n_m_tr:]
        print(f"  {name}: train {len(h_train)}h/{len(m_train)}m, test {len(h_test)}h/{len(m_test)}m")
        ghost.train(h_train, m_train)
        for nt in TOKENS:
            ghost.max_length = nt
            hs_g = np.array(ghost.compute_score(h_test))
            ms_g = np.array(ghost.compute_score(m_test))
            results[name]["Ghostbuster (ID)"][str(nt)] = compute_roc(hs_g, ms_g, higher_human=False)["tpr_at_fpr"]
            print(f"    {nt} tokens: TPR@0.01%FPR={results[name]['Ghostbuster (ID)'][str(nt)]:.4f}")
    del ghost; torch.cuda.empty_cache()

    save_json(results, f"{a.out}/figure2_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ghost_path", default="/content/ghostbuster_repo")
    p.add_argument("--out", default="results")
    p.add_argument("--max_n", type=int, default=None)
    main(p.parse_args())
