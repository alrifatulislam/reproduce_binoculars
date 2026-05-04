"""Section 2.4 — Figure 7: Binoculars score distribution on ESL essays."""
import argparse, sys
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from fig_utils import load_essayforum, score_binoculars, save_json

def main(a):
    print("Loading EssayForum dataset...")
    original, corrected = load_essayforum()
    if a.max_n: original, corrected = original[:a.max_n], corrected[:a.max_n]
    bino = Binoculars(mode="accuracy", max_token_observed=512)
    print("Scoring original essays...")
    orig_scores = score_binoculars(bino, original).tolist()
    print("Scoring corrected essays...")
    corr_scores = score_binoculars(bino, corrected).tolist()
    save_json({"original": orig_scores, "corrected": corr_scores},
              f"{a.out}/figure7_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--max_n", type=int, default=None)
    main(p.parse_args())
