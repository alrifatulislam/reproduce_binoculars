"""Extension 1 (Section 3.1): Evaluate Binoculars on newer open-source LLMs.

Uses Open Orca prompts. Existing GPT-3.5/GPT-4 responses serve as baseline.
Generation IS needed for newer models (Mistral-7B-Instruct-v0.2, Qwen2-7B-Instruct).
"""
import argparse, sys, os, torch, numpy as np
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD as THRESH
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fig_utils import score_binoculars, save_json
from tqdm import tqdm

# Models to generate from (Extension 1 — Section 3.1 of proposal)
GEN_MODELS = {
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
}

def load_orca_prompts_and_responses(n_samples=500):
    """Load Open Orca: prompts + existing GPT-3.5 responses."""
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    gpt35, prompts = [], []
    for row in ds:
        q, r = row.get("question", ""), row.get("response", "")
        if len(q) < 30 or len(r) < 50: continue
        if len(gpt35) < n_samples:
            gpt35.append(r)
        if len(prompts) < n_samples:
            prompts.append(q)
        if len(gpt35) >= n_samples and len(prompts) >= n_samples:
            break
    print(f"  Loaded {len(prompts)} prompts, {len(gpt35)} GPT-3.5 responses")
    return prompts, gpt35

def generate_from_model(model_name, prompts, max_new=256, device="cuda:0"):
    """Generate completions from a model given prompts."""
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, token=hf_token)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True, token=hf_token).eval()
    texts = []
    for p in tqdm(prompts, desc=f"  Gen ({model_name.split('/')[-1]})"):
        try:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9)
            texts.append(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip())
        except: continue
    del model; torch.cuda.empty_cache()
    return texts

def main(a):
    prompts, gpt35_resp = load_orca_prompts_and_responses(a.n_samples)

    # All texts to score: existing + newly generated
    all_texts = {
        "GPT-3.5": gpt35_resp,
    }

    # Generate from newer models
    for label, model_id in GEN_MODELS.items():
        print(f"\n=== Generating from {label} ===")
        all_texts[label] = generate_from_model(model_id, prompts[:a.n_samples])
        print(f"  Generated {len(all_texts[label])} texts")

    # Score everything with Binoculars
    print("\nInitializing Binoculars...")
    bino = Binoculars(mode="accuracy", max_token_observed=512)
    results = {}
    for label, texts in all_texts.items():
        if not texts: continue
        print(f"\nScoring {label}...")
        scores = score_binoculars(bino, texts)
        fn_rate = float(np.mean(scores >= THRESH))
        results[label] = {
            "accuracy": 1.0 - fn_rate,
            "false_negative_rate": fn_rate,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "n_samples": len(texts),
            "scores": scores.tolist(),
        }
        print(f"  Acc={1-fn_rate:.4f}  FNR={fn_rate:.4f}  Mean={np.mean(scores):.4f}")

    save_json(results, f"{a.out}/ext1_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--n_samples", type=int, default=500)
    main(p.parse_args())
