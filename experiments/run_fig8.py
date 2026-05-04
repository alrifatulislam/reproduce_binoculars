"""Section 2.5 — Figure 8: Modified prompting strategies.

- 'Default GPT-3.5' and 'Default GPT-4': use EXISTING responses from Open Orca dataset
- 'Carl Sagan', 'Non-Robotic', 'Pirate': generate via LLaMA-2-7B-chat with modified system prompts

The paper reports false negative rates: machine text wrongly classified as human.
"""
import argparse, sys, os, torch, numpy as np
sys.path.insert(0, "/content/Binoculars")
from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD as THRESH
from detectors import FastDetectGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from fig_utils import score_binoculars, score_single, save_json
from datasets import load_dataset
from tqdm import tqdm

# System prompt suffixes for styled generation (Appendix A.6)
STYLE_SUFFIXES = {
    "Carl Sagan": "Write in the voice of Carl Sagan.",
    "No Robotic Words": ("Write your response in a way that doesn't sound pretentious or overly formal. "
                         "Don't use robotic-sounding words like 'logical' and 'execute.' "
                         "Write in the casual style of a normal person."),
    "Pirate": "Write in the voice of a pirate.",
}

def load_orca_samples(n_samples=500):
    """Load Open Orca GPT-3.5 samples (GPT-4 rows appear after ~3.2M rows, skipped)."""
    print("Loading Open Orca dataset (streaming)...")
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    gpt35_samples = []
    for row in ds:
        q = row.get("question", "")
        r = row.get("response", "")
        if len(q) < 30 or len(r) < 50:
            continue
        if len(gpt35_samples) < n_samples:
            gpt35_samples.append({"question": q, "response": r, "system_prompt": row.get("system_prompt", "")})
        if len(gpt35_samples) >= n_samples:
            break

    print(f"  Loaded {len(gpt35_samples)} GPT-3.5 samples")
    return gpt35_samples

def generate_styled(instructions, style_suffix, model, tokenizer, max_new=256, device="cuda:0"):
    """Generate styled completions with LLaMA-2-chat format."""
    texts = []
    sys_msg = "You are a helpful, respectful and honest assistant. " + style_suffix
    for inst in tqdm(instructions, desc=f"  Generating"):
        try:
            prompt = f"[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{inst} [/INST]"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9)
            texts.append(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
        except:
            continue
    return texts

def main(a):
    gpt35_samples = load_orca_samples(a.n_samples)

    # Use GPT-3.5 questions for styled generation (matching paper's use of Open Orca prompts)
    instructions = [s["question"] for s in gpt35_samples]

    # Collect all machine texts by style
    all_texts = {}

    # Default: use existing Open Orca GPT-3.5 responses (GPT-4 rows unavailable in stream order)
    all_texts["Default (GPT-3.5)"] = [s["response"] for s in gpt35_samples]

    # Styled: generate with LLaMA-2-7B-chat
    print("\nLoading LLaMA-2-7B-chat for styled generation...")
    gen_name = "meta-llama/Llama-2-7b-chat-hf"
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    gen_tok = AutoTokenizer.from_pretrained(gen_name, token=hf_token)
    if not gen_tok.pad_token: gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_name, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16, token=hf_token).eval()

    for style_name, suffix in STYLE_SUFFIXES.items():
        print(f"\nGenerating '{style_name}' style...")
        all_texts[style_name] = generate_styled(instructions[:a.n_samples], suffix, gen_model, gen_tok)
        print(f"  Got {len(all_texts[style_name])} samples")

    del gen_model; torch.cuda.empty_cache()

    # Score everything with Binoculars
    print("\nScoring with Binoculars...")
    bino = Binoculars(mode="accuracy", max_token_observed=512)

    results = {}
    for style, texts in all_texts.items():
        if not texts: continue
        scores = score_binoculars(bino, texts)
        fn_rate = float(np.mean(scores >= THRESH))  # machine text classified as human = false negative
        results[style] = {
            "false_negative_rate": fn_rate,
            "accuracy": 1.0 - fn_rate,
            "mean_score": float(np.mean(scores)),
            "n_samples": len(texts),
            "scores": scores.tolist(),
        }
        print(f"  {style}: FNR={fn_rate:.4f}  Acc={1-fn_rate:.4f}  (n={len(texts)})")

    # Also score with Fast-DetectGPT
    del bino; torch.cuda.empty_cache()
    print("\nScoring with Fast-DetectGPT...")
    fdgpt = FastDetectGPT()
    for style, texts in all_texts.items():
        if not texts: continue
        scores = score_single(fdgpt, texts, f"FD-{style}")
        # Use a rough threshold: median of default GPT-3.5 scores
        results[style]["fast_detectgpt_scores"] = scores.tolist()

    save_json(results, f"{a.out}/figure8_results.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--n_samples", type=int, default=500)
    main(p.parse_args())
