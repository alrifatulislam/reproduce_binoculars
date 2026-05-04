"""
DetectGPT (Mitchell et al., 2023): Zero-Shot Machine-Generated Text Detection
using Probability Curvature via T5 mask-filling perturbations.
"""
import re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

class DetectGPT:
    def __init__(self, scoring_model_name="tiiuae/falcon-7b-instruct",
                 mask_model_name="t5-small", n_perturbations=10,
                 mask_pct=0.15, max_length=512, device="cuda:0"):
        self.n_perturbations, self.mask_pct, self.max_length, self.device = n_perturbations, mask_pct, max_length, device
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        if not self.scoring_tokenizer.pad_token:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_model_name, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.scoring_model.eval()
        self.mask_tokenizer = T5Tokenizer.from_pretrained(mask_model_name)
        self.mask_model = T5ForConditionalGeneration.from_pretrained(mask_model_name).to(device)
        self.mask_model.eval()

    @torch.inference_mode()
    def _get_ll(self, text):
        enc = self.scoring_tokenizer(text, return_tensors="pt", truncation=True,
                                     max_length=self.max_length, return_token_type_ids=False).to(self.device)
        logits = self.scoring_model(**enc).logits
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            logits[..., :-1, :].contiguous().transpose(1, 2), enc.input_ids[..., 1:].contiguous())
        mask = enc.attention_mask[..., 1:].contiguous()
        return -(loss * mask).sum() / mask.sum()

    def _perturb(self, text):
        words = text.split()
        if len(words) < 5: return text
        n = max(1, int(len(words) * self.mask_pct))
        idxs = sorted(np.random.choice(len(words), min(n, len(words)), replace=False))
        masked = list(words)
        for j, idx in enumerate(idxs): masked[idx] = f"<extra_id_{j}>"
        inp = self.mask_tokenizer(" ".join(masked), return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.inference_mode():
            out = self.mask_model.generate(**inp, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.95)
        fills = [f.strip() for f in re.split(r"<extra_id_\d+>", self.mask_tokenizer.decode(out[0], skip_special_tokens=False)) if f.strip()]
        result = list(words)
        for j, idx in enumerate(idxs):
            if j < len(fills): result[idx] = fills[j]
        return " ".join(result)

    def compute_score(self, input_text):
        if isinstance(input_text, str): input_text = [input_text]
        scores = []
        for text in input_text:
            orig = self._get_ll(text).item()
            plls = []
            for _ in range(self.n_perturbations):
                try: plls.append(self._get_ll(self._perturb(text)).item())
                except: continue
            if len(plls) < 2: scores.append(0.0); continue
            scores.append((orig - np.mean(plls)) / (np.std(plls) + 1e-10))
        return scores if len(scores) > 1 else scores[0]
