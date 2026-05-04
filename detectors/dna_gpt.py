"""
DNA-GPT (Yang et al., 2023): Divergent N-Gram Analysis for Training-Free Detection.
Uses re-generation divergence: if a model re-generates text from a truncated prefix,
machine text will have higher n-gram overlap with the original than human text.
"""
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

class DNAGPT:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct", n_regenerations=5,
                 truncate_ratio=0.5, ngram=4, max_length=512, device="cuda:0"):
        self.n_regenerations, self.truncate_ratio = n_regenerations, truncate_ratio
        self.ngram, self.max_length, self.device = ngram, max_length, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.eval()

    def _get_ngrams(self, tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

    def _ngram_overlap(self, tokens_a, tokens_b, n):
        ng_a, ng_b = self._get_ngrams(tokens_a, n), self._get_ngrams(tokens_b, n)
        shared = sum((ng_a & ng_b).values())
        total = max(sum(ng_a.values()), 1)
        return shared / total

    @torch.inference_mode()
    def compute_score(self, input_text):
        if isinstance(input_text, str): input_text = [input_text]
        scores = []
        for text in input_text:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
            cut = max(10, int(len(tokens) * self.truncate_ratio))
            prefix_ids = torch.tensor([tokens[:cut]]).to(self.device)
            suffix_tokens = tokens[cut:]
            overlaps = []
            for _ in range(self.n_regenerations):
                try:
                    out = self.model.generate(prefix_ids, max_new_tokens=len(suffix_tokens),
                                              do_sample=True, temperature=0.8, top_p=0.95)
                    regen = out[0][cut:].cpu().tolist()
                    overlaps.append(self._ngram_overlap(suffix_tokens, regen, self.ngram))
                except: continue
            scores.append(np.mean(overlaps) if overlaps else 0.0)
        return scores if len(scores) > 1 else scores[0]
