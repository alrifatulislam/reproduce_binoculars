"""
Fast-DetectGPT (Bao et al., 2023): Efficient Zero-Shot Detection via
Conditional Probability Curvature — no perturbations needed.
"""
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FastDetectGPT:
    def __init__(self, scoring_model_name="gpt2-xl",
                 reference_model_name="gpt2-xl", max_length=512, device="cuda:0"):
        self.max_length, self.device = max_length, device
        self.tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        if not self.tokenizer.pad_token: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_model_name, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.scoring_model.eval()
        if reference_model_name != scoring_model_name:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                reference_model_name, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.reference_model.eval()
        else:
            self.reference_model = self.scoring_model

    @torch.inference_mode()
    def compute_score(self, input_text):
        if isinstance(input_text, str): input_text = [input_text]
        scores = []
        for text in input_text:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.max_length, return_token_type_ids=False).to(self.device)
            s_lp = torch.nn.functional.log_softmax(self.scoring_model(**enc).logits, dim=-1)
            r_p = torch.nn.functional.softmax(self.reference_model(**enc).logits, dim=-1)
            labels, mask = enc.input_ids[..., 1:], enc.attention_mask[..., 1:].float()
            s_lp_s = s_lp[..., :-1, :]
            actual = s_lp_s.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            r_p_s = r_p[..., :-1, :]
            e_lp = (r_p_s * s_lp_s).sum(-1)
            var = (r_p_s * s_lp_s ** 2).sum(-1) - e_lp ** 2
            curv = (actual - e_lp) / torch.sqrt(var.clamp(min=1e-10))
            scores.append(((curv * mask).sum(-1) / mask.sum(-1)).item())
        return scores if len(scores) > 1 else scores[0]
