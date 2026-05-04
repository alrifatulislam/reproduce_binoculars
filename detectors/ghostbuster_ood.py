"""
Ghostbuster (OOD) — Verma et al., 2023
Open-source reproduction using local models instead of OpenAI API.

Ghostbuster works by:
1. Computing per-token log-probabilities under multiple weaker language models
2. Extracting features (mean, std, max, etc.) from these log-prob vectors
3. Training a logistic regression classifier on these features

The OOD (out-of-domain) variant trains on two domains and evaluates on the third.
Here we use open-source models (Falcon-7B, GPT-2) instead of GPT-3 ada/davinci.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class GhostbusterOOD:
    def __init__(self, device="cuda:0", max_length=512):
        self.device = device
        self.max_length = max_length
        self.classifier = None
        self.scaler = StandardScaler()

        # Use two open-source models as proxy for ada/davinci
        print("  Loading GPT-2 (proxy for ada)...")
        self.model_weak_tok = GPT2Tokenizer.from_pretrained("gpt2")
        if not self.model_weak_tok.pad_token:
            self.model_weak_tok.pad_token = self.model_weak_tok.eos_token
        self.model_weak = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

        print("  Loading Falcon-7B (proxy for davinci)...")
        self.model_strong_tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        if not self.model_strong_tok.pad_token:
            self.model_strong_tok.pad_token = self.model_strong_tok.eos_token
        self.model_strong = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b", device_map={"": device},
            torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    @torch.inference_mode()
    def _get_token_logprobs(self, text, model, tokenizer):
        """Get per-token log-probabilities under a model."""
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=self.max_length, return_token_type_ids=False).to(self.device)
        logits = model(**enc).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log-prob of each actual token
        labels = enc.input_ids[..., 1:]
        token_lps = log_probs[..., :-1, :].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return token_lps[0].cpu().float().numpy()

    def _extract_features(self, text):
        """Extract Ghostbuster-style features from a single text."""
        try:
            lp_weak = self._get_token_logprobs(text, self.model_weak, self.model_weak_tok)
            lp_strong = self._get_token_logprobs(text, self.model_strong, self.model_strong_tok)
        except Exception:
            return np.zeros(14)

        features = []
        for lp in [lp_weak, lp_strong]:
            if len(lp) == 0:
                features.extend([0.0] * 7)
                continue
            features.extend([
                np.mean(lp),            # avg log-prob
                np.std(lp),             # std of log-probs
                np.max(lp),             # max log-prob
                np.min(lp),             # min log-prob
                np.median(lp),          # median
                np.mean(lp > -1.0),     # fraction of "easy" tokens
                np.mean(np.diff(lp)) if len(lp) > 1 else 0.0,  # avg change
            ])
        return np.array(features)

    def _extract_features_batch(self, texts, label=""):
        """Extract features for a list of texts."""
        features = []
        for text in tqdm(texts, desc=f"  Ghostbuster features ({label})"):
            features.append(self._extract_features(text))
        return np.array(features)

    def train(self, human_texts, machine_texts):
        """Train the logistic regression classifier."""
        print("  Extracting training features...")
        h_feat = self._extract_features_batch(human_texts, "human")
        m_feat = self._extract_features_batch(machine_texts, "machine")

        X = np.vstack([h_feat, m_feat])
        y = np.concatenate([np.zeros(len(h_feat)), np.ones(len(m_feat))])

        # Remove any NaN/Inf
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.classifier.fit(X_scaled, y)
        train_acc = self.classifier.score(X_scaled, y)
        print(f"  Train accuracy: {train_acc:.4f}")

    def compute_score(self, input_text):
        """
        Returns probability of being machine-generated.
        Higher score = more likely machine.
        """
        if self.classifier is None:
            raise RuntimeError("Ghostbuster must be trained first. Call .train()")

        if isinstance(input_text, str):
            input_text = [input_text]

        features = self._extract_features_batch(input_text, "scoring")
        mask = np.isfinite(features).all(axis=1)
        features[~mask] = 0.0

        X_scaled = self.scaler.transform(features)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]  # P(machine)

        return probs.tolist() if len(probs) > 1 else probs[0].item()

    def train_ood(self, domain_data, test_domain):
        """
        Train in OOD setting: train on all domains except test_domain.
        domain_data: dict {domain_name: (human_texts, machine_texts)}
        test_domain: domain name to hold out
        """
        train_h, train_m = [], []
        for domain, (h, m) in domain_data.items():
            if domain != test_domain:
                train_h.extend(h)
                train_m.extend(m)
        print(f"  OOD training (excluding {test_domain}): {len(train_h)} human, {len(train_m)} machine")
        self.train(train_h, train_m)
