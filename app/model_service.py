from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import shap

from app.config import settings
try:
    import lime
    from lime.lime_text import LimeTextExplainer
except Exception:
    lime = None
    LimeTextExplainer = None  # type: ignore


class ModelService:
    """Transformer inference service with simple SHAP-compatible hooks."""

    def __init__(self, model_name: str | None = None, num_labels: int = 2):
        self.model_name = model_name or settings.model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_source = settings.model_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.current_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.current_source, num_labels=self.num_labels
        ).to(self.device)
        self.model.eval()
        self._explainer: Optional[shap.Explainer] = None
        self._lime_explainer: Optional[LimeTextExplainer] = None  # type: ignore
        self.temperature: float = 1.0

    def classify_text(self, text: str) -> Dict[str, Any]:
        # Normalize to plain string
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""
        if not text or not text.strip():
            return {
                "prediction": 0,
                "confidence": 0.5,
                "probabilities": {"real": 0.5, "fake": 0.5},
            }

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
        except Exception:
            # As an extreme fallback, return neutral prediction instead of error
            return {
                "prediction": 0,
                "confidence": 0.5,
                "probabilities": {"real": 0.5, "fake": 0.5},
                "uncertainty": {"predictive_entropy": float(np.log(2.0)), "margin": 0.0},
            }
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Assume index 1 = fake, index 0 = real for binary classifiers
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())
        # Uncertainty metrics
        probs_np = probs.detach().cpu().numpy()
        eps = 1e-12
        entropy = float(-np.sum(probs_np * np.log(probs_np + eps)))
        margin = float(abs((probs_np[1] if probs_np.shape[-1] > 1 else 1 - probs_np[0]) - probs_np[0]))

        return {
            "prediction": pred_idx,  # 0 or 1
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0].item()),
                "fake": float(probs[1].item() if probs.shape[-1] > 1 else 1 - probs[0].item()),
            },
            "uncertainty": {"predictive_entropy": entropy, "margin": margin},
        }

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not texts:
            return results

        # Normalize any incoming structure to a clean list of strings
        texts_norm = self._to_list_of_strings(texts)

        try:
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        except Exception:
            # Fallback: coerce everything to plain strings
            texts_norm = [str(x) for x in texts_norm]
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)

        for i in range(len(texts_norm)):
            p = probs[i]
            pred_idx = int(torch.argmax(p).item())
            # Uncertainty metrics per example
            p_np = p.detach().cpu().numpy()
            eps = 1e-12
            entropy = float(-np.sum(p_np * np.log(p_np + eps)))
            margin = float(abs((p_np[1] if p.shape[-1] > 1 else 1 - p_np[0]) - p_np[0]))

            results.append(
                {
                    "prediction": pred_idx,
                    "confidence": float(torch.max(p).item()),
                    "probabilities": {
                        "real": float(p[0].item()),
                        "fake": float(p[1].item() if p.shape[-1] > 1 else 1 - p[0].item()),
                    },
                    "uncertainty": {"predictive_entropy": entropy, "margin": margin},
                }
            )
        return results

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.num_labels), dtype=float)
        texts_norm = self._to_list_of_strings(texts)
        try:
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        except Exception:
            texts_norm = [str(x) for x in texts_norm]
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def predict_logits(self, texts: List[str]) -> np.ndarray:
        """Return raw logits for a batch of texts as numpy array (N, C)."""
        if not texts:
            return np.zeros((0, self.num_labels), dtype=float)
        texts_norm = self._to_list_of_strings(texts)
        try:
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        except Exception:
            texts_norm = [str(x) for x in texts_norm]
            encodings = self.tokenizer(
                texts_norm,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
        return logits.detach().cpu().numpy()

    def predict_proba_mc(self, texts: List[str], mc_samples: int = 10) -> Dict[str, Any]:
        """Monte Carlo dropout predictive distribution and uncertainty for texts.

        Returns dict with keys:
          - mean_proba: np.ndarray shape (N, num_labels)
          - predictive_entropy: List[float]
          - expected_entropy: List[float]
          - mutual_information: List[float]
        """
        if not texts or mc_samples <= 0:
            return {
                "mean_proba": np.zeros((0, self.num_labels), dtype=float),
                "predictive_entropy": [],
                "expected_entropy": [],
                "mutual_information": [],
            }

        texts_norm = self._to_list_of_strings(texts)
        encodings = self.tokenizer(
            texts_norm,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Enable dropout
        self.model.train()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(mc_samples):
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.detach().cpu().numpy())
        # Restore eval mode
        self.model.eval()

        probs_stack = np.stack(all_probs, axis=0)  # (S, N, C)
        mean_proba = probs_stack.mean(axis=0)

        eps = 1e-12
        # Predictive entropy H[ E[p] ]
        predictive_entropy = (-np.sum(mean_proba * np.log(mean_proba + eps), axis=1)).tolist()
        # Expected entropy E[ H[p] ]
        expected_entropy = (-np.sum(probs_stack * np.log(probs_stack + eps), axis=2).mean(axis=0)).tolist()
        # Mutual information = predictive_entropy - expected_entropy
        mi = (np.array(predictive_entropy) - np.array(expected_entropy)).tolist()

        return {
            "mean_proba": mean_proba,
            "predictive_entropy": [float(x) for x in predictive_entropy],
            "expected_entropy": [float(x) for x in expected_entropy],
            "mutual_information": [float(x) for x in mi],
        }

    def explain_text(self, text: str, max_evals: Optional[int] = None) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"tokens": [], "shap_values": [], "base_value": 0.0}

        try:
            if self._explainer is None:
                # Use a regex-based text masker to ensure inputs remain plain strings
                # This avoids incompatibilities with certain SHAP + HF tokenizer versions.
                text_masker = shap.maskers.Text(r"\W+")  # split on non-word boundaries
                self._explainer = shap.Explainer(self._predict_proba, text_masker)

            evals = max_evals if max_evals is not None else settings.explanation_max_evals
            sv = self._explainer([text], max_evals=evals)

            # Extract tokens
            tokens: List[str]
            try:
                tokens = list(sv.data[0])  # type: ignore[attr-defined]
            except Exception:
                tokens = self.tokenizer.tokenize(text)

            # Extract values for class index 1 if available
            values_arr = sv.values
            base_arr = sv.base_values
            if isinstance(values_arr, np.ndarray):
                if values_arr.ndim == 3:  # [samples, tokens, classes]
                    shap_vals = values_arr[0, :, 1 if self.num_labels > 1 else 0].tolist()
                elif values_arr.ndim == 2:  # [samples, tokens]
                    shap_vals = values_arr[0, :].tolist()
                else:
                    shap_vals = []
            else:
                shap_vals = []

            if isinstance(base_arr, np.ndarray):
                if base_arr.ndim == 2:  # [samples, classes]
                    base_value = float(base_arr[0, 1 if self.num_labels > 1 else 0])
                elif base_arr.ndim == 1:
                    base_value = float(base_arr[0])
                else:
                    base_value = 0.0
            else:
                base_value = 0.0

            return {"tokens": tokens, "shap_values": shap_vals, "base_value": base_value}
        except Exception:
            # Fallback: return tokenizer tokens with zero weights instead of erroring out
            safe_tokens = self.tokenizer.tokenize(text)
            return {"tokens": safe_tokens, "shap_values": [0.0] * len(safe_tokens), "base_value": 0.0}

    def explain_text_lime(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """LIME explanation at feature-level (word n-grams). Best-effort if LIME available."""
        if not text or not text.strip():
            return {"features": [], "weights": []}
        try:
            from lime.lime_text import LimeTextExplainer  # type: ignore
        except Exception:
            return {"features": [], "weights": []}
        if self._lime_explainer is None:
            self._lime_explainer = LimeTextExplainer(class_names=["real", "fake"])  # type: ignore

    def explain_text_attention(
        self,
        text: str,
        aggregate_layers: str = "last",  # 'last' or 'mean'
        aggregate_heads: str = "mean",   # 'mean' or 'max'
    ) -> Dict[str, Any]:
        """Attention-based explanation using transformer attention maps.

        Returns tokens and per-token attention weights derived from the [CLS] token's
        attention distribution. Best-effort across models that expose attentions.
        """
        if not text or not text.strip():
            return {"tokens": [], "weights": []}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # type: ignore[attr-defined]

        if attentions is None or len(attentions) == 0:
            return {"tokens": [], "weights": []}

        # Select layer aggregation
        if aggregate_layers == "mean":
            # mean over layers -> tensor shape (num_heads, seq_len, seq_len)
            att = torch.stack(attentions, dim=0).mean(dim=0)[0]
        else:
            # last layer
            att = attentions[-1][0]

        # Aggregate heads
        if aggregate_heads == "max":
            att_mean = torch.max(att, dim=0).values  # (seq_len, seq_len)
        else:
            att_mean = torch.mean(att, dim=0)  # (seq_len, seq_len)

        # CLS attends to tokens: row 0
        cls_to_tokens = att_mean[0]  # (seq_len,)
        weights = cls_to_tokens.detach().cpu().numpy().astype(float)

        # Convert ids to tokens
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return {"tokens": [], "weights": []}
        token_ids = input_ids[0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Normalize weights to [0,1]
        w_min, w_max = float(weights.min()), float(weights.max())
        if w_max > w_min:
            weights_norm = ((weights - w_min) / (w_max - w_min)).tolist()
        else:
            weights_norm = [0.0 for _ in tokens]

        return {"tokens": tokens, "weights": weights_norm}

    def _to_list_of_strings(self, inputs: Any) -> List[str]:
        """Best-effort normalization to a list of raw strings for tokenization.

        Accepts a single string, list/tuple/ndarray of strings, or a nested
        list of tokens (which will be joined with spaces). Falls back to str(x).
        This is robust to SHAP text masker inputs which can be numpy arrays or
        object-dtype containers.
        """
        # Single string
        if isinstance(inputs, str):
            return [inputs]
        # Already a list/tuple/np.ndarray
        if isinstance(inputs, (list, tuple, np.ndarray)):
            result: List[str] = []
            for item in list(inputs):
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, (list, tuple, np.ndarray)):
                    # pretokenized tokens or nested arrays -> join into a string
                    try:
                        flat_tokens: List[str] = []
                        for tok in list(item):
                            if isinstance(tok, str):
                                flat_tokens.append(tok)
                            elif isinstance(tok, (list, tuple, np.ndarray)):
                                flat_tokens.extend([str(t) for t in list(tok)])
                            else:
                                flat_tokens.append(str(tok))
                        result.append(" ".join(flat_tokens))
                    except Exception:
                        result.append(str(item))
                else:
                    result.append(str(item))
            return result
        # Fallback
        return [str(inputs)]

    def get_top_influential_tokens_shap(self, text: str, top_k: int = 5, by: str = "abs") -> List[str]:
        """Return top-k influential tokens from SHAP explanation.

        by: 'abs' for absolute shap values, 'positive' for tokens pushing towards class 1 (fake).
        """
        exp = self.explain_text(text, max_evals=None)
        tokens = exp.get("tokens", [])
        values = exp.get("shap_values", [])
        if not tokens or not values or len(tokens) != len(values):
            return []
        if by == "positive":
            order = np.argsort(-np.array(values))
        else:
            order = np.argsort(-np.abs(np.array(values)))
        top_idx = order[: max(1, top_k)]
        return [str(tokens[i]) for i in top_idx if 0 <= i < len(tokens)]

    def generate_counterfactuals(self, text: str, max_candidates: int = 3) -> List[Dict[str, Any]]:
        """Generate simple counterfactuals by removing top influential tokens.

        Strategy: identify top SHAP tokens, create variants by removing one or more,
        return up to max_candidates variants with their predictions and probability deltas.
        """
        if not text or not text.strip():
            return []
        base = self.classify_text(text)
        base_prob_fake = float(base.get("probabilities", {}).get("fake", 0.0))
        top_tokens = self.get_top_influential_tokens_shap(text, top_k=6, by="abs")
        if not top_tokens:
            return []
        variants: List[str] = []
        # Create single-token removal variants, then two-token removals if needed
        words = text.split()
        for tok in top_tokens:
            v = " ".join([w for w in words if w != tok])
            if v and v != text:
                variants.append(v)
            if len(variants) >= max_candidates:
                break
        if len(variants) < max_candidates:
            # two-token removals
            for i in range(min(3, len(top_tokens))):
                for j in range(i + 1, min(4, len(top_tokens))):
                    t1, t2 = top_tokens[i], top_tokens[j]
                    v = " ".join([w for w in words if w not in {t1, t2}])
                    if v and v != text:
                        variants.append(v)
                    if len(variants) >= max_candidates:
                        break
                if len(variants) >= max_candidates:
                    break

        results: List[Dict[str, Any]] = []
        for v in variants[:max_candidates]:
            pred = self.classify_text(v)
            prob_fake = float(pred.get("probabilities", {}).get("fake", 0.0))
            results.append(
                {
                    "text": v,
                    "prediction": int(pred.get("prediction", 0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                    "probabilities": pred.get("probabilities", {}),
                    "delta_fake_probability": float(prob_fake - base_prob_fake),
                }
            )
        return results

    def explanation_confidence(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """Estimate confidence of explanation by measuring prediction stability
        after removing top-k influential tokens.
        Returns: { base_prob_fake, perturbed_prob_fake, delta, stability_label }
        """
        base = self.classify_text(text)
        base_prob_fake = float(base.get("probabilities", {}).get("fake", 0.0))
        top_tokens = self.get_top_influential_tokens_shap(text, top_k=top_k, by="abs")
        if not top_tokens:
            return {"base_prob_fake": base_prob_fake, "perturbed_prob_fake": base_prob_fake, "delta": 0.0, "stability_label": "high"}
        words = text.split()
        v = " ".join([w for w in words if w not in set(top_tokens)])
        if not v:
            return {"base_prob_fake": base_prob_fake, "perturbed_prob_fake": base_prob_fake, "delta": 0.0, "stability_label": "unknown"}
        pred = self.classify_text(v)
        prob_fake = float(pred.get("probabilities", {}).get("fake", 0.0))
        delta = abs(prob_fake - base_prob_fake)
        # Heuristic label
        if delta < 0.05:
            label = "high"
        elif delta < 0.15:
            label = "medium"
        else:
            label = "low"
        return {
            "base_prob_fake": base_prob_fake,
            "perturbed_prob_fake": prob_fake,
            "delta": float(delta),
            "stability_label": label,
        }

    def reload(self, model_source: Optional[str] = None) -> Dict[str, Any]:
        """Reload model/tokenizer, optionally specifying a model path/name."""
        source = model_source or settings.model_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=self.num_labels
        ).to(self.device)
        self.model.eval()
        self.current_source = source
        self._explainer = None
        self.temperature = 1.0
        return self.get_model_info()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "current_source": self.current_source,
            "device": str(self.device),
            "num_labels": self.num_labels,
            "uses_local_path": settings.model_path is not None,
            "configured_model_name": self.model_name,
            "temperature": self.temperature,
        }

    def set_temperature(self, temperature: float) -> None:
        try:
            t = float(temperature)
            if t <= 0:
                return
            self.temperature = t
        except Exception:
            pass

    def calibrate_temperature(self, texts: List[str], labels: List[int], max_steps: int = 200, lr: float = 0.01) -> float:
        if not texts or not labels:
            return self.temperature
        # Get logits without applying current temperature
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits.detach()

        y = torch.tensor(labels, dtype=torch.long, device=logits.device)
        temperature = torch.tensor([1.0], device=logits.device, requires_grad=True)
        optimizer = torch.optim.Adam([temperature], lr=lr)

        for _ in range(max_steps):
            optimizer.zero_grad()
            scaled = logits / torch.clamp(temperature, min=1e-6)
            loss = torch.nn.functional.cross_entropy(scaled, y)
            loss.backward()
            optimizer.step()

        new_t = float(temperature.detach().cpu().item())
        if new_t > 0:
            self.temperature = new_t
        return self.temperature


# Singleton service
model_service = ModelService()


