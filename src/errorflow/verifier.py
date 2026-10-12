"""Verifier interface and Qwen3-compatible prompt/parser utilities."""

from dataclasses import dataclass
import re
import time

from .contracts import PredictionRecord


LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")


@dataclass(frozen=True)
class VerifierConfig:
    model_name_or_path: str
    checkpoint: str = "base"
    seed: int | None = None
    max_new_tokens: int = 32
    label_space: tuple[str, ...] = LABELS


def build_prompt(claim: str, evidence: list[str], instruction: str = "", label_space: tuple[str, ...] = LABELS) -> str:
    joined = "\n".join(f"[{i + 1}] {item}" for i, item in enumerate(evidence))
    return (
        "Classify the claim using the evidence. Return exactly one label: "
        f"{', '.join(label_space)}.\n\n"
        f"Claim: {claim}\nEvidence:\n{joined}\n{instruction}\nLabel:"
    )


def parse_label(text: str, label_space: tuple[str, ...] = LABELS) -> str:
    normalized = re.sub(r"[^A-Z ]", " ", text.upper())
    for label in label_space:
        if label in normalized:
            return label
    raise ValueError(f"Could not parse a supported label from: {text!r}")


class Verifier:
    """Dependency-light interface; model-specific loading is intentionally deferred."""

    def __init__(self, config: VerifierConfig):
        self.config = config

    def record(
        self,
        claim_id: str,
        claim: str,
        evidence: list[str],
        raw_output: str,
        *,
        token_count: int = 0,
        latency_ms: float = 0.0,
        split: str = "unknown",
    ) -> PredictionRecord:
        return PredictionRecord(
            claim_id=claim_id,
            claim=claim,
            prediction=parse_label(raw_output, self.config.label_space),
            evidence_ids=[str(i) for i in range(len(evidence))],
            model=self.config.model_name_or_path,
            checkpoint=self.config.checkpoint,
            seed=self.config.seed,
            split=split,
            token_count=token_count,
            latency_ms=latency_ms,
            metadata={"raw_output": raw_output},
        )


class Qwen3TransformersVerifier(Verifier):
    """Single-example Qwen3 inference with lazy model loading."""

    def __init__(self, config: VerifierConfig, *, device_map: str = "auto"):
        super().__init__(config)
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers in the server environment") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            dtype=dtype,
            device_map=self.device_map,
        )
        self._model.eval()

    def predict(self, claim_id: str, claim: str, evidence: list[str], *, split: str = "unknown", instruction: str = "") -> PredictionRecord:
        if self._model is None or self._tokenizer is None:
            self.load()
        prompt = build_prompt(claim, evidence, instruction, self.config.label_space)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        started = time.perf_counter()
        with __import__("torch").inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = output.sequences
        new_tokens = sequences[0, inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        token_count = int(new_tokens.shape[-1])
        latency_ms = (time.perf_counter() - started) * 1000
        record = self.record(
            claim_id, claim, evidence, raw,
            token_count=token_count, latency_ms=latency_ms, split=split,
        )
        if output.scores:
            first_probs = __import__("torch").softmax(output.scores[0][0].float(), dim=-1)
            record.confidence = float(first_probs.max().item())
            record.metadata["confidence_type"] = "first_generated_token_max_probability"
        return record

    def score_labels(self, claim_id: str, claim: str, evidence: list[str], *, split: str = "unknown", instruction: str = "") -> PredictionRecord:
        """Choose the label with highest mean conditional token log-probability."""
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        prompt = build_prompt(claim, evidence, instruction, self.config.label_space)
        prompt_inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_ids = prompt_inputs["input_ids"]
        scores = {}
        candidate_lengths = {}
        started = time.perf_counter()
        with torch.inference_mode():
            for label in self.config.label_space:
                candidate_ids = self._tokenizer(
                    " " + label, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self._model.device)
                full_ids = torch.cat((prompt_ids, candidate_ids), dim=1)
                attention_mask = torch.ones_like(full_ids)
                logits = self._model(input_ids=full_ids, attention_mask=attention_mask).logits
                start = prompt_ids.shape[1] - 1
                candidate_logits = logits[0, start:start + candidate_ids.shape[1], :]
                token_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
                selected = token_log_probs.gather(1, candidate_ids[0].unsqueeze(1)).squeeze(1)
                scores[label] = float(selected.mean().item())
                candidate_lengths[label] = int(candidate_ids.shape[1])
        latency_ms = (time.perf_counter() - started) * 1000
        best = max(self.config.label_space, key=lambda label: scores[label])
        probabilities = torch.softmax(torch.tensor([scores[label] for label in self.config.label_space]), dim=0)
        record = self.record(claim_id, claim, evidence, best, token_count=0, latency_ms=latency_ms, split=split)
        record.confidence = float(probabilities[self.config.label_space.index(best)].item())
        record.metadata.update({
            "decoding": "mean_label_log_probability",
            "label_scores": scores,
            "candidate_token_lengths": candidate_lengths,
            "model_forward_passes": len(self.config.label_space),
            "confidence_type": "softmax_over_mean_candidate_label_log_probabilities",
        })
        return record
