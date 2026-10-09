"""Verifier interface and Qwen3-compatible prompt/parser utilities."""

from dataclasses import dataclass
import re

from .contracts import PredictionRecord


LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")


@dataclass(frozen=True)
class VerifierConfig:
    model_name_or_path: str
    checkpoint: str = "base"
    seed: int | None = None
    max_new_tokens: int = 32


def build_prompt(claim: str, evidence: list[str]) -> str:
    joined = "\n".join(f"[{i + 1}] {item}" for i, item in enumerate(evidence))
    return (
        "Classify the claim using the evidence. Return exactly one label: "
        "SUPPORTED, REFUTED, or NOT ENOUGH INFO.\n\n"
        f"Claim: {claim}\nEvidence:\n{joined}\nLabel:"
    )


def parse_label(text: str) -> str:
    normalized = re.sub(r"[^A-Z ]", " ", text.upper())
    for label in LABELS:
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
            prediction=parse_label(raw_output),
            evidence_ids=[str(i) for i in range(len(evidence))],
            model=self.config.model_name_or_path,
            checkpoint=self.config.checkpoint,
            seed=self.config.seed,
            split=split,
            token_count=token_count,
            latency_ms=latency_ms,
            metadata={"raw_output": raw_output},
        )
