"""Interpretable, versioned residual action memory for ERRORFlow V3."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from .error_memory import distribution_similarity, jaccard_similarity
from .io import read_jsonl, write_jsonl


def residual_memory_id(dataset, claim_id, action, version=1):
    # The logical ID is stable across versions so upsert can enforce monotonic updates.
    payload = "\x1f".join((dataset, claim_id, action))
    return "REM_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class ResidualMemoryRecord:
    memory_id: str
    version: int
    source_dataset: str
    source_split: str
    source_claim_id: str
    source_claim: str
    error_pattern: str
    applicability_conditions: list[str]
    correction_strategy: str
    retrieval_features: dict
    training_support: int
    recovery_count: int
    regression_count: int
    estimated_net_utility: float | None
    estimated_extra_cost: float | None
    validation_status: str
    active: bool = True
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def record_from_dict(payload):
    allowed = {item.name for item in fields(ResidualMemoryRecord)}
    return ResidualMemoryRecord(**{key: value for key, value in payload.items() if key in allowed})


def smoothed_utility(recovery_count, regression_count, support, alpha=1.0, beta=1.0):
    if support < recovery_count + regression_count:
        raise ValueError("support cannot be smaller than recovery plus regression")
    recovery = (recovery_count + alpha) / (support + alpha + beta)
    regression = (regression_count + alpha) / (support + alpha + beta)
    return recovery - regression


class ResidualMemoryStore:
    def __init__(self, records=()):
        self._records = {}
        self._ranked_cache = None
        for record in records:
            self.upsert(record)

    def upsert(self, record):
        current = self._records.get(record.memory_id)
        if current and record.version < current.version:
            raise ValueError("Cannot replace a memory with an older version")
        self._records[record.memory_id] = record
        self._ranked_cache = None

    def deactivate(self, memory_id, reason):
        record = self._records[memory_id]
        provenance = dict(record.provenance)
        provenance["deactivation_reason"] = reason
        self._records[memory_id] = replace(record, active=False, provenance=provenance)
        self._ranked_cache = None

    def save(self, path):
        write_jsonl(path, (self._records[key].to_dict() for key in sorted(self._records)))

    @classmethod
    def load(cls, path):
        return cls(record_from_dict(row) for row in read_jsonl(path))

    def retrieve(self, query, top_k=10, allowed_splits=None, action=None,
                 exclude_claim_ids=None, exclude_folds=None, max_records=None):
        scored = []
        query_features = query.get("retrieval_features", {})
        baseline_prediction = query_features.get("baseline_prediction")
        records = sorted(self._records.values(), key=lambda record: record.memory_id)
        if max_records is not None:
            if self._ranked_cache is None:
                self._ranked_cache = sorted(
                    self._records.values(),
                    key=lambda record: (-(abs(record.estimated_net_utility or 0.0)), record.memory_id),
                )
            records = self._ranked_cache[:max_records]
        allowed_splits = {"train"} if allowed_splits is None else allowed_splits
        for record in records:
            if not record.active or record.source_split not in allowed_splits:
                continue
            if action and record.correction_strategy != action:
                continue
            if exclude_claim_ids and record.source_claim_id in exclude_claim_ids:
                continue
            if exclude_folds and record.provenance.get("outer_fold") in exclude_folds:
                continue
            features = record.retrieval_features
            if baseline_prediction and features.get("baseline_prediction") != baseline_prediction:
                continue
            lexical = jaccard_similarity(query.get("claim", ""), record.source_claim)
            score_profile = distribution_similarity(
                query_features.get("label_scores", {}), features.get("label_scores", {})
            )
            length_delta = abs(float(query_features.get("evidence_tokens", 0)) - float(features.get("evidence_tokens", 0)))
            length_similarity = 1.0 / (1.0 + length_delta / 100.0)
            score = 0.35 * lexical + 0.50 * score_profile + 0.15 * length_similarity
            scored.append((record, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].memory_id))[:top_k]

    def __len__(self):
        return len(self._records)


def build_action_memories(baseline_rows, named_candidates, dataset="AVeriTeC"):
    """Create action outcomes for both residual errors and correct controls."""
    baseline = {str(row["example_id"]): row for row in baseline_rows}
    candidates = {action: {str(row["example_id"]): row for row in rows} for action, rows in named_candidates}
    records = []
    for key, row in baseline.items():
        if row.get("split") != "train" or key.startswith("dev:"):
            raise ValueError(f"Only AVeriTeC train records may enter V3 memory: {key}")
        for action, action_rows in candidates.items():
            if key not in action_rows:
                raise ValueError(f"Candidate {action} is missing {key}")
            after = action_rows[key]
            before_correct = bool(row["correct"])
            after_correct = after["prediction"] == row["gold_label"]
            recovered = not before_correct and after_correct
            regressed = before_correct and not after_correct
            utility = 1.0 if recovered else (-1.0 if regressed else 0.0)
            # V3 label scoring always evaluates four labels in four forward
            # passes. Early cached rows record this at manifest level only, so
            # the per-row fallback must be four rather than one.
            extra_passes = float(after.get("metadata", {}).get("model_forward_passes", 4))
            records.append(ResidualMemoryRecord(
                memory_id=residual_memory_id(dataset, key, action), version=1,
                source_dataset=dataset, source_split="train", source_claim_id=key,
                source_claim=row["claim"], error_pattern=(row["diagnosis"]["error_type"] if not before_correct else "correct_control"),
                applicability_conditions=[f"baseline_prediction={row['prediction']}"],
                correction_strategy=action,
                retrieval_features={
                    "baseline_prediction": row["prediction"],
                    "label_scores": row.get("label_scores", {}),
                    "prediction_margin": row.get("prediction_margin"),
                    "claim_tokens": row.get("claim_tokens", 0),
                    "evidence_count": row.get("evidence_count", 0),
                    "evidence_tokens": row.get("evidence_tokens", 0),
                },
                training_support=1, recovery_count=int(recovered), regression_count=int(regressed),
                estimated_net_utility=utility, estimated_extra_cost=extra_passes / 4.0,
                validation_status="unvalidated_train_outcome", active=True,
                provenance={
                    "outer_fold": row.get("outer_fold"), "gold_label_offline_only": row["gold_label"],
                    "baseline_correct_offline_only": before_correct, "candidate_prediction": after["prediction"],
                    "candidate_latency_ms": float(after.get("latency_ms", 0)),
                    "candidate_tokens": int(after.get("token_count", 0)),
                },
            ))
    return records
