from dataclasses import replace

from errorflow.meta_routing import CANONICAL_TOKENIZER, build_meta_features, decide_policies
from errorflow.v3_memory import ResidualMemoryRecord, ResidualMemoryStore


ACTION = "conflicting_rescue_v1"


def memory(index, label="SUPPORTED", recovery=1, regression=0, fold=1, claim="alpha beta"):
    return ResidualMemoryRecord(
        memory_id=f"m{index}", version=1, source_dataset="AVeriTeC",
        source_split="train", source_claim_id=f"train:{index}", source_claim=claim,
        error_pattern="x", applicability_conditions=[], correction_strategy=ACTION,
        retrieval_features={
            "baseline_prediction": label,
            "label_scores": {"SUPPORTED": -0.1, "REFUTED": -2.0},
            "evidence_tokens": 10,
        },
        training_support=1, recovery_count=recovery, regression_count=regression,
        estimated_net_utility=float(recovery - regression), estimated_extra_cost=1.0,
        validation_status="test", provenance={"outer_fold": fold},
    )


def baseline(label="SUPPORTED"):
    # Deliberately contains neither gold nor A3 output.
    return {
        "example_id": "query", "claim": "alpha beta", "prediction": label,
        "label_scores": {"SUPPORTED": -0.1, "REFUTED": -2.0},
        "evidence_tokens": 10,
    }


def test_canonical_tokenizer_is_frozen_repository_pattern():
    assert CANONICAL_TOKENIZER == r"[\w'-]+"


def test_features_contain_no_gold_or_candidate_information():
    store = ResidualMemoryStore([memory(1), memory(2)])
    payload = build_meta_features("query", baseline(), store).to_dict()
    serialized = str(payload).lower()
    assert "gold" not in serialized
    assert "candidate_prediction" not in serialized


def test_fold_and_self_exclusion_are_applied():
    records = [memory(1, fold=0), memory(2, fold=1), memory(3, fold=1)]
    feature = build_meta_features(
        "train:1", baseline(), ResidualMemoryStore(records),
        exclude_claim_ids={"train:1"}, exclude_folds={0},
    )
    assert "train:1" not in feature.nlf.neighbor_claim_ids
    assert feature.nlf.support == 2


def test_m1_is_union_and_m2_beta_zero_is_m1():
    store = ResidualMemoryStore([memory(1), memory(2), memory(3, label="REFUTED")])
    feature = build_meta_features("query", baseline(), store)
    decisions = decide_policies(feature, beta=0.0)
    assert decisions["M1_MaxUtility"].route == (
        decisions["D0_HardLabel"].route or decisions["D1_NLF"].route
    )
    assert decisions["M2_Reliability"].route == decisions["M1_MaxUtility"].route


def test_m3a_has_no_effect_when_fixed_policies_agree():
    store = ResidualMemoryStore([memory(1), memory(2)])
    feature = build_meta_features("query", baseline(), store)
    decisions = decide_policies(feature, beta=2.0)
    assert decisions["D0_HardLabel"].route == decisions["D1_NLF"].route
    assert decisions["M3a_DisagreementGate"].route == decisions["D0_HardLabel"].route
    assert decisions["M3a_DisagreementGate"].reason == "fixed_policies_agree"


def test_random_arbitration_is_deterministic_per_seed():
    records = [memory(1), memory(2), memory(3, label="REFUTED", regression=1, recovery=0)]
    feature = build_meta_features("query", baseline(), ResidualMemoryStore(records))
    first = decide_policies(feature, random_seed=2026)["D4_Random"]
    second = decide_policies(feature, random_seed=2026)["D4_Random"]
    assert first == second


def test_insufficient_support_has_deterministic_fallback():
    feature = build_meta_features("query", baseline(), ResidualMemoryStore([memory(1)]), min_support=2)
    decisions = decide_policies(feature)
    assert not decisions["D0_HardLabel"].route
    assert not decisions["D1_NLF"].route
    assert feature.hard.ineligibility_reason == "insufficient_support"


def test_uncertainty_is_finite_with_identical_weights():
    feature = build_meta_features(
        "query", baseline(), ResidualMemoryStore([memory(1), memory(2, regression=1, recovery=0)])
    )
    assert feature.hard.effective_sample_size > 0
    assert feature.hard.standard_error >= 0


def test_neighbor_identity_is_auditable():
    feature = build_meta_features("query", baseline(), ResidualMemoryStore([memory(1), memory(2)]))
    assert feature.hard.neighbor_ids == ("m1", "m2")
    assert feature.nlf.neighbor_claim_ids == ("train:1", "train:2")


def test_cross_label_counts_are_pre_action_features():
    store = ResidualMemoryStore([memory(1), memory(2), memory(3, label="REFUTED")])
    feature = build_meta_features("query", baseline(), store)
    assert feature.hard.cross_label_count == 0
    assert feature.nlf.cross_label_count == 1
