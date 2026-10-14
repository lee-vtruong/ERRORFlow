from errorflow.v3_memory import ResidualMemoryRecord
from scripts.evaluate_v3_external_memory_sensitivity import subset_definitions
from scripts.evaluate_v3_nlf_sensitivity import stable_sample, utility_bin
from scripts.evaluate_v3_substitution_ablation import VARIANTS


def memory(index, predicted="SUPPORTED", gold="REFUTED"):
    return ResidualMemoryRecord(
        memory_id=f"m{index}", version=1, source_dataset="x", source_split="train",
        source_claim_id=f"c{index}", source_claim="claim", error_pattern="x",
        applicability_conditions=[], correction_strategy="a",
        retrieval_features={"baseline_prediction": predicted}, training_support=1,
        recovery_count=0, regression_count=0, estimated_net_utility=0.0,
        estimated_extra_cost=1.0, validation_status="test",
        provenance={"gold_label_offline_only": gold},
    )


def test_nc_variants_are_all_no_label_filter():
    names = [f"NC{index}" for index in range(9)]
    matched = [config for name, config in VARIANTS.items() if any(name.startswith(x) for x in names)]
    assert len(matched) == 9
    assert all(config["compatibility"] == "none" for config in matched)


def test_stable_memory_sample_is_order_invariant():
    rows = [memory(index) for index in range(10)]
    left = [record.memory_id for record in stable_sample(rows, 4, 2026)]
    right = [record.memory_id for record in stable_sample(list(reversed(rows)), 4, 2026)]
    assert left == right


def test_external_subsets_include_full_predicted_and_gold_groups():
    rows = [memory(1), memory(2, predicted="REFUTED", gold="SUPPORTED")]
    subsets, topics = subset_definitions(rows)
    assert not topics
    assert len(subsets["full"]) == 2
    assert len(subsets["source_predicted_label=SUPPORTED"]) == 1
    assert len(subsets["source_gold_label=SUPPORTED"]) == 1


def test_utility_bins_keep_sign_near_zero():
    assert utility_bin(-0.005) == "-0.01<=u<0"
    assert utility_bin(0.005) == "0<=u<0.01"
