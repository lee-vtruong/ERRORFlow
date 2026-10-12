from scripts.compare_v2_interventions import compare


def report(recovered, regressed):
    return {
        "n": 10, "helpful": recovered, "harmful": regressed,
        "net_correct_change": recovered - regressed, "baseline_accuracy": 0.5,
        "candidate_accuracy": 0.5, "candidate_macro_f1": 0.5,
        "avg_candidate_latency_ms": 1.0, "avg_candidate_forward_passes": 4.0,
        "by_group": {"error:A->B": {"n": 5, "recovered": recovered, "regressed": 0, "changed": recovered}},
    }


def test_compare_selects_best_group_action():
    result = compare([("weak", report(1, 0)), ("strong", report(3, 0))])
    assert result["by_group"]["error:A->B"]["best_action"] == "strong"
    assert result["by_group"]["error:A->B"]["actions"]["strong"]["recovery_rate"] == 0.6
