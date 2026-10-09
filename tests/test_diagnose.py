import json, subprocess, sys

def test_diagnosis_is_explicitly_heuristic(tmp_path):
    source = tmp_path / "in.jsonl"; output = tmp_path / "out.jsonl"
    source.write_text(json.dumps({"gold_label": "NOT ENOUGH INFO", "baseline_prediction": "REFUTED"}) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "scripts/diagnose_errors.py", "--input", str(source), "--output", str(output)], check=True)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["diagnosis_status"] == "heuristic_candidate"
    assert "evidence_critic" in row["candidate_actions"]
