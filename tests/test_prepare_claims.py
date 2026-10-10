import json
import subprocess
import sys


def test_prepare_claims_does_not_copy_gold_evidence(tmp_path):
    source = tmp_path / "dev.jsonl"
    output = tmp_path / "out.jsonl"
    source.write_text(json.dumps({"id": 1, "claim": "claim", "label": "SUPPORTS", "evidence": [[[0, 0, "Page", 1]]]}) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "scripts/prepare_fever_claims.py", "--input", str(source), "--output", str(output)], check=True, env={"PYTHONPATH": "src"})
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["evidence"] == []
    assert row["preparation"] == "claim_only_no_gold_evidence"
