import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen3_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_qwen3_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_completed_claim_ids_reads_valid_output(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"claim_id":"a"}\n{"claim_id":"b"}\n', encoding="utf-8")
    assert MODULE.completed_claim_ids(output) == {"a", "b"}


def test_completed_claim_ids_rejects_duplicates(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"claim_id":"a"}\n{"claim_id":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate claim_id"):
        MODULE.completed_claim_ids(output)


def test_load_jsonl_reports_corrupt_partial_line(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"claim_id":"a"}\n{"claim_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        MODULE.load_jsonl(output)
