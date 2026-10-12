import json

from scripts.download_averitec import download


def test_download_reuses_and_audits_existing_file(tmp_path):
    target = tmp_path / "train.json"
    target.write_text(json.dumps([{"claim": "x", "label": "Supported"}]), encoding="utf-8")
    result = download("https://unused.invalid/train.json", target)
    assert result["status"] == "reused"
    assert result["rows"] == 1
    assert result["labels"] == {"Supported": 1}
