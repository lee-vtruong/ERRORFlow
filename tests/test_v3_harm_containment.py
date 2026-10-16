import json
import sys

from errorflow.io import write_jsonl
from scripts.summarize_v3_harm_containment import main


def test_harm_containment_cli(tmp_path, monkeypatch):
    gold = [
        {"example_id": "a", "label": "SUPPORTED"},
        {"example_id": "b", "label": "SUPPORTED"},
        {"example_id": "c", "label": "REFUTED"},
    ]
    base = [
        {"example_id": "a", "prediction": "REFUTED"},
        {"example_id": "b", "prediction": "SUPPORTED"},
        {"example_id": "c", "prediction": "REFUTED"},
    ]
    alt = [
        {"example_id": "a", "prediction": "SUPPORTED"},
        {"example_id": "b", "prediction": "REFUTED"},
        {"example_id": "c", "prediction": "SUPPORTED"},
    ]
    router = [
        {"example_id": "a", "policy": "R", "mode": "natural", "routed": True, "prediction": "SUPPORTED"},
        {"example_id": "b", "policy": "R", "mode": "natural", "routed": False, "prediction": "SUPPORTED"},
        {"example_id": "c", "policy": "R", "mode": "natural", "routed": False, "prediction": "REFUTED"},
    ]
    paths = [tmp_path / name for name in ("gold.jsonl", "base.jsonl", "alt.jsonl", "router.jsonl")]
    for path, rows in zip(paths, (gold, base, alt, router)):
        write_jsonl(path, rows)
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", [
        "summarize_v3_harm_containment.py", "--name", "test",
        "--gold", str(paths[0]), "--baseline", str(paths[1]),
        "--candidate", str(paths[2]), "--router-predictions", str(paths[3]),
        "--router-policy", "R", "--router-mode", "natural", "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"]["Alt-All"]["helpful"] == 1
    assert report["results"]["Alt-All"]["harmful"] == 2
    assert report["results"]["Router"]["helpful"] == 1
    assert report["results"]["Router"]["harmful"] == 0
    assert report["harm_containment"]["harm_avoided_fraction"] == 1.0
    assert report["harm_containment"]["helpful_retained_fraction"] == 1.0
