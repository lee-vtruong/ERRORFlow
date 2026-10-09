"""Generate a tiny deterministic baseline/error-trace fixture."""

import json
from pathlib import Path

from errorflow.contracts import ExecutionTrace, PredictionRecord


def main() -> None:
    output = Path("outputs/step2_demo")
    output.mkdir(parents=True, exist_ok=True)
    predictions = [
        PredictionRecord("demo_001", "The event happened in 2020.", "SUPPORTED", ["e1"], 0.62, "demo", "fixture", 0, "train"),
        PredictionRecord("demo_002", "The event happened in 2024.", "SUPPORTED", ["e2"], 0.58, "demo", "fixture", 0, "train"),
        PredictionRecord("demo_003", "The event happened in 2019.", "REFUTED", ["e3"], 0.81, "demo", "fixture", 0, "train"),
    ]
    traces = [
        ExecutionTrace(p.claim_id, ["initial_verifier"], [], p.token_count, 1, p.latency_ms)
        for p in predictions
    ]
    for name, records in (("predictions.jsonl", predictions), ("traces.jsonl", traces)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    (output / "README.md").write_text(
        "Deterministic Step 2 fixture. It is not an experiment result.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(predictions)} predictions and {len(traces)} traces to {output}")


if __name__ == "__main__":
    main()
