"""Dataset label normalization helpers."""

FEVER_LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
}


def normalize_fever_label(label: str) -> str:
    try:
        return FEVER_LABEL_MAP[label]
    except KeyError as exc:
        raise ValueError(f"Unknown FEVER label: {label!r}") from exc
