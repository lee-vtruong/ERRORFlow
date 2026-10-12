"""Dataset adapters for the V2 canonical schema."""

from .averitec import AVeriTeCAdapter
from .climate_fever import ClimateFeverAdapter
from .fever import FeverAdapter
from .scifact import SciFactAdapter

ADAPTERS = {
    "averitec": AVeriTeCAdapter,
    "climate-fever": ClimateFeverAdapter,
    "fever": FeverAdapter,
    "scifact": SciFactAdapter,
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter: {name}") from exc


__all__ = ["ADAPTERS", "get_adapter", "AVeriTeCAdapter", "ClimateFeverAdapter", "FeverAdapter", "SciFactAdapter"]
