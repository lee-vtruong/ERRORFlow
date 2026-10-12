"""Embedding backends with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class HashingEmbedder:
    """Stateless signed hashing vectorizer for tests and offline pilots."""

    dimensions: int = 256
    name: str = "hashing-v1"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += 1.0 if (value >> 8) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerEmbedder:
    """Lazy optional backend; importing ERRORFlow never loads model weights."""

    def __init__(self, model: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install ERRORFlow with the 'v2' extra for dense embeddings") from exc
        self.name = model
        self._model = SentenceTransformer(model, device=device)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right))
