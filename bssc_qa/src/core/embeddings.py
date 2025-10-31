"""Offline-friendly embedding implementations."""
import hashlib
import math
import re
from typing import Iterable, List


class OfflineHashEmbeddings:
    """Lightweight hash-based embeddings that avoid network downloads."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._token_pattern = re.compile(r"\w+")

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed a collection of documents."""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """Generate a normalized hash-based vector for the given text."""
        vector = [0.0] * self.dimension
        tokens = self._token_pattern.findall(text.lower())

        for token in tokens:
            index = self._hash_token(token)
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]

        return vector

    def _hash_token(self, token: str) -> int:
        """Map a token to a deterministic bucket in the vector."""
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dimension
