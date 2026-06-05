"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Contract for any embedding backend.

    Concrete implementations must supply both sync and async variants so the
    calling code can choose whichever fits its execution model.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* synchronously.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of float vectors in the same order as *texts*.

        Raises:
            EmbeddingError: On generation failure.
        """

    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* asynchronously.

        Args:
            texts: Non-empty list of strings.

        Returns:
            List of float vectors in the same order as *texts*.

        Raises:
            EmbeddingError: On generation failure.
        """

    # ------------------------------------------------------------------
    # Convenience single-text wrappers
    # ------------------------------------------------------------------

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    async def aembed_one(self, text: str) -> List[float]:
        return (await self.aembed([text]))[0]
