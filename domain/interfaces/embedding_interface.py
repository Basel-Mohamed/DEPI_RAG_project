from abc import ABC, abstractmethod
from typing import List

class EmbeddingInterface(ABC):

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass