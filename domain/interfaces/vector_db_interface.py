from abc import ABC, abstractmethod
from typing import Optional

from domain.models.document import Document


class VectorDBInterface(ABC):
    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ):
        pass

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        top_k: int,
        filter: Optional[dict] = None,
    ):
        pass

    @abstractmethod
    async def delete_documents(self, ids: list[str]):
        pass
