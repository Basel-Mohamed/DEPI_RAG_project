from abc import ABC, abstractmethod
from typing import List, Optional

class VectorDBInterface(ABC):

    @abstractmethod
    async def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[dict]
    ):
        pass

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filter: Optional[dict] = None
    ):
        pass
    
    @abstractmethod
    async def delete_documents(self, ids: List[str]):
        pass