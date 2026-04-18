class CohereEmbeddings:
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0"):
        self.api_key = api_key
        self.model_name = model_name
        self._langchain_embeddings = None
        self._cohere_client = None

        if not api_key:
            raise ValueError("COHERE_API_KEY is required for document embedding.")

        try:
            from langchain_cohere import CohereEmbeddings as LangChainCohereEmbeddings
        except ImportError:
            LangChainCohereEmbeddings = None

        if LangChainCohereEmbeddings is not None:
            self._langchain_embeddings = LangChainCohereEmbeddings(
                cohere_api_key=api_key,
                model=model_name,
            )
            return

        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "Install either 'langchain-cohere' or 'cohere' to use Cohere embeddings."
            ) from exc

        self._cohere_client = cohere.Client(api_key)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._langchain_embeddings is not None:
            return self._langchain_embeddings.embed_documents(texts)

        response = self._cohere_client.embed(texts=texts, model=self.model_name)
        return response.embeddings

    async def embed_query(self, query: str) -> list[float]:
        if self._langchain_embeddings is not None:
            return self._langchain_embeddings.embed_query(query)

        response = self._cohere_client.embed(texts=[query], model=self.model_name)
        return response.embeddings[0]
