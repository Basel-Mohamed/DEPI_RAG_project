class RagBuildPipeline:

    def __init__(
        self,
        parser,
        chunker,
        embedding_service,
        vector_store
    ):
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store


    async def run(self, file_path):

        text = self.parser.parse(file_path)

        chunks = self.chunker.split(text)

        embeddings = await self.embedding_service.embed_documents(chunks)

        ids = [str(i) for i in range(len(chunks))]

        metadata = [{"text": chunk} for chunk in chunks]

        await self.vector_store.add_documents(
            ids,
            embeddings,
            metadata
        )