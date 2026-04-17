from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkingService:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def split(self, text: str):

        return self.splitter.split_text(text)