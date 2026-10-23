from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkCreator:
    def __init__(
        self,
        chunk_size=1500,
        chunk_overlap=300
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    def create_chunks(self, documents):
        chunks = self.text_splitter.split_documents(
            documents
        )
        return chunks