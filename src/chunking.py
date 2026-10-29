# from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkCreator:
    def __init__(self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=30):
        
        self.text_splitter = (
        RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
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
        )
    def create_chunks(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        return chunks
