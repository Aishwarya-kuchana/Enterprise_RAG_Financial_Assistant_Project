from pathlib import Path
from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.vector_db_path = (
            self.project_root /
            "data" /
            "vector_db"
        )
    def create_vector_db(self, chunks, embedding_model):
        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        return vector_db
    def save_vector_db(self, vector_db):
        self.vector_db_path.mkdir(
            parents=True,
            exist_ok=True
        )
        vector_db.save_local(str(self.vector_db_path))
        print("Vector Database Saved Successfully!")
    def load_vector_db(self, embedding_model):
        vector_db = FAISS.load_local(
            str(self.vector_db_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_db