from src.document_loader import DocumentLoader
from src.chunking import ChunkCreator
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def main():

    print("=" * 60)
    print("Building Financial RAG Knowledge Base")
    print("=" * 60)

    # ----------------------------
    # Load Documents
    # ----------------------------

    print("\nLoading Documents...")

    loader = DocumentLoader()

    documents = loader.load_documents()

    print(f"Loaded {len(documents)} documents.")

    # ----------------------------
    # Chunking
    # ----------------------------

    print("\nCreating Chunks...")

    chunker = ChunkCreator()

    chunks = chunker.create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    # ----------------------------
    # Embedding Model
    # ----------------------------

    print("\nLoading Embedding Model...")

    embedding_loader = EmbeddingModel()

    embedding_model = embedding_loader.load_model()

    print("Embedding Model Loaded.")

    # ----------------------------
    # Create Vector Database
    # ----------------------------

    print("\nCreating Vector Database...")

    vector_store = VectorStore()

    vector_db = vector_store.create_vector_db(
        chunks,
        embedding_model
    )

    print("Vector Database Created.")

    # ----------------------------
    # Save Vector Database
    # ----------------------------

    print("\nSaving Vector Database...")

    vector_store.save_vector_db(vector_db)

    print("\nKnowledge Base Built Successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()