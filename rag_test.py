from dotenv import load_dotenv

# ----------------------------
# Embedding imports FIRST
# ----------------------------

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever

print("Loading environment...")
load_dotenv()

print("Loading embedding model...")
embedding_loader = EmbeddingModel()
embedding_model = embedding_loader.load_model()

print("Loading vector database...")
vector_store = VectorStore()
vector_db = vector_store.load_vector_db(embedding_model)

print("Creating retriever...")
retriever = Retriever(vector_db)

# ----------------------------
# LLM imports AFTER embeddings
# ----------------------------

from src.llm import LLMModel
from src.rag_chain import RAGChain

print("Loading Gemini...")
llm_loader = LLMModel()
llm = llm_loader.load_llm()

print("Creating RAG chain...")
rag = RAGChain(retriever, llm)

print("\nAsking Question...\n")

result = rag.ask("Who is the CEO of Alphabet?")

print("=" * 60)
print("ANSWER")
print("=" * 60)
print(result["answer"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, doc in enumerate(result["documents"][:3], start=1):
    print(f"\nSource {i}")
    print(doc.metadata)

print("\nSUCCESS")