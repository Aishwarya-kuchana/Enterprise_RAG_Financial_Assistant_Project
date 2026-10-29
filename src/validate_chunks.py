from transformers import AutoTokenizer

from src.document_loader import DocumentLoader
from src.chunking import ChunkCreator

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens = 256
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("\nLoading PDF documents...")
    loader = DocumentLoader()
    documents = loader.load_documents()
    print(f"Total document pages loaded: {len(documents)}")
    print("\nCreating chunks...")
    chunker = ChunkCreator(
    model_name=model_name,
    chunk_size=200,
    chunk_overlap=30
    )
    chunks = chunker.create_chunks(documents)
    print(f"Total chunks created: {len(chunks)}")
    token_counts = []
    for chunk in chunks:
        tokens = tokenizer(
        chunk.page_content,
        add_special_tokens=True,
        truncation=False
        )["input_ids"]

        token_counts.append(len(tokens))

    oversized = [
    count for count in token_counts
    if count > max_tokens
    ]

    print("\n========== CHUNK VALIDATION ==========")
    print(f"Model token limit: {max_tokens}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks within limit: {len(chunks) - len(oversized)}")
    print(f"Chunks exceeding limit: {len(oversized)}")
    if token_counts:
        print(f"Shortest chunk: {min(token_counts)} tokens")
        print(f"Longest chunk: {max(token_counts)} tokens")
        print(
        f"Percentage exceeding limit: "
        f"{len(oversized) / len(chunks) * 100:.2f}%"
        )
        print("\n========== METADATA CHECK ==========")
        if chunks:
            print(f"Company: {chunks[0].metadata.get('company')}")
            print(f"Year: {chunks[0].metadata.get('year')}")
            print(f"Source: {chunks[0].metadata.get('source')}")
            print(f"Page: {chunks[0].metadata.get('page')}")

        if oversized:
            print("\nSome chunks exceed the model limit.")
            print("We need to investigate before rebuilding FAISS.")
        else:
            print("\nAll chunks are within the token limit.")
            print("\nExisting FAISS index has not been modified.")


if __name__ == "__main__":
    main()
