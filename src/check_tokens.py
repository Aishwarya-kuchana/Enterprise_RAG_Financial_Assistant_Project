
from sentence_transformers import SentenceTransformer

from src.document_loader import DocumentLoader
from src.chunking import ChunkCreator


def main():
    # 1. Load the embedding model for inspection
    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    max_tokens = model.max_seq_length

    print(f"Maximum sequence length: {max_tokens} tokens")

    # 2. Load documents using your existing DocumentLoader
    print("\nLoading PDF documents...")

    loader = DocumentLoader()
    documents = loader.load_documents()

    print(f"Total documents/pages loaded: {len(documents)}")

    # 3. Create chunks using your existing ChunkCreator
    print("\nCreating chunks in memory...")

    chunk_creator = ChunkCreator()
    chunks = chunk_creator.create_chunks(documents)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks were created.")
        return

    # 4. Count tokens for every chunk without truncation
    print("\nChecking token counts...")

    token_counts = []

    for chunk in chunks:
        tokens = model.tokenizer(
            chunk.page_content,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        token_counts.append(len(tokens))

    # 5. Calculate summary statistics
    exceeding = [
        count for count in token_counts
        if count > max_tokens
    ]

    within_limit = len(token_counts) - len(exceeding)

    print("\n========== TOKEN ANALYSIS ==========")
    print(f"Model token limit: {max_tokens}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks within limit: {within_limit}")
    print(f"Chunks exceeding limit: {len(exceeding)}")

    print(
        f"Shortest chunk: {min(token_counts)} tokens"
    )
    print(
        f"Longest chunk: {max(token_counts)} tokens"
    )

    percentage = len(exceeding) / len(token_counts) * 100

    print(
        f"Percentage exceeding limit: {percentage:.2f}%"
    )

    # 6. Display five examples of oversized chunks
    print("\n========== OVERSIZED CHUNK EXAMPLES ==========")

    shown = 0

    for i, chunk in enumerate(chunks):
        if token_counts[i] > max_tokens:
            print(f"\nChunk index: {i}")
            print(f"Token count: {token_counts[i]}")
            print(
                f"Company: {chunk.metadata.get('company')}"
            )
            print(
                f"Year: {chunk.metadata.get('year')}"
            )
            print(
                f"Page: {chunk.metadata.get('page')}"
            )
            print(
                f"Text preview: {chunk.page_content[:200]}"
            )

            shown += 1

            if shown >= 5:
                break

    print("\nDiagnostic completed.")
    print("No embeddings or FAISS index were modified.")


if __name__ == "__main__":
    main()

