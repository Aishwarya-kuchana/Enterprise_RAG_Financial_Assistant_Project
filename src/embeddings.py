import torch

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Loads HuggingFace embedding model.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

    def load_model(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using embedding device: {device}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        return embedding_model