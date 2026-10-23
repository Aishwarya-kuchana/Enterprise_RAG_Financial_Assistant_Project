import os
from langchain_google_genai import ChatGoogleGenerativeAI
class LLMModel:
    def __init__(
        self,
        model_name=None
    ):
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )
    def load_llm(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not found."
            )
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.2
        )
        return llm