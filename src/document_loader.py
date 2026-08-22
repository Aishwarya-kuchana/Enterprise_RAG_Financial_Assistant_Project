from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


class DocumentLoader:

    def __init__(self):

        self.data_dir = Path("data/raw")

    def load_documents(self):

        documents = []

        pdf_files = list(self.data_dir.rglob("*.pdf"))

        if not pdf_files:

            raise FileNotFoundError(
                f"No PDF files found in {self.data_dir}"
            )

        for pdf_file in pdf_files:

            loader = PyPDFLoader(str(pdf_file))

            docs = loader.load()

            # ------------------------------------
            # Extract Company & Year
            # ------------------------------------

            filename = pdf_file.stem

            parts = filename.split("_")

            company = parts[0]
            year = int(parts[1])

            # ------------------------------------
            # Add Metadata
            # ------------------------------------

            for doc in docs:
                page = doc.metadata.get("page", 0)
                doc.metadata = {
                        "company": company,
                        "year": year,
                        "source": pdf_file.name,
                        "page": page
                }
                
            documents.extend(docs)

        return documents