import os
import re
import torch
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.chunking import ChunkCreator
from src.llm import LLMModel
from src.rag_chain import RAGChain
st.set_page_config(
    page_title="Enterprise Financial RAG Assistant",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Enterprise Financial RAG Assistant")
st.markdown(
    "Ask questions about company annual reports "
    "or upload your own financial PDF reports."
)


load_dotenv()
@st.cache_resource
def load_embedding_model():
    return EmbeddingModel().load_model()
@st.cache_resource
def load_base_retriever(embedding_model):
    vector_store = VectorStore()
    vector_db = vector_store.load_vector_db(
        embedding_model
    )
    return Retriever(vector_db)
@st.cache_resource
def load_llm():
    return LLMModel().load_llm()
with st.spinner("Loading financial knowledge base..."):
    embedding_model = load_embedding_model()
    base_retriever = load_base_retriever(
        embedding_model
    )
with st.spinner("Loading Gemini..."):
    llm = load_llm()
rag = RAGChain(
    base_retriever,
    llm
)
st.success("System Ready!")
if "uploaded_retrievers" not in st.session_state:
    st.session_state.uploaded_retrievers = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
COMPANY_MAPPING = {

    "tesla": "Tesla",
    "nvidia": "NVIDIA",
    "amazon": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "apple": "Apple",
    "microsoft": "Microsoft",
    "meta": "Meta",
    "facebook": "Meta",
    "ford": "Ford",
    "walmart": "Walmart",
    "intel": "Intel",
    "netflix": "Netflix"

}
def detect_company_from_filename(filename):

    filename_lower = filename.lower()

    for key, company in COMPANY_MAPPING.items():

        if key in filename_lower:
            return company

    return "Unknown"


def detect_year_from_filename(filename):
    match = re.search(
        r"\b(20\d{2})\b",
        filename
    )
    if match:
        return int(match.group(1))
    return None
st.subheader("📄 Upload Financial Reports")
st.markdown(
    "Upload one or multiple annual reports. "
    "Uploaded reports can be queried together with "
    "the permanent financial knowledge base."
)
uploaded_files = st.file_uploader(
    "Upload PDF reports",
    type=["pdf"],
    accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_id = (
            file_name,
            uploaded_file.size
        )
        
        if file_id not in st.session_state.processed_files:
            with st.spinner(
                f"Processing {file_name}..."
            ):
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:
                        temp_file.write(
                            uploaded_file.getbuffer()
                        )
                        temp_file_path = temp_file.name                   
                    detected_company = (
                        detect_company_from_filename(
                            file_name
                        )
                    )
                    detected_year = (
                        detect_year_from_filename(
                            file_name
                        )
                    )
                    loader = PyPDFLoader(
                        temp_file_path
                    )
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata["source"] = (
                            file_name
                        )
                        doc.metadata["document_type"] = (
                            "user_upload"
                        )
                        doc.metadata["company"] = (
                            detected_company
                        )
                        doc.metadata["year"] = (
                            detected_year
                        )
                    chunk_creator = ChunkCreator()
                    chunks = (
                        chunk_creator.create_chunks(
                            documents
                        )
                    )
                    vector_store = VectorStore()
                    vector_db = (
                        vector_store.create_vector_db(
                            chunks,
                            embedding_model
                        )
                    )
                    uploaded_retriever = Retriever(
                        vector_db
                    )
                    st.session_state.uploaded_retrievers[
                        file_name
                    ] = uploaded_retriever
                    st.session_state.processed_files[
                        file_id
                    ] = {
                        "chunks": len(chunks),
                        "file_name": file_name,
                        "company": detected_company,
                        "year": detected_year
                    }
                finally:
                    if (
                        temp_file_path
                        and os.path.exists(temp_file_path)
                    ):
                        os.remove(
                            temp_file_path
                        )

if st.session_state.uploaded_retrievers:
    st.subheader("📚 Uploaded Reports")
    for file_name in (
        st.session_state.uploaded_retrievers.keys()
    ):
        file_info = None

        for file_id, info in (
            st.session_state.processed_files.items()
        ):
            if info["file_name"] == file_name:
                file_info = info
                break
        if file_info:
            company = file_info.get(
                "company",
                "Unknown"
            )
            year = file_info.get(
                "year",
                "Unknown"
            )
            st.write(
                f"✓ {file_name} "
                f"— {company}, {year}"
            )
        else:
            st.write(
                f"✓ {file_name}"
            )
st.subheader("💬 Ask a Question")
question = st.text_input(
    "Ask your financial question",
    placeholder=(
        "Example: Compare Apple and Tesla revenue in 2025"
    )
)
if st.button("Get Answer"):
    if not question.strip():
        st.warning(
            "Please enter a question."
        )
        st.stop()
    with st.spinner(
        "Understanding your question..."
    ):
        analysis = rag.analyze_question(
            question
        )
    if analysis.get(
        "ambiguous",
        False
    ):
        st.subheader(
            "Clarification Needed"
        )
        st.warning(
            analysis.get(
                "clarification",
                "Please clarify the financial metric."
            )
        )
        with st.expander(
            "Query Analysis"
        ):
            st.json(
                analysis
            )
        st.stop()
    analyzed_companies = (
        analysis.get(
            "companies",
            []
        )
    )
    companies = []
    for company in analyzed_companies:
        company_lower = company.lower()
        if company_lower in COMPANY_MAPPING:
            normalized = COMPANY_MAPPING[
                company_lower
            ]
        else:
            normalized = company
        if normalized not in companies:
            companies.append(
                normalized
            )
    year = analysis.get(
        "year"
    )
    retrieval_query = analysis.get(
        "retrieval_query",
        question
    )
    with st.spinner(
        "Retrieving relevant financial information..."
    ):

        all_documents = []
        if len(companies) >= 2:

            base_documents = (
                base_retriever.retrieve_for_entities(
                    query=retrieval_query,
                    entities=companies,
                    year=year,
                    k_per_entity=5
                )
            )
        elif len(companies) == 1:
            company = companies[0]
            permanent_companies = {
                "Amazon",
                "Alphabet",
                "Apple",
                "Microsoft",
                "NVIDIA"
            }
            if company in permanent_companies:
                base_documents = (
                    base_retriever.retrieve(
                        query=retrieval_query,
                        k=5,
                        company=company,
                        year=year,
                        candidate_k=50
                    )
                )
            else:
                base_documents = []
        else:
            base_documents = (
                base_retriever.retrieve(
                    query=retrieval_query,
                    k=5,
                    candidate_k=50
                )
            )

        all_documents.extend(
            base_documents
        )
        
        for (
            file_name,
            uploaded_retriever
        ) in (
            st.session_state.uploaded_retrievers.items()
        ):
            uploaded_documents = (
                uploaded_retriever.retrieve(
                    query=retrieval_query,
                    k=5,
                    candidate_k=50
                )
            )
            all_documents.extend(
                uploaded_documents
            )
        unique_documents = []
        seen = set()
        for doc in all_documents:
            key = (
                doc.metadata.get(
                    "source"
                ),
                doc.metadata.get(
                    "page"
                ),
                doc.page_content
            )
            if key not in seen:
                seen.add(key)
                unique_documents.append(
                    doc
                )
        all_documents = (
            unique_documents
        )
    if not all_documents:
        st.subheader(
            "Answer"
        )
        st.write(
            "I couldn't find this information "
            "in the available financial reports."
        )
        with st.expander(
            "Query Analysis"
        ):
            st.json(
                analysis
            )
        st.stop()

    with st.spinner(
        "Generating answer..."
    ):

        context = rag.create_context(
            all_documents
        )
        prompt = rag.prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )
        response = llm.invoke(
            prompt
        )
        answer = rag._extract_text(
            response
        )
    st.subheader(
        "Answer"
    )
    st.write(
        answer
    )
    with st.expander(
        "Query Analysis"
    ):

        st.json(
            analysis
        )
    st.subheader(
        "Retrieved Sources"
    )
    for i, doc in enumerate(
        all_documents[:10],
        start=1
    ):
        with st.expander(
            f"Source {i}"
        ):
            st.write(
                doc.metadata
            )
            st.write(
                doc.page_content[:700]
                + "..."
            )