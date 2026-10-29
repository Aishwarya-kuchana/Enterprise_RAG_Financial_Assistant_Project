# 💰 Enterprise Financial RAG Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** application designed to answer financial questions from company annual reports. The system retrieves relevant information from financial documents and uses a Large Language Model (LLM) to generate context-aware responses.

🔗 **Live Application:** [http://localhost:8501/](YOUR_STREAMLIT_LINK_HERE)

---

## 📌 Project Overview

Enterprise Financial RAG Assistant enables users to ask natural language questions about company financial reports.

The application processes annual reports from companies and builds a semantic knowledge base using embeddings and a FAISS vector database. When a user asks a question, the system retrieves relevant document sections, reranks them based on relevance, and provides the context to an LLM to generate an accurate response.

The system can answer questions such as:

- What was Apple's revenue in 2023?
- Compare Amazon and Alphabet revenue in 2025.
- How large was NVIDIA's workforce in 2024?
- Which company generated the highest revenue?
- Compare financial metrics across different companies and years.

Users can also upload additional financial reports dynamically and ask questions based on the uploaded documents.

---

## 🚀 Live Demo

🔗 **Try the application here:**  
[Enterprise Financial RAG Assistant]:https://enterpriseragfinancialassistantproject-pjbc3ufxkawximnvsfi4ie.streamlit.app/

---

## ✨ Key Features

- 📄 Question answering over company annual reports
- 🔍 Semantic document retrieval using vector embeddings
- 🧠 Query analysis for identifying companies, year, financial metric, and intent
- 📊 Support for financial comparison questions
- 🔄 Cross-encoder reranking for improved retrieval relevance
- 🤖 Context-aware answer generation using Gemini LLM
- 📑 Dynamic PDF upload functionality
- 🗂️ Support for querying both existing knowledge base and uploaded documents
- 🚫 Reduced hallucination through retrieval-based context grounding
- 💻 Interactive user interface built with Streamlit

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   Financial PDFs │
                    │  Annual Reports  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Text Extraction  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Document Chunking│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ HuggingFace      │
                    │ Embeddings       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FAISS Vector DB  │
                    └──────────────────┘


        User Question
              │
              ▼
     ┌──────────────────┐
     │ Query Analysis   │
     │                  │
     │ • Company        │
     │ • Year           │
     │ • Metric         │
     │ • Intent         │
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │ Vector Retrieval │
     │      FAISS       │
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │    Reranking     │
     │ Cross-Encoder    │
     └────────┬─────────┘
              │
              ▼
     ┌──────────────────┐
     │   Gemini LLM     │
     └────────┬─────────┘
              │
              ▼
          Final Answer
