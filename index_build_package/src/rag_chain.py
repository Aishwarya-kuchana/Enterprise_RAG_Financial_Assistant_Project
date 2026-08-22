import re

from langchain_core.prompts import ChatPromptTemplate


class RAGChain:

    """
    Handles query understanding, retrieval,
    context creation and LLM response generation.
    """

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert Financial Intelligence Assistant.

Your task is to answer the user's question using ONLY
the financial report context provided below.

IMPORTANT RULES:

1. Carefully read ALL provided context before answering.
2. Use information from the context even when the wording
   is different from the wording used in the question.
3. Financial reports may use different terminology for
   the same concept. Use the terminology and values given
   in the report.
4. If the requested information is explicitly present
   in the context, answer it directly.
5. For comparison questions, identify the answer for EACH
   company separately before comparing them.
6. For year-specific questions, use the value belonging
   to the requested year.
7. Do not use outside knowledge.
8. Do not guess or invent values.
9. Only say:
   "I couldn't find this information in the available financial reports."
   when the requested information genuinely cannot be found
   in the provided context.

Give a clear, concise answer.

Include units such as million or billion when available.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
        )

        self.companies = [
            "alphabet",
            "amazon",
            "apple",
            "microsoft",
            "nvidia"
        ]

    def create_retrieval_query(self, question):

        """
        Expands common financial terminology to improve
        semantic retrieval while preserving the original
        user question.
        """

        query = question

        financial_expansions = {

            "revenue":
                "revenue revenues net sales total net sales",

            "revenues":
                "revenue revenues net sales total net sales",

            "net sales":
                "net sales total net sales revenue revenues",

            "profit":
                "profit net income earnings",

            "net income":
                "net income profit earnings",

            "earnings":
                "earnings net income profit",

            "employees":
                "employees workforce headcount",

            "employee":
                "employee employees workforce headcount",

            "headcount":
                "headcount employees workforce",

            "ai":
                "AI artificial intelligence",

            "artificial intelligence":
                "artificial intelligence AI",

            "investment":
                "investment investments capital expenditure",

            "investments":
                "investments investment capital expenditure",

            "debt":
                "debt borrowings liabilities",

            "cash":
                "cash cash equivalents",

            "expenses":
                "expenses costs operating expenses",

            "cost":
                "cost costs expenses",

            "research and development":
                "research and development R&D",

            "r&d":
                "R&D research and development"
        }

        question_lower = question.lower()

        expansions = []

        for term, expansion in financial_expansions.items():

            if term in question_lower:

                expansions.append(expansion)

        if expansions:

            query += " " + " ".join(expansions)

        return query

    def detect_companies(self, question):

        """
        Detect companies mentioned in the question.
        """

        question_lower = question.lower()

        company_mapping = {
            "alphabet": "Alphabet",
            "amazon": "Amazon",
            "apple": "Apple",
            "microsoft": "Microsoft",
            "nvidia": "NVIDIA"
        }

        detected_companies = []

        for name, metadata_name in company_mapping.items():

            if name in question_lower:

                detected_companies.append(
                    metadata_name
                )

        return detected_companies

    def detect_year(self, question):

        """
        Detect a four-digit year from the question.
        """

        match = re.search(
            r"\b(20\d{2})\b",
            question
        )

        if match:

            return int(match.group(1))

        return None

    def retrieve_documents(self, question):

        """
        Select the appropriate retrieval strategy.

        Multi-company:
            Retrieve separately for each company.

        Single-company:
            Retrieve using company/year metadata.

        General:
            Normal semantic retrieval.
        """

        retrieval_query = self.create_retrieval_query(
            question
        )

        companies = self.detect_companies(
            question
        )

        year = self.detect_year(
            question
        )

        # ---------------------------------------------
        # Multi-company question
        # ---------------------------------------------

        if len(companies) >= 2:

            documents = (
                self.retriever.retrieve_for_entities(
                    query=retrieval_query,
                    entities=companies,
                    year=year,
                    k_per_entity=10
                )
            )

        # ---------------------------------------------
        # Single-company question
        # ---------------------------------------------

        elif len(companies) == 1:

            documents = (
                self.retriever.retrieve(
                    query=retrieval_query,
                    k=20,
                    company=companies[0],
                    year=year
                )
            )

        # ---------------------------------------------
        # General question
        # ---------------------------------------------

        else:

            documents = (
                self.retriever.retrieve(
                    query=retrieval_query,
                    k=10
                )
            )

        return documents

    def create_context(self, documents):

        """
        Create structured context.

        Documents are grouped by company so that the LLM
        can clearly distinguish evidence belonging to
        different companies.
        """

        company_documents = {}

        for doc in documents:

            company = doc.metadata.get(
                "company",
                "Unknown"
            )

            if company not in company_documents:

                company_documents[company] = []

            company_documents[company].append(
                doc
            )

        context_parts = []

        for company, docs in company_documents.items():

            context_parts.append(
                f"\n========== {company.upper()} ==========\n"
            )

            for i, doc in enumerate(docs, 1):

                metadata = doc.metadata

                year = metadata.get(
                    "year",
                    "Unknown"
                )

                source = metadata.get(
                    "source",
                    "Unknown"
                )

                page = metadata.get(
                    "page",
                    "Unknown"
                )

                context_parts.append(
                    f"""
Document {i}
Company: {company}
Year: {year}
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
                )

        return "\n\n".join(
            context_parts
        )

    def ask(self, question):

        """
        Complete RAG pipeline:

        Question
            ↓
        Query understanding
            ↓
        Retrieval
            ↓
        Structured context
            ↓
        LLM
            ↓
        Answer + sources
        """

        documents = self.retrieve_documents(
            question
        )

        if not documents:

            return {
                "answer": (
                    "I couldn't find this information "
                    "in the available financial reports."
                ),
                "documents": []
            }

        context = self.create_context(
            documents
        )

        prompt = self.prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = self.llm.invoke(
            prompt
        )

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
            )
        return {
        "answer": answer,
        "documents": documents
        }