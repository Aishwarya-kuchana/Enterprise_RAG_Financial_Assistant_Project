import json
import re
from langchain_core.prompts import ChatPromptTemplate
class RAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.query_analysis_prompt = ChatPromptTemplate.from_template(
            """
You are a financial query-understanding system.
Analyze the user's financial question BEFORE document retrieval.
Your job is NOT to answer the question.
Extract:
1. intent
   - factual
   - comparison
   - trend
   - calculation
   - other
2. companies
   - Identify ALL company names mentioned in the question.
   - Companies may belong to the permanent knowledge base OR
     may come from a user-uploaded financial document.
   - Do NOT restrict companies to a predefined list.
   - Return the company names exactly as they appear in the
     question, using normal company names.
3. year
   - Extract the four-digit year if explicitly mentioned.
   - If no year is mentioned, return null.
4. financial_metric
   - Identify the financial concept the user is asking about.
   - Examples:
     revenue
     net income
     operating income
     gross profit
     employees
     total assets
     liabilities
     shareholders equity
     cash
     debt
     expenses
     etc.
5. ambiguity
   - true if the financial term is ambiguous or could refer
     to multiple different financial concepts.
   - false if the requested metric is clear.
6. clarification
   - If ambiguity is true, briefly explain what needs
     clarification.
   - Otherwise return null.
7. retrieval_query
   - Create a concise retrieval-oriented query containing:
     companies, year, financial metric and useful financial
     terminology.
   - The query should help retrieve relevant sections from
     annual reports.
   - Do not invent facts.
IMPORTANT FINANCIAL LANGUAGE RULES:

- Understand the meaning of the question, not just individual
  keywords.

- When the user asks about company financial performance,
  "sales" normally refers to revenue or net sales unless the
  question explicitly refers to units sold or products sold.

- "generated sales", "total sales", "sales revenue", and
  "net sales" can be treated as revenue/net sales for financial
  report retrieval.

- "made money" may refer to revenue or profit depending on
  context. If the intended metric cannot be determined,
  mark it ambiguous.

- "workforce", "headcount", and "number of employees" refer
  to employee count when the context is company human capital.

- "net worth" is NOT automatically the same as revenue,
  net income, shareholders' equity, or market capitalization.

- "net value" is NOT automatically the same as another
  financial metric.

- Do not invent a definition for an ambiguous financial term.

- Do not reject a company merely because it is not part of
  the permanent knowledge base.

- A company may come from a user-uploaded document.

Return ONLY valid JSON.

Required JSON format:

{{
    "intent": "...",
    "companies": [],
    "year": null,
    "financial_metric": "...",
    "ambiguous": false,
    "clarification": null,
    "retrieval_query": "..."
}}

USER QUESTION:
{question}
"""
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert Financial Intelligence Assistant.

Answer the user's question using ONLY the financial report
context provided below.

IMPORTANT RULES:

1. Carefully read ALL provided context before answering.

2. Use information from the context even when the wording
   is different from the wording used in the question.

3. Financial reports may use different terminology for the
   same concept. Use the terminology and values actually
   provided in the report.

4. If the requested information is explicitly present in
   the context, answer it directly.

5. For comparison questions, identify the answer for EACH
   company separately before comparing them.

6. For year-specific questions, use the value belonging to
   the requested year.

7. Do not use outside knowledge.

8. Do not guess or invent values.

9. If the user's question uses an ambiguous financial term,
   do not silently substitute another metric.

10. Only say:

"I couldn't find this information in the available financial reports."

when the requested information genuinely cannot be found
in the provided context.

11. Write the answer as a complete, grammatically correct sentence. Never return only a number or currency value.

12. When the context supports it, explicitly identify the company, financial metric, reporting year, and value.

13. Preserve the units used in the financial report. If the report states that figures are in millions, make this clear in the answer.

14. You may convert millions to billions when useful, but ensure the conversion is mathematically correct.

15. Keep the answer concise and do not repeat unnecessary information.

16. Do not invent missing details. If the context does not support identifying the metric, company, reporting year, or value, state only what can be verified from the context.


CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
        )
    def _extract_text(self, response):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get(
                        "text",
                        ""
                    )
                    if text:
                        parts.append(text)
            return "".join(parts)
        return str(content)
    
    def analyze_question(self, question):
        prompt = self.query_analysis_prompt.invoke(
            {
                "question": question
            }
        )
        response = self.llm.invoke(
            prompt
        )
        text = self._extract_text(
            response
        ).strip()
        
        text = re.sub(
            r"```json\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"```\s*$",
            "",
            text
        )
        try:
            analysis = json.loads(
                text
            )
        except json.JSONDecodeError:
            analysis = {
                "intent": "factual",
                "companies": [],
                "year": None,
                "financial_metric": "",
                "ambiguous": False,
                "clarification": None,
                "retrieval_query": question
            }
        return analysis
    
    def detect_companies(self, question):
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
        match = re.search(
            r"\b(20\d{2})\b",
            question
        )
        if match:
            return int(
                match.group(1)
            )
        return None
    
    def create_retrieval_query(self, question):
        """
        Uses LLM-based query understanding to create
        a retrieval-oriented query.
        """
        analysis = self.analyze_question(
            question
        )
        retrieval_query = analysis.get(
            "retrieval_query"
        )
        if not retrieval_query:
            return question
        return retrieval_query

    def retrieve_documents(self, question):
        """
        Complete query understanding + retrieval process.
        """

        analysis = self.analyze_question(
            question
        )
        if analysis.get(
            "ambiguous",
            False
        ):
            return {
                "documents": [],
                "analysis": analysis
            }
        retrieval_query = analysis.get(
            "retrieval_query",
            question
        )
        if not retrieval_query:
            retrieval_query = question
        companies = analysis.get(
            "companies",
            []
        )
        if not companies:
            companies = self.detect_companies(
                question
            )
        year = analysis.get(
            "year"
        )
        if year is None:

            year = self.detect_year(
                question
            )
        
        if len(companies) >= 2:
            documents = (
                self.retriever.retrieve_for_entities(
                    query=retrieval_query,
                    entities=companies,
                    year=year,
                    k_per_entity=10
                )
            )
        
        elif len(companies) == 1:
            documents = (
                self.retriever.retrieve(
                    query=retrieval_query,
                    k=10,
                    company=companies[0],
                    year=year,
                    candidate_k=100
                )
            )
        
        else:
            documents = (
                self.retriever.retrieve(
                    query=retrieval_query,
                    k=10,
                    candidate_k=100
                )
            )
        return {
            "documents": documents,
            "analysis": analysis
        }
    
    def create_context(self, documents):
        company_documents = {}
        for doc in documents:
            company = doc.metadata.get(
                "company",
                "Uploaded Document"
            )
            if company not in company_documents:
                company_documents[company] = []
            company_documents[
                company
            ].append(doc)
        context_parts = []
        for company, docs in (
            company_documents.items()
        ):
            context_parts.append(
                f"\n========== {company.upper()} ==========\n"
            )
            for i, doc in enumerate(
                docs,
                1
            ):
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
        retrieval_result = (
            self.retrieve_documents(
                question
            )
        )
        documents = retrieval_result[
            "documents"
        ]
        analysis = retrieval_result[
            "analysis"
        ]
        
        if analysis.get(
            "ambiguous",
            False
        ):
            clarification = analysis.get(
                "clarification"
            )
            if not clarification:
                clarification = (
                    "Could you clarify which financial "
                    "metric you mean?"
                )
            return {
                "answer": clarification,
                "documents": [],
                "analysis": analysis
            }
        
        if not documents:
            return {
                "answer": (
                    "I couldn't find this information "
                    "in the available financial reports."
                ),
                "documents": [],
                "analysis": analysis
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
        answer = self._extract_text(
            response
        )
        return {
            "answer": answer,
            "documents": documents,
            "analysis": analysis
        }