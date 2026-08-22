from sentence_transformers import CrossEncoder


class Retriever:
    """
    Hybrid Retriever for Financial RAG.

    Retrieval strategy:
    1. Dense semantic retrieval using FAISS
    2. Exact keyword retrieval from stored chunks
    3. CrossEncoder reranking
    4. Duplicate removal
    """

    def __init__(self, vector_db):

        self.vector_db = vector_db

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def _get_all_documents(self):

        return list(
            self.vector_db.docstore._dict.values()
        )

    def _lexical_search(
        self,
        query,
        company=None,
        year=None
    ):

        documents = self._get_all_documents()

        if company is not None:

            documents = [
                doc
                for doc in documents
                if doc.metadata.get("company") == company
            ]

        if year is not None:

            documents = [
                doc
                for doc in documents
                if doc.metadata.get("year") == year
            ]

        query_lower = query.lower()

        stop_words = {
            "how", "many", "did", "does", "do",
            "what", "was", "were", "is", "are",
            "the", "a", "an", "of", "in", "for",
            "to", "and", "or", "with", "on", "by"
        }

        query_terms = [
            word.strip(".,?!()")
            for word in query_lower.split()
            if len(word.strip(".,?!()")) >= 4
            and word.strip(".,?!()") not in stop_words
        ]

        scored = []

        for doc in documents:

            content_lower = doc.page_content.lower()

            score = sum(
                1
                for term in query_terms
                if term in content_lower
            )

            if score > 0:
                scored.append((score, doc))

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            doc
            for score, doc in scored[:50]
        ]

    def _deduplicate(self, documents):

        unique_documents = []
        seen = set()

        for doc in documents:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content
            )

            if key not in seen:

                seen.add(key)
                unique_documents.append(doc)

        return unique_documents

    def retrieve(
        self,
        query,
        k=10,
        company=None,
        year=None,
        candidate_k=100
    ):

        filter_dict = {}

        if company is not None:
            filter_dict["company"] = company

        if year is not None:
            filter_dict["year"] = year

        # --------------------------------------------------
        # 1. Dense retrieval
        # --------------------------------------------------

        if filter_dict:

            dense_docs = self.vector_db.similarity_search(
                query,
                k=candidate_k,
                filter=filter_dict
            )

        else:

            dense_docs = self.vector_db.similarity_search(
                query,
                k=candidate_k
            )

        # --------------------------------------------------
        # 2. Lexical retrieval
        # --------------------------------------------------

        lexical_docs = self._lexical_search(
            query,
            company,
            year
        )

        # --------------------------------------------------
        # 3. Combine and deduplicate
        # --------------------------------------------------

        candidates = self._deduplicate(
            dense_docs + lexical_docs
        )

        if not candidates:
            return []

        # --------------------------------------------------
        # 4. CrossEncoder reranking
        # --------------------------------------------------

        pairs = [
            (query, doc.page_content)
            for doc in candidates
        ]

        scores = self.reranker.predict(pairs)

        ranked_documents = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )

        # --------------------------------------------------
        # 5. Final deduplication after ranking
        # --------------------------------------------------

        final_documents = []

        seen = set()

        for score, doc in ranked_documents:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content
            )

            if key not in seen:

                seen.add(key)
                final_documents.append(doc)

            if len(final_documents) == k:
                break

        return final_documents

    def retrieve_for_entities(
        self,
        query,
        entities,
        year=None,
        k_per_entity=10
    ):

        all_results = []

        for entity in entities:

            results = self.retrieve(
                query=query,
                k=k_per_entity,
                company=entity,
                year=year,
                candidate_k=100
            )

            all_results.extend(results)

        return self._deduplicate(all_results)