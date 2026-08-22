from sentence_transformers import CrossEncoder


class Retriever:
    """
    Retrieves relevant chunks from the Vector Database
    and reranks them using a cross-encoder.
    """

    def __init__(self, vector_db):

        self.vector_db = vector_db

        # Generic reranking model.
        # It is not specific to financial questions.
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def retrieve(
        self,
        query,
        k=10,
        company=None,
        year=None,
        candidate_k=100
    ):
        """
        Retrieve a larger candidate set using vector similarity,
        then rerank candidates using a cross-encoder.
        """

        filter_dict = {}

        if company is not None:
            filter_dict["company"] = company

        if year is not None:
            filter_dict["year"] = year

        # Step 1: retrieve broad candidate pool
        if filter_dict:

            candidates = self.vector_db.similarity_search(
                query,
                k=candidate_k,
                filter=filter_dict
            )

        else:

            candidates = self.vector_db.similarity_search(
                query,
                k=candidate_k
            )

        if not candidates:
            return []

        # Step 2: prepare query-document pairs
        pairs = [
            (query, doc.page_content)
            for doc in candidates
        ]

        # Step 3: cross-encoder reranking
        scores = self.reranker.predict(pairs)

        # Step 4: attach scores and sort
        ranked_documents = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )

        # Step 5: return only the best k documents
        return [
            doc
            for score, doc in ranked_documents[:k]
        ]

    def retrieve_for_entities(
        self,
        query,
        entities,
        year=None,
        k_per_entity=10
    ):
        """
        Retrieve and rerank documents separately
        for each company in a comparison question.
        """

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

        # Remove duplicate documents
        unique_results = []
        seen = set()

        for doc in all_results:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content
            )

            if key not in seen:

                seen.add(key)
                unique_results.append(doc)

        return unique_results