
from app.rag_chat import build_components, hybrid_retrieve, choose_group, SOURCE_GROUPS

TEST_CASES = [
    {
        "question": "What training opportunities are available for women?",
        "relevant_sources": {
            "training_resources.csv",
            "women_support_and_livelihoods.csv",
            "career_pathways.csv",
        },
    },
    {
        "question": "How can I start a small business from home?",
        "relevant_sources": {
            "women_support_and_livelihoods.csv",
            "career_pathways.csv",
        },
    },
    {
        "question": "What government career opportunities are available?",
        "relevant_sources": {
            "government_career_paths.csv",
            "career_pathways.csv",
        },
    },
    {
        "question": "How can I restart my career after a break?",
        "relevant_sources": {
            "career_guidance_faqs.csv",
            "career_pathways.csv",
            "training_resources.csv",
        },
    },
]


def precision_at_k(documents, relevant_sources, k=4):
    top_documents = documents[:k]

    if not top_documents:
        return 0.0

    relevant_count = sum(
        doc.metadata.get("source_file") in relevant_sources
        for doc in top_documents
    )

    return relevant_count / k


def main():
    vectorstore, _, _ = build_components()

    vector_scores = []
    hybrid_scores = []

    for case in TEST_CASES:
        question = case["question"]
        relevant_sources = case["relevant_sources"]
        group = choose_group(question)
        allowed_sources = SOURCE_GROUPS[group]

        vector_docs = vectorstore.similarity_search(
            question,
            k=4,
            filter={"source_file": {"$in": allowed_sources}},
        )

        hybrid_docs = hybrid_retrieve(
            question=question,
            vectorstore=vectorstore,
            allowed_sources=allowed_sources,
            k=4,
            candidate_k=12,
        )

        vector_p = precision_at_k(vector_docs, relevant_sources)
        hybrid_p = precision_at_k(hybrid_docs, relevant_sources)

        vector_scores.append(vector_p)
        hybrid_scores.append(hybrid_p)

        print("\n" + "=" * 65)
        print(f"Question: {question}")
        print(f"Source group: {group}")
        print(f"Vector Precision@4: {vector_p:.2f}")
        print(f"Hybrid Precision@4: {hybrid_p:.2f}")

        print("\nHybrid top-4 sources:")
        for doc in hybrid_docs[:4]:
            print("-", doc.metadata.get("source_file", "Unknown"))

    print("\n" + "=" * 65)
    print("SUMMARY")

    print(
        "Mean Vector Precision@4:",
        f"{sum(vector_scores) / len(vector_scores):.3f}",
    )
    print(
        "Mean Hybrid Precision@4:",
        f"{sum(hybrid_scores) / len(hybrid_scores):.3f}",
    )


if __name__ == "__main__":
    main()
