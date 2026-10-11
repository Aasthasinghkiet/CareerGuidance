
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "arogyini_career_knowledge"

TESTS = [
    {
        "question": "What careers can a woman start without technical skills?",
        "sources": [
            "career_guidance_faqs.csv",
            "career_pathways.csv",
            "training_resources.csv",
            "women_support_and_livelihoods.csv",
        ],
    },
    {
        "question": "What training can help someone return to work after a career break?",
        "sources": [
            "career_guidance_faqs.csv",
            "career_pathways.csv",
            "training_resources.csv",
        ],
    },
    {
        "question": "What are some pathways to government jobs in India?",
        "sources": [
            "government_career_paths.csv",
            "career_guidance_faqs.csv",
        ],
    },
]


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    for test in TESTS:
        print("\n" + "=" * 70)
        print("QUESTION:", test["question"])
        print("=" * 70)

        results = vectorstore.similarity_search_with_score(
            test["question"],
            k=3,
            filter={"source_file": {"$in": test["sources"]}},
        )

        for rank, (document, score) in enumerate(results, start=1):
            print(f"\nResult {rank} | Distance: {score:.4f}")
            print(document.page_content[:1000])
            print("Source:", document.metadata.get("source_file"))


if __name__ == "__main__":
    main()
