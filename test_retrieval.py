
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "arogyini_career_knowledge"


def main():
    if not CHROMA_DIR.exists():
        print("ERROR: Chroma database folder was not found.")
        print("Run python scripts\\ingest_data.py first.")
        return

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    print(f"Stored chunks: {vectorstore._collection.count()}")

    questions = [
        "What training opportunities are available for women?",
        "How can I start a small business from home?",
        "What government career opportunities are available?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")

        if "government" in question.lower():
           allowed_sources = [
        "government_career_paths.csv",
        "career_guidance_faqs.csv",
        "career_pathways.csv",
            ]
        elif "business" in question.lower() or      "home" in question.lower():
          allowed_sources = [
        "women_support_and_livelihoods.csv",
        "career_pathways.csv",
        "training_resources.csv",
        "career_guidance_faqs.csv",
       ]
        else:
          allowed_sources = [
        "training_resources.csv",
        "women_support_and_livelihoods.csv",
        "career_pathways.csv",
        "career_guidance_faqs.csv",
        ]

        results = vectorstore.similarity_search_with_score(
         question,
         k=4,
         filter={"source_file": {"$in": allowed_sources}},
      )

        for rank, (document, distance) in enumerate(results, start=1):
            print(f"\nResult {rank} | Distance: {distance:.4f}")
            print(f"Source: {document.metadata.get('source_file')}")
            print(f"Chunk: {document.metadata.get('chunk_index')}")
            print(document.page_content[:700])


if __name__ == "__main__":
    main()
