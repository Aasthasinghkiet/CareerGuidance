
from pathlib import Path
import hashlib

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "arogyini_career_knowledge"

# Start with the most useful guidance files.
PRIORITY_FILES = [
    "career_pathways.csv",
    "career_guidance_faqs.csv",
    "training_resources.csv",
    "government_career_paths.csv",
    "international_opportunity_types.csv",
    "women_support_and_livelihoods.csv",
]

# Large job datasets are limited during the first test.
MAX_ROWS_PER_FILE = 1000


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    if not RAW_DIR.exists():
        print(f"Raw data folder not found: {RAW_DIR}")
        return

    files = []
    for filename in PRIORITY_FILES:
        path = RAW_DIR / filename
        if path.exists():
            files.append(path)

    # Include other CSV files, but avoid processing a file twice.
    for path in sorted(RAW_DIR.glob("*.csv")):
        if path not in files:
            files.append(path)

    if not files:
        print("No CSV files found in data/raw.")
        return

    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Use a fresh collection so reruns do not leave stale records.
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    vectorstore.delete_collection()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    seen_ids = set()
    total_added = 0

    for path in files:
        print(f"\nReading: {path.name}")

        try:
            frame = pd.read_csv(
                path,
                nrows=MAX_ROWS_PER_FILE,
                low_memory=False,
                encoding_errors="replace",
            )
        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            continue

        documents = []

        for _, row in frame.iterrows():
            parts = [f"Source file: {path.name}"]

            for column in frame.columns:
                value = clean_value(row[column])
                if value and value.lower() not in {"nan", "none", "null"}:
                    parts.append(f"{column}: {value}")

            content = "\n".join(parts)
            record_id = hashlib.sha256(
                f"{path.name}|{content}".encode("utf-8")
            ).hexdigest()

            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_file": path.name,
                        "record_type": "career_knowledge",
                    },
                    id=record_id,
                )
            )

        # Add documents in batches to control memory use.
        for start in range(0, len(documents), 64):
            batch = documents[start:start + 64]
            vectorstore.add_documents(
                documents=batch,
                ids=[
                    hashlib.sha256(
                        doc.page_content.encode("utf-8")
                    ).hexdigest()
                    for doc in batch
                ],
            )

        total_added += len(documents)
        print(f"Prepared {len(documents)} records")

    print("\nRAG database build finished.")
    print(f"Total records added: {total_added}")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
