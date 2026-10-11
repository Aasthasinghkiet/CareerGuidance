
from pathlib import Path
import hashlib

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "arogyini_career_knowledge"

PRIORITY_FILES = [
    "career_pathways.csv",
    "career_guidance_faqs.csv",
    "training_resources.csv",
    "government_career_paths.csv",
    "international_opportunity_types.csv",
    "women_support_and_livelihoods.csv",
]

MAX_ROWS_PER_FILE = 1000
BATCH_SIZE = 64
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def create_record_id(source_file, content):
    text = f"{source_file}|{content}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if not RAW_DIR.exists():
        print(f"Raw data folder not found: {RAW_DIR}")
        return

    files = []

    for filename in PRIORITY_FILES:
        path = RAW_DIR / filename
        if path.exists():
            files.append(path)

    for path in sorted(RAW_DIR.glob("*.csv")):
        if path not in files:
            files.append(path)

    if not files:
        print("No CSV files found in data/raw.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", " ", ""],
        add_start_index=True,
    )

    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Rebuild the collection from the selected source data.
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
    total_records = 0
    total_chunks = 0

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

        record_documents = []

        for _, row in frame.iterrows():
            parts = [f"Source file: {path.name}"]

            for column in frame.columns:
                value = clean_value(row[column])

                if value and value.lower() not in {
                    "nan", "none", "null"
                }:
                    parts.append(f"{column}: {value}")

            content = "\n".join(parts)

            if not content.strip():
                continue

            record_id = create_record_id(path.name, content)

            if record_id in seen_ids:
                continue

            seen_ids.add(record_id)

            record_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_file": path.name,
                        "record_type": "career_knowledge",
                        "record_id": record_id,
                    },
                )
            )

        chunks = []

        for document in record_documents:
            record_chunks = splitter.split_documents([document])

            for chunk_index, chunk in enumerate(record_chunks):
                chunk.metadata["chunk_index"] = chunk_index

                chunk_id = hashlib.sha256(
                    (
                        f"{chunk.metadata['record_id']}|"
                        f"{chunk_index}|{chunk.page_content}"
                    ).encode("utf-8")
                ).hexdigest()

                chunk.metadata["chunk_id"] = chunk_id
                chunks.append((chunk, chunk_id))

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            vectorstore.add_documents(
                documents=[item[0] for item in batch],
                ids=[item[1] for item in batch],
            )

        total_records += len(record_documents)
        total_chunks += len(chunks)

        print(f"Records prepared: {len(record_documents)}")
        print(f"Chunks created: {len(chunks)}")

    print("\nRAG database build finished.")
    print(f"Total records: {total_records}")
    print(f"Total chunks: {total_chunks}")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
