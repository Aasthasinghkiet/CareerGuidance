
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

KEYWORDS = [
    "indian-job-market",
    "career-jobs-sample",
    "job_dataset",
    "job_recommendation",
    "postings",
    "glassdoor",
    "simplyhired",
]

IMPORTANT_TERMS = [
    "title", "company", "employer", "location", "salary",
    "experience", "skill", "education", "degree",
    "description", "url", "apply", "date", "status",
    "country", "remote", "employment", "jobid",
]


def read_sample(path):
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path, nrows=3)

    if suffix == ".csv":
        return pd.read_csv(
            path,
            nrows=3,
            low_memory=False,
            encoding_errors="replace",
        )

    if suffix == ".zip":
        # pandas can read ZIP-compressed CSVs when the archive
        # contains a single CSV file.
        return pd.read_csv(
            path,
            nrows=3,
            low_memory=False,
            encoding_errors="replace",
            compression="zip",
        )

    return None


def main():
    files = [
        path for path in sorted(RAW_DIR.iterdir())
        if path.is_file()
        and any(word in path.name.lower() for word in KEYWORDS)
        and path.suffix.lower() in {".csv", ".xlsx", ".zip"}
    ]

    if not files:
        print("No matching job dataset files found.")
        return

    for path in files:
        print("\n" + "=" * 80)
        print("FILE:", path.name)

        try:
            frame = read_sample(path)

            if frame is None:
                continue

            print("\nCOLUMN NAMES:")
            for column in frame.columns:
                print("-", column)

            relevant = [
                column for column in frame.columns
                if any(term in str(column).lower() for term in IMPORTANT_TERMS)
            ]

            print("\nSAMPLE VALUES FROM RELEVANT COLUMNS:")
            if relevant:
                print(frame[relevant].head(2).to_string(index=False))
            else:
                print("No obvious job-related columns identified.")

        except Exception as error:
            print("Could not inspect this file:", error)

    print("\nInspection complete. No source files were modified.")


if __name__ == "__main__":
    main()
