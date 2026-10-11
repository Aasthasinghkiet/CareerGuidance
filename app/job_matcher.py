
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "indian-job-market-dataset-2025 (1).xlsx"
)

DATASET_NAME = "Indian job-market dataset (2025)"


def load_jobs():
    """Load and clean job records without modifying the source file."""
    df = pd.read_excel(DATA_PATH)

    required_columns = [
        "title",
        "companyName",
        "location",
        "tagsAndSkills",
        "minimumExperience",
        "minimumSalary",
        "maximumSalary",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    for column in ["title", "companyName", "location", "tagsAndSkills"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[df["title"] != ""].copy()

    df = df.drop_duplicates(
        subset=["title", "companyName", "location"]
    )

    return df


def classify_job(title, skills=""):
    """Classify a role using job-title and skill keywords."""
    text = f"{title} {skills}".lower()

    technical_keywords = [
        "software", "developer", "programmer", "data analyst",
        "data scientist", "cybersecurity", "network engineer",
        "cloud engineer", "devops", "site reliability",
        "it support", "technical support", "qa engineer",
        "automation tester", "database administrator",
        "machine learning", "artificial intelligence",
        "electrical engineer", "mechanical engineer",
        "maintenance technician", "technical lab",
        "system administrator", "web developer",
    ]

    non_technical_keywords = [
        "accountant", "accounting", "human resources",
        "hr recruiter", "recruiter", "marketing", "sales",
        "customer service", "customer support", "administration",
        "office assistant", "teacher", "content writer",
        "receptionist", "finance", "payroll", "data entry",
        "operations executive", "business development",
        "medical billing", "talent acquisition",
    ]

    # Check specific technical terms first.
    if any(keyword in text for keyword in technical_keywords):
        return "Technical"

    if any(keyword in text for keyword in non_technical_keywords):
        return "Non-technical"

    return "Other / Review needed"


def education_keywords(education):
    """
    Return education-related terms for a soft relevance signal.
    These terms do not establish formal eligibility.
    """
    value = education.lower().strip()

    aliases = {
        "no formal education": ["no formal education"],
        "class 10": ["class 10", "10th", "secondary school"],
        "class 12": ["class 12", "12th", "higher secondary"],
        "iti": ["iti", "industrial training institute"],
        "diploma": ["diploma"],
        "b.com": ["b.com", "bachelor of commerce", "bcom"],
        "b.a.": ["b.a.", "bachelor of arts", "ba degree"],
        "b.sc.": ["b.sc.", "bachelor of science", "bsc"],
        "bba": ["bba", "bachelor of business administration"],
        "bca": ["bca", "bachelor of computer applications"],
        "b.tech": ["b.tech", "bachelor of technology", "btech"],
        "m.com": ["m.com", "master of commerce", "mcom"],
        "m.a.": ["m.a.", "master of arts"],
        "m.sc.": ["m.sc.", "master of science"],
        "mba": ["mba", "master of business administration"],
        "mca": ["mca", "master of computer applications"],
        "m.tech": ["m.tech", "master of technology"],
        "postgraduate": ["postgraduate", "post graduate", "master's degree"],
        "graduate": ["graduate", "bachelor's degree", "bachelor degree"],
    }

    # Prefer the user's selected label and known aliases.
    if value in aliases:
        return aliases[value]

    # Also allow common labels such as "B.Com graduate".
    for key, terms in aliases.items():
        if key in value:
            return terms

    return [value] if value else []


def recommend_jobs(
    education="",
    skills="",
    experience_years=0,
    location="",
    job_type="Any",
    limit=10,
):
    """
    Rank job records against a candidate profile.

    Education and experience are relevance signals, not proof of eligibility.
    All returned records are historical unless independently verified.
    """
    df = load_jobs()

    if not isinstance(experience_years, (int, float)):
        raise ValueError("experience_years must be a number.")

    if experience_years < 0:
        raise ValueError("experience_years cannot be negative.")

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    limit = max(1, min(limit, 50))

    valid_types = ["any", "technical", "non-technical"]

    if job_type.lower().strip() not in valid_types:
        raise ValueError(
            "job_type must be Any, Technical, or Non-technical."
        )

    # Normalize fields that may be missing in the source dataset.
    df["_title"] = df["title"].fillna("").astype(str)
    df["_company"] = df["companyName"].fillna("").astype(str)
    df["_location"] = df["location"].fillna("").astype(str)
    df["_skills"] = df["tagsAndSkills"].fillna("").astype(str)

    if "jobDescription" in df.columns:
        df["_description"] = (
            df["jobDescription"].fillna("").astype(str)
        )
    else:
        df["_description"] = ""

    df["_category"] = df.apply(
        lambda row: classify_job(row["_title"], row["_skills"]),
        axis=1,
    )

    requested_type = job_type.lower().strip()

    if requested_type in ["technical", "non-technical"]:
        df = df[
            df["_category"].str.lower() == requested_type
        ].copy()

    if df.empty:
        return []

    # Accept skills separated by commas or semicolons.
    if isinstance(skills, str):
        candidate_skills = [
            item.strip().lower()
            for item in skills.replace(",", ";").split(";")
            if item.strip()
        ]
    else:
        candidate_skills = []

    candidate_education_terms = education_keywords(education)

    requested_location = location.strip().lower()

    def score_job(row):
        title = row["_title"].lower()
        job_skills = row["_skills"].lower()
        description = row["_description"].lower()
        combined_text = f"{title} {job_skills} {description}"

        score = 0
        matched_skills = []

        # Skills contribute most to the match score.
        for skill in candidate_skills:
            if skill in job_skills:
                score += 3
                matched_skills.append(skill)
            elif skill in title:
                score += 2
                matched_skills.append(skill)
            elif skill in description:
                score += 1
                matched_skills.append(skill)

        # Experience: reward compatible requirements and penalize
        # roles asking for substantially more experience.
        minimum_exp = pd.to_numeric(
            pd.Series([row.get("minimumExperience")]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(minimum_exp):
            if minimum_exp <= experience_years:
                score += 2
            else:
                score -= min(
                    10,
                    int(minimum_exp - experience_years),
                )

        # Education is only a weak text-based signal. This dataset does
        # not provide a dedicated, validated education requirement field.
        education_match = any(
            term in combined_text
            for term in candidate_education_terms
            if term
        )

        if education_match:
            score += 1

        # Location preference adds a ranking bonus.
        location_match = bool(
            requested_location
            and requested_location in row["_location"].lower()
        )

        if location_match:
            score += 4

        return pd.Series(
            [score, ", ".join(matched_skills), location_match]
        )

    df[["_score", "_matched_skills", "_location_match"]] = df.apply(
        score_job,
        axis=1,
    )

    # When the requested location has matches, prefer those records.
    # Otherwise, keep results from other locations rather than returning
    # no results at all.
    if requested_location and df["_location_match"].any():
        df = df[df["_location_match"]].copy()

    df = df.sort_values(
        by=["_score", "_title"],
        ascending=[False, True],
    )

    results = []

    for _, row in df.head(limit).iterrows():
        minimum_salary = pd.to_numeric(
            pd.Series([row.get("minimumSalary")]),
            errors="coerce",
        ).iloc[0]

        maximum_salary = pd.to_numeric(
            pd.Series([row.get("maximumSalary")]),
            errors="coerce",
        ).iloc[0]

        salary = "Not available"

        # The dataset's example salary values appear to be annual INR.
        # Treat zero or invalid values as missing.
        if (
            pd.notna(minimum_salary)
            and pd.notna(maximum_salary)
            and minimum_salary >= 0
            and maximum_salary >= 0
            and (minimum_salary > 0 or maximum_salary > 0)
        ):
            salary = (
                f"₹{minimum_salary:,.0f}–"
                f"₹{maximum_salary:,.0f} per year"
            )

        minimum_exp = pd.to_numeric(
            pd.Series([row.get("minimumExperience")]),
            errors="coerce",
        ).iloc[0]

        results.append(
            {
                "title": row["_title"],
                "company": (
                    row["_company"]
                    if row["_company"]
                    else "Company not specified"
                ),
                "location": row["_location"] or "Not specified",
                "category": row["_category"],
                "skills": row["_skills"] or "Skills not specified",
                "matched_skills": (
                    row["_matched_skills"] or "No direct skill match found"
                ),
                "minimum_experience": (
                    float(minimum_exp)
                    if pd.notna(minimum_exp)
                    else "Not specified"
                ),
                "salary": salary,
                "match_score": int(row["_score"]),
                "source": DATASET_NAME,
                "status": (
                    "Historical dataset record; "
                    "current vacancy and eligibility unverified"
                ),
            }
        )

    return results


if __name__ == "__main__":
    jobs = recommend_jobs(
        education="B.Com",
        skills="Tally; GST; Excel",
        experience_years=1,
        location="Noida",
        job_type="Non-technical",
        limit=5,
    )

    if not jobs:
        print("No matching job records found.")

    for job in jobs:
        print("-" * 60)
        for key, value in job.items():
            print(f"{key}: {value}")
