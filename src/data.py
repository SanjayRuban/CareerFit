"""Loading and cleaning of the job postings dataset."""
from __future__ import annotations

import pandas as pd

from src import config


def split_skills(value: str) -> list[str]:
    """Turn a 'Python, SQL, AWS' string into a clean list of skills."""
    if not isinstance(value, str):
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def load_jobs(path=None) -> pd.DataFrame:
    """Read the CSV, normalise text columns and drop unusable rows."""
    path = path or config.DATA_PATH
    df = pd.read_csv(path)

    required = [
        config.COL_TITLE,
        config.COL_COMPANY,
        config.COL_LOCATION,
        config.COL_EXPERIENCE,
        config.COL_SALARY,
        config.COL_INDUSTRY,
        config.COL_SKILLS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    for col in [
        config.COL_TITLE,
        config.COL_COMPANY,
        config.COL_LOCATION,
        config.COL_EXPERIENCE,
        config.COL_INDUSTRY,
        config.COL_SKILLS,
    ]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df[config.COL_SALARY] = pd.to_numeric(df[config.COL_SALARY], errors="coerce")
    df[config.COL_SALARY] = df[config.COL_SALARY].fillna(df[config.COL_SALARY].median())

    # A posting with neither a title nor skills carries no signal.
    df = df[(df[config.COL_TITLE] != "") | (df[config.COL_SKILLS] != "")]

    # Exact duplicates are noise in the recommendation list.
    df = df.drop_duplicates(subset=required).reset_index(drop=True)

    df["job_id"] = df.index.astype(int)
    df["skills_list"] = df[config.COL_SKILLS].apply(split_skills)
    return df


def vocabulary(df: pd.DataFrame) -> dict:
    """Allowed values for each categorical field, useful for validating input."""
    return {
        "locations": sorted(df[config.COL_LOCATION].unique().tolist()),
        "experience_levels": sorted(df[config.COL_EXPERIENCE].unique().tolist()),
        "industries": sorted(df[config.COL_INDUSTRY].unique().tolist()),
        "skills": sorted({s for row in df["skills_list"] for s in row}),
    }
