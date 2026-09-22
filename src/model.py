"""The recommender itself: a content-based nearest-neighbour model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data import split_skills
from src.features import FeatureBuilder


@dataclass
class JobRecommender:
    """Content-based recommender over job postings.

    Jobs and candidate profiles are embedded in the same weighted vector space
    (skills, title, industry, location, seniority, salary) and matched with
    cosine similarity via a NearestNeighbors index.
    """

    weights: dict = field(default_factory=lambda: dict(config.FEATURE_WEIGHTS))
    features: FeatureBuilder = None
    index: NearestNeighbors = None
    jobs: pd.DataFrame = None
    vocab: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "JobRecommender":
        from src.data import vocabulary

        self.features = FeatureBuilder(self.weights)
        matrix = self.features.fit_transform(df)
        n_neighbors = min(config.INDEX_NEIGHBOURS, len(df))
        self.index = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute"
        ).fit(matrix)
        self.jobs = df.reset_index(drop=True)
        self.matrix_ = matrix
        self.vocab = vocabulary(df)
        return self

    # -------------------------------------------------------- recommending
    def recommend(
        self,
        profile: dict,
        top_k: int = config.DEFAULT_TOP_K,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """Return the top_k postings that best match a candidate profile.

        filters (all optional): location, experience_level, industry,
        min_salary, max_salary. These are hard constraints applied after
        scoring; the soft preferences live in the profile itself.
        """
        query = self.features.transform_profile(profile)
        if query.nnz == 0:
            raise ValueError(
                "Profile produced an empty query - provide at least skills or a title."
            )
        return self._search(query, top_k=top_k, filters=filters, exclude=None)

    def similar_jobs(self, job_id: int, top_k: int = config.DEFAULT_TOP_K) -> pd.DataFrame:
        """Item-to-item recommendations: 'more jobs like this one'."""
        if job_id not in set(self.jobs["job_id"]):
            raise ValueError(f"Unknown job_id: {job_id}")
        row = self.jobs.index[self.jobs["job_id"] == job_id][0]
        query = self.features.transform_jobs(self.jobs.iloc[[row]])
        return self._search(query, top_k=top_k, filters=None, exclude={job_id})

    # ----------------------------------------------------------- internals
    def _search(self, query, top_k, filters, exclude):
        pool = min(
            len(self.jobs),
            max(top_k * 20, config.INDEX_NEIGHBOURS if filters else top_k * 5),
        )
        distances, indices = self.index.kneighbors(query, n_neighbors=pool)
        distances, indices = distances[0], indices[0]

        out = self.jobs.iloc[indices].copy()
        out["match_score"] = np.round(1.0 - distances, 4)

        if exclude:
            out = out[~out["job_id"].isin(exclude)]
        out = self._apply_filters(out, filters or {})
        out = out.head(top_k)

        cols = [
            "job_id",
            config.COL_TITLE,
            config.COL_COMPANY,
            config.COL_LOCATION,
            config.COL_EXPERIENCE,
            config.COL_INDUSTRY,
            config.COL_SALARY,
            config.COL_SKILLS,
            "match_score",
        ]
        return out[cols].reset_index(drop=True)

    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        if filters.get("location"):
            df = df[df[config.COL_LOCATION].str.lower() == str(filters["location"]).lower()]
        if filters.get("experience_level"):
            df = df[
                df[config.COL_EXPERIENCE].str.lower()
                == str(filters["experience_level"]).lower()
            ]
        if filters.get("industry"):
            df = df[df[config.COL_INDUSTRY].str.lower() == str(filters["industry"]).lower()]
        if filters.get("min_salary") is not None:
            df = df[df[config.COL_SALARY] >= float(filters["min_salary"])]
        if filters.get("max_salary") is not None:
            df = df[df[config.COL_SALARY] <= float(filters["max_salary"])]
        return df

    # ------------------------------------------------------------ helpers
    def explain(self, profile: dict, job_row: pd.Series) -> dict:
        """Why was this job suggested? Overlap between profile and posting."""
        wanted = {s.lower() for s in (profile.get("skills") or [])}
        if isinstance(profile.get("skills"), str):
            wanted = {s.lower() for s in split_skills(profile["skills"])}
        have = {s.lower() for s in split_skills(job_row[config.COL_SKILLS])}
        return {
            "matched_skills": sorted(wanted & have),
            "missing_skills": sorted(have - wanted),
            "location_match": bool(
                profile.get("location")
                and profile["location"].lower() == job_row[config.COL_LOCATION].lower()
            ),
            "industry_match": bool(
                profile.get("industry")
                and profile["industry"].lower() == job_row[config.COL_INDUSTRY].lower()
            ),
        }

    # -------------------------------------------------------- persistence
    def save(self, path=None) -> str:
        path = Path(path) if path else config.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights,
            "features": self.features,
            "index": self.index,
            "jobs": self.jobs,
            "vocab": self.vocab,
        }
        joblib.dump(payload, path, compress=3)
        return str(path)

    @classmethod
    def load(cls, path=None) -> "JobRecommender":
        path = Path(path) if path else config.MODEL_PATH
        payload = joblib.load(path)
        model = cls(weights=payload["weights"])
        model.features = payload["features"]
        model.index = payload["index"]
        model.jobs = payload["jobs"]
        model.vocab = payload["vocab"]
        return model
