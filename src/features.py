"""Feature engineering: turns job postings and user profiles into one vector space."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, normalize

from src import config
from src.data import split_skills


def skills_analyzer(value: str) -> list[str]:
    """Comma separated skills are already tokens - keep them whole and lowercased.

    Defined at module level (not as a lambda) so the fitted vectorizer pickles.
    """
    return [s.lower() for s in split_skills(value)]


class FeatureBuilder:
    """Fits the vectorizers on the catalogue and re-uses them for queries."""

    def __init__(self, weights: dict | None = None):
        self.weights = dict(weights or config.FEATURE_WEIGHTS)
        self.skill_vec = TfidfVectorizer(analyzer=skills_analyzer)
        self.title_vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.cat_enc = OneHotEncoder(handle_unknown="ignore")
        self.salary_scaler = MinMaxScaler()
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        skills = self.skill_vec.fit_transform(df[config.COL_SKILLS])
        titles = self.title_vec.fit_transform(df[config.COL_TITLE])
        cats = self.cat_enc.fit_transform(
            df[[config.COL_LOCATION, config.COL_EXPERIENCE, config.COL_INDUSTRY]]
        )
        salary = self.salary_scaler.fit_transform(df[[config.COL_SALARY]])
        self.fitted = True
        return self._assemble(skills, titles, cats, salary)

    # -------------------------------------------------------------- queries
    def transform_jobs(self, df: pd.DataFrame) -> sparse.csr_matrix:
        skills = self.skill_vec.transform(df[config.COL_SKILLS])
        titles = self.title_vec.transform(df[config.COL_TITLE])
        cats = self.cat_enc.transform(
            df[[config.COL_LOCATION, config.COL_EXPERIENCE, config.COL_INDUSTRY]]
        )
        salary = self.salary_scaler.transform(df[[config.COL_SALARY]])
        return self._assemble(skills, titles, cats, salary)

    def transform_profile(self, profile: dict) -> sparse.csr_matrix:
        """Build a query vector from a candidate profile.

        Any field may be omitted; omitted fields simply contribute nothing.
        """
        skills_raw = profile.get("skills") or []
        if isinstance(skills_raw, str):
            skills_raw = split_skills(skills_raw)
        skills_text = ", ".join(skills_raw)

        frame = pd.DataFrame(
            [
                {
                    config.COL_TITLE: profile.get("title", "") or "",
                    config.COL_SKILLS: skills_text,
                    config.COL_LOCATION: profile.get("location", "") or "",
                    config.COL_EXPERIENCE: profile.get("experience_level", "") or "",
                    config.COL_INDUSTRY: profile.get("industry", "") or "",
                }
            ]
        )
        skills = self.skill_vec.transform(frame[config.COL_SKILLS])
        titles = self.title_vec.transform(frame[config.COL_TITLE])
        cats = self.cat_enc.transform(
            frame[[config.COL_LOCATION, config.COL_EXPERIENCE, config.COL_INDUSTRY]]
        )

        salary_pref = profile.get("expected_salary")
        if salary_pref is None:
            # Neutral value: contributes nothing but keeps the shape consistent.
            salary = np.zeros((1, 1))
            salary_weight_override = 0.0
        else:
            salary = self.salary_scaler.transform(
                pd.DataFrame({config.COL_SALARY: [float(salary_pref)]})
            )
            salary_weight_override = None

        return self._assemble(
            skills, titles, cats, salary, salary_weight=salary_weight_override
        )

    # ----------------------------------------------------------- internals
    def _assemble(self, skills, titles, cats, salary, salary_weight=None):
        n_loc = len(self.cat_enc.categories_[0])
        n_exp = len(self.cat_enc.categories_[1])
        cats = sparse.csr_matrix(cats)
        loc = cats[:, :n_loc]
        exp = cats[:, n_loc : n_loc + n_exp]
        ind = cats[:, n_loc + n_exp :]

        w = self.weights
        sw = w["salary"] if salary_weight is None else salary_weight

        blocks = [
            self._block(skills, w["skills"]),
            self._block(titles, w["title"]),
            self._block(ind, w["industry"]),
            self._block(loc, w["location"]),
            self._block(exp, w["experience"]),
            sparse.csr_matrix(np.asarray(salary) * sw),
        ]
        return sparse.hstack(blocks).tocsr()

    @staticmethod
    def _block(matrix, weight: float) -> sparse.csr_matrix:
        matrix = sparse.csr_matrix(matrix)
        if matrix.nnz == 0:
            return matrix * weight
        return normalize(matrix, norm="l2") * weight
