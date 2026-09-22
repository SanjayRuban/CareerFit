"""Offline evaluation.

The dataset has no click/application log, so there is no ground-truth label of
"the user liked this job". We therefore evaluate with held-out proxy relevance:

  * take a held-out posting, treat its skills/title/location as a candidate
    profile (the posting itself is never in the index for that query),
  * a recommendation counts as relevant if it shares the industry AND at least
    one skill with the held-out posting,
  * report Precision@k, Recall-style hit rate, MAP@k, NDCG@k and mean skill
    Jaccard, each against a random-recommender baseline.

These numbers say how coherent the retrieval is, not how happy a real user is.
Replace this module with log-based evaluation once application data exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import split_skills


def _relevant(profile_row: pd.Series, rec_row: pd.Series) -> bool:
    a = {s.lower() for s in split_skills(profile_row[config.COL_SKILLS])}
    b = {s.lower() for s in split_skills(rec_row[config.COL_SKILLS])}
    same_industry = profile_row[config.COL_INDUSTRY] == rec_row[config.COL_INDUSTRY]
    return bool(same_industry and (a & b))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ndcg(rels: list[int]) -> float:
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / np.log2(i + 2) for i in range(sum(rels)))
    return float(dcg / ideal) if ideal else 0.0


def _average_precision(rels: list[int]) -> float:
    hits, score = 0, 0.0
    for i, r in enumerate(rels):
        if r:
            hits += 1
            score += hits / (i + 1)
    return score / hits if hits else 0.0


def evaluate(model, holdout: pd.DataFrame, k: int = 10, sample: int = 500,
             seed: int = config.RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    if len(holdout) > sample:
        holdout = holdout.sample(sample, random_state=seed)

    catalogue = model.jobs
    precisions, jaccards, ndcgs, maps, hits = [], [], [], [], []
    base_precisions, base_jaccards = [], []

    for _, row in holdout.iterrows():
        profile = {
            "title": row[config.COL_TITLE],
            "skills": split_skills(row[config.COL_SKILLS]),
            "location": row[config.COL_LOCATION],
            "experience_level": row[config.COL_EXPERIENCE],
            "industry": None,  # industry is part of what we are testing for
        }
        recs = model.recommend(profile, top_k=k)
        rels = [int(_relevant(row, r)) for _, r in recs.iterrows()]
        precisions.append(np.mean(rels) if rels else 0.0)
        hits.append(1.0 if any(rels) else 0.0)
        ndcgs.append(_ndcg(rels))
        maps.append(_average_precision(rels))

        want = {s.lower() for s in split_skills(row[config.COL_SKILLS])}
        jaccards.append(
            float(
                np.mean(
                    [
                        _jaccard(want, {s.lower() for s in split_skills(r[config.COL_SKILLS])})
                        for _, r in recs.iterrows()
                    ]
                )
            )
            if len(recs)
            else 0.0
        )

        idx = rng.integers(0, len(catalogue), size=k)
        rand = catalogue.iloc[idx]
        base_precisions.append(np.mean([int(_relevant(row, r)) for _, r in rand.iterrows()]))
        base_jaccards.append(
            float(
                np.mean(
                    [
                        _jaccard(want, {s.lower() for s in split_skills(r[config.COL_SKILLS])})
                        for _, r in rand.iterrows()
                    ]
                )
            )
        )

    return {
        "k": k,
        "evaluated_profiles": int(len(holdout)),
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "hit_rate_at_k": round(float(np.mean(hits)), 4),
        "map_at_k": round(float(np.mean(maps)), 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4),
        "mean_skill_jaccard": round(float(np.mean(jaccards)), 4),
        "baseline_random_precision_at_k": round(float(np.mean(base_precisions)), 4),
        "baseline_random_skill_jaccard": round(float(np.mean(base_jaccards)), 4),
        "lift_over_random": round(
            float(np.mean(precisions) / max(np.mean(base_precisions), 1e-9)), 2
        ),
    }


def save_metrics(metrics: dict, path=None) -> str:
    path = Path(path) if path else config.METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return str(path)
