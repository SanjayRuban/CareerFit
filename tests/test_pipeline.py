"""Smoke tests: pipeline trains, persists and returns sane recommendations."""
import pandas as pd
import pytest

from src import config
from src.data import load_jobs, split_skills
from src.model import JobRecommender


@pytest.fixture(scope="module")
def sample_df():
    df = load_jobs()
    return df.sample(2000, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="module")
def model(sample_df):
    return JobRecommender().fit(sample_df)


def test_split_skills():
    assert split_skills("Python, SQL,  AWS ") == ["Python", "SQL", "AWS"]
    assert split_skills(None) == []


def test_load_has_expected_columns(sample_df):
    for col in [config.COL_TITLE, config.COL_SKILLS, "job_id", "skills_list"]:
        assert col in sample_df.columns
    assert sample_df[config.COL_SALARY].notna().all()


def test_recommend_returns_top_k(model):
    recs = model.recommend({"skills": ["Python", "SQL"]}, top_k=5)
    assert len(recs) == 5
    assert recs["match_score"].between(0, 1).all()
    assert recs["match_score"].is_monotonic_decreasing


def test_hard_filters_are_respected(model):
    recs = model.recommend(
        {"skills": ["Python", "SQL"]},
        top_k=10,
        filters={"location": "Bangalore", "min_salary": 80000},
    )
    assert (recs[config.COL_LOCATION] == "Bangalore").all()
    assert (recs[config.COL_SALARY] >= 80000).all()


def test_recommendations_beat_random_on_skill_overlap(model, sample_df):
    wanted = {"python", "sql", "machine learning"}
    recs = model.recommend({"skills": sorted(wanted)}, top_k=10)
    overlap = recs[config.COL_SKILLS].apply(
        lambda s: len(wanted & {x.lower() for x in split_skills(s)})
    )
    random_overlap = sample_df.sample(10, random_state=1)[config.COL_SKILLS].apply(
        lambda s: len(wanted & {x.lower() for x in split_skills(s)})
    )
    assert overlap.mean() > random_overlap.mean()


def test_similar_jobs_excludes_itself(model):
    recs = model.similar_jobs(int(model.jobs["job_id"].iloc[0]), top_k=5)
    assert int(model.jobs["job_id"].iloc[0]) not in set(recs["job_id"])


def test_empty_profile_raises(model):
    with pytest.raises(ValueError):
        model.recommend({"skills": []})


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = JobRecommender.load(path)
    a = model.recommend({"skills": ["Python"]}, top_k=3)
    b = loaded.recommend({"skills": ["Python"]}, top_k=3)
    pd.testing.assert_frame_equal(a, b)
