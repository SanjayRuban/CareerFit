#!/usr/bin/env python3
"""Job recommender UI.

Run it either way:
    streamlit run app.py
    python app.py            (launches streamlit for you)
"""
from __future__ import annotations

import subprocess
import sys

# --- allow `python app.py` by handing off to `streamlit run app.py` -------
if __name__ == "__main__" and "streamlit" not in sys.modules:
    import os

    if os.environ.get("_JR_STREAMLIT_LAUNCHED") != "1":
        os.environ["_JR_STREAMLIT_LAUNCHED"] = "1"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", __file__, "--server.headless=true"]
            )
        )

import pandas as pd
import streamlit as st

from src import config
from src.data import split_skills
from src.model import JobRecommender

st.set_page_config(page_title="Job Recommender", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .job-card {
        border: 1px solid #2a2a33; border-radius: 10px; padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem; background: #14141a;
    }
    .job-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.15rem; }
    .job-meta { color: #9a9aa8; font-size: 0.88rem; margin-bottom: 0.55rem; }
    .score-pill {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; background: #1f3a2e; color: #6fd89a;
    }
    .skill-chip {
        display: inline-block; padding: 0.1rem 0.55rem; margin: 0.15rem 0.3rem 0 0;
        border-radius: 6px; font-size: 0.78rem;
    }
    .chip-match { background: #1f3a2e; color: #6fd89a; }
    .chip-missing { background: #3a1f1f; color: #d88a8a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model() -> JobRecommender:
    if not config.MODEL_PATH.exists():
        st.error(
            f"No trained model at `{config.MODEL_PATH}`. "
            "Run `python main.py train` first, then reload this page."
        )
        st.stop()
    return JobRecommender.load()


model = get_model()
vocab = model.vocab

st.title("🧭 Job Recommender")
st.caption(
    f"Content-based matching over {len(model.jobs):,} postings — skills, title, "
    "industry, location, seniority and salary, in one weighted vector space."
)

# ------------------------------------------------------------------ input
with st.sidebar:
    st.header("Your profile")
    title = st.text_input("Target / current job title", placeholder="e.g. Data Scientist")
    skills = st.multiselect("Skills", options=vocab["skills"])
    extra_skills = st.text_input(
        "Other skills (comma separated)", placeholder="not in the list above"
    )
    location = st.selectbox("Preferred location", ["Any"] + vocab["locations"])
    experience = st.selectbox("Experience level", ["Any"] + vocab["experience_levels"])
    industry = st.selectbox("Industry", ["Any"] + vocab["industries"])
    expected_salary = st.number_input(
        "Expected salary (optional, 0 = ignore)", min_value=0, max_value=300000,
        value=0, step=5000,
    )

    st.divider()
    st.subheader("Hard filters")
    st.caption("These exclude non-matching jobs outright, instead of just scoring them lower.")
    filter_location = st.checkbox("Require the location above")
    filter_experience = st.checkbox("Require the experience level above")
    filter_industry = st.checkbox("Require the industry above")
    min_salary = st.number_input("Minimum salary", min_value=0, max_value=300000, value=0, step=5000)

    top_k = st.slider("How many recommendations", 3, 25, 10)
    submitted = st.button("Find jobs", type="primary", use_container_width=True)

st.divider()

# ------------------------------------------------------------ item lookup
with st.expander("🔎 Or find jobs similar to a specific posting"):
    job_id_input = st.number_input(
        "job_id", min_value=0, max_value=int(model.jobs["job_id"].max()), value=0, step=1
    )
    similar_clicked = st.button("Show similar jobs")
    if similar_clicked:
        similar = model.similar_jobs(int(job_id_input), top_k=8)
        base = model.jobs[model.jobs["job_id"] == job_id_input].iloc[0]
        st.markdown(f"**Similar to:** {base[config.COL_TITLE]} at {base[config.COL_COMPANY]} "
                    f"({base[config.COL_LOCATION]}, {base[config.COL_SKILLS]})")
        for _, row in similar.iterrows():
            st.markdown(
                f"- **{row[config.COL_TITLE]}** — {row[config.COL_COMPANY]} · "
                f"{row[config.COL_LOCATION]} · match `{row['match_score']:.3f}` · "
                f"skills: {row[config.COL_SKILLS]}"
            )

# --------------------------------------------------------------- results
def render_card(row: pd.Series, explanation: dict) -> None:
    chips = "".join(
        f'<span class="skill-chip chip-match">✓ {s}</span>' for s in explanation["matched_skills"]
    ) + "".join(
        f'<span class="skill-chip chip-missing">missing: {s}</span>'
        for s in explanation["missing_skills"][:6]
    )
    tags = []
    if explanation["location_match"]:
        tags.append("📍 location match")
    if explanation["industry_match"]:
        tags.append("🏢 industry match")
    tag_str = " · ".join(tags)

    st.markdown(
        f"""
        <div class="job-card">
            <div class="job-title">{row[config.COL_TITLE]}
                <span class="score-pill">match {row['match_score']:.2f}</span>
            </div>
            <div class="job-meta">
                {row[config.COL_COMPANY]} · {row[config.COL_LOCATION]} ·
                {row[config.COL_EXPERIENCE]} · {row[config.COL_INDUSTRY]} ·
                ${row[config.COL_SALARY]:,.0f} · job_id {row['job_id']}
            </div>
            <div>{chips}</div>
            <div class="job-meta" style="margin-top:0.4rem;">{tag_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if submitted:
    all_skills = list(skills) + split_skills(extra_skills)
    profile = {
        "title": title,
        "skills": all_skills,
        "location": None if location == "Any" else location,
        "experience_level": None if experience == "Any" else experience,
        "industry": None if industry == "Any" else industry,
        "expected_salary": expected_salary or None,
    }
    filters = {
        "location": location if (filter_location and location != "Any") else None,
        "experience_level": experience if (filter_experience and experience != "Any") else None,
        "industry": industry if (filter_industry and industry != "Any") else None,
        "min_salary": min_salary or None,
    }

    if not all_skills and not title:
        st.warning("Add at least a job title or one skill so there's something to match on.")
    else:
        try:
            recs = model.recommend(profile, top_k=top_k, filters=filters)
        except ValueError as exc:
            st.error(str(exc))
            recs = pd.DataFrame()

        if recs.empty:
            st.info("No postings matched those filters. Try loosening a filter or salary floor.")
        else:
            st.subheader(f"Top {len(recs)} matches")
            for _, row in recs.iterrows():
                explanation = model.explain(profile, row)
                render_card(row, explanation)

            with st.expander("See raw scored table"):
                st.dataframe(recs, use_container_width=True, hide_index=True)
else:
    st.info("Fill in your profile in the sidebar and click **Find jobs**.")
