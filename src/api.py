"""Minimal REST API around the exported model.

    python -m src.api            # http://127.0.0.1:8000

Endpoints
    GET  /health
    GET  /meta                  -> known locations, industries, skills
    POST /recommend             -> {"skills": [...], "location": "...", ...}
    GET  /similar/<job_id>?top_k=10
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

from src import config
from src.data import split_skills
from src.model import JobRecommender

app = Flask(__name__)
_model: JobRecommender | None = None


def get_model() -> JobRecommender:
    global _model
    if _model is None:
        _model = JobRecommender.load(os.environ.get("MODEL_PATH") or None)
    return _model


@app.get("/health")
def health():
    return jsonify(status="ok", jobs=len(get_model().jobs))


@app.get("/meta")
def meta():
    return jsonify(get_model().vocab)


@app.post("/recommend")
def recommend():
    body = request.get_json(silent=True) or {}
    skills = body.get("skills") or []
    if isinstance(skills, str):
        skills = split_skills(skills)

    profile = {
        "title": body.get("title", ""),
        "skills": skills,
        "location": body.get("location", ""),
        "experience_level": body.get("experience_level", ""),
        "industry": body.get("industry", ""),
        "expected_salary": body.get("expected_salary"),
    }
    filters = body.get("filters") or {}
    top_k = int(body.get("top_k", config.DEFAULT_TOP_K))

    model = get_model()
    try:
        recs = model.recommend(profile, top_k=top_k, filters=filters)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    results = recs.to_dict(orient="records")
    for item, (_, row) in zip(results, recs.iterrows()):
        item["explanation"] = model.explain(profile, row)
    return jsonify(count=len(results), results=results)


@app.get("/similar/<int:job_id>")
def similar(job_id: int):
    top_k = int(request.args.get("top_k", config.DEFAULT_TOP_K))
    try:
        recs = get_model().similar_jobs(job_id, top_k=top_k)
    except ValueError as exc:
        return jsonify(error=str(exc)), 404
    return jsonify(count=len(recs), results=recs.to_dict(orient="records"))


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
