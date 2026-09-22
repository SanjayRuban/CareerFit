"""Command line inference against the exported model."""
from __future__ import annotations

import argparse
import json

from src import config
from src.data import split_skills
from src.model import JobRecommender


def format_table(recs, explain=None) -> str:
    lines = []
    for i, row in recs.iterrows():
        lines.append(
            f"{i + 1:>2}. {row[config.COL_TITLE]}  ({row['match_score']:.3f})\n"
            f"    {row[config.COL_COMPANY]} | {row[config.COL_LOCATION]} | "
            f"{row[config.COL_EXPERIENCE]} | {row[config.COL_INDUSTRY]} | "
            f"${row[config.COL_SALARY]:,.0f}\n"
            f"    skills: {row[config.COL_SKILLS]}"
        )
        if explain:
            matched = explain[i]["matched_skills"]
            if matched:
                lines[-1] += f"\n    you match: {', '.join(matched)}"
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Get job recommendations")
    p.add_argument("--model", default=None, help="path to the exported model")
    p.add_argument("--skills", default="", help='comma separated, e.g. "Python, SQL, AWS"')
    p.add_argument("--title", default="", help="current or target job title")
    p.add_argument("--location", default="", help="preferred location")
    p.add_argument("--experience", default="", help="Entry Level | Mid Level | Senior Level")
    p.add_argument("--industry", default="", help="preferred industry")
    p.add_argument("--expected-salary", type=float, default=None)
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--similar-to", type=int, default=None, help="job_id for item-to-item")
    p.add_argument("--filter-location", default=None, help="hard filter on location")
    p.add_argument("--filter-experience", default=None)
    p.add_argument("--filter-industry", default=None)
    p.add_argument("--min-salary", type=float, default=None)
    p.add_argument("--max-salary", type=float, default=None)
    p.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    args = p.parse_args()

    model = JobRecommender.load(args.model)

    if args.similar_to is not None:
        recs = model.similar_jobs(args.similar_to, top_k=args.top_k)
        explanations = None
        header = f"Jobs similar to job_id {args.similar_to}"
    else:
        profile = {
            "title": args.title,
            "skills": split_skills(args.skills),
            "location": args.location,
            "experience_level": args.experience,
            "industry": args.industry,
            "expected_salary": args.expected_salary,
        }
        filters = {
            "location": args.filter_location,
            "experience_level": args.filter_experience,
            "industry": args.filter_industry,
            "min_salary": args.min_salary,
            "max_salary": args.max_salary,
        }
        recs = model.recommend(profile, top_k=args.top_k, filters=filters)
        explanations = [model.explain(profile, r) for _, r in recs.iterrows()]
        header = "Top matches for your profile"

    if args.json:
        payload = recs.to_dict(orient="records")
        if explanations:
            for item, ex in zip(payload, explanations):
                item["explanation"] = ex
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"\n{header}\n" + "=" * len(header))
        print(format_table(recs, explanations) or "No jobs matched those filters.")
        print()


if __name__ == "__main__":
    main()
