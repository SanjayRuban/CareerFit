#!/usr/bin/env python3
"""Single entrypoint for the job recommendation system.

    python main.py              # train (if needed) + demo recommendations
    python main.py train        # force retraining
    python main.py recommend --skills "Python, SQL" --location Bangalore
    python main.py serve        # REST API on http://127.0.0.1:8000
    python main.py demo         # sample profiles against the trained model
"""
from __future__ import annotations

import sys

from src import config


def run_demo() -> None:
    from src.model import JobRecommender
    from src.recommend import format_table

    model = JobRecommender.load()
    profiles = [
        {
            "name": "Data scientist, Bangalore",
            "profile": {
                "title": "Data Scientist",
                "skills": ["Python", "SQL", "Machine Learning"],
                "location": "Bangalore",
                "experience_level": "Mid Level",
                "industry": "Software",
                "expected_salary": 120000,
            },
        },
        {
            "name": "Marketing associate, London",
            "profile": {
                "title": "Marketing Executive",
                "skills": ["SEO", "Content Writing", "Google Ads"],
                "location": "London",
                "experience_level": "Entry Level",
                "industry": "Marketing",
            },
        },
        {
            "name": "Nurse, New York (hard filter on city)",
            "profile": {
                "title": "Registered Nurse",
                "skills": ["Patient Care", "Nursing"],
                "location": "New York",
                "experience_level": "Senior Level",
                "industry": "Healthcare",
            },
            "filters": {"location": "New York", "min_salary": 90000},
        },
    ]

    for case in profiles:
        recs = model.recommend(
            case["profile"], top_k=5, filters=case.get("filters")
        )
        explanations = [model.explain(case["profile"], r) for _, r in recs.iterrows()]
        print(f"\n### {case['name']}")
        print("-" * (len(case["name"]) + 4))
        print(format_table(recs, explanations))

    print("\n### Item-to-item: jobs similar to job_id 0")
    print("-" * 42)
    print(format_table(model.similar_jobs(0, top_k=5)))
    print()


def main() -> None:
    args = sys.argv[1:]
    command = args[0] if args and not args[0].startswith("-") else None

    if command == "train":
        sys.argv = ["train"] + args[1:]
        from src.train import main as train_main

        train_main()
        return

    if command == "recommend":
        sys.argv = ["recommend"] + args[1:]
        from src.recommend import main as rec_main

        rec_main()
        return

    if command == "serve":
        from src.api import main as api_main

        print("API on http://127.0.0.1:8000  (try POST /recommend)")
        api_main()
        return

    if command == "demo":
        run_demo()
        return

    if command is not None:
        print(__doc__)
        sys.exit(1)

    # Default: train if no artefact exists, then show the demo.
    if not config.MODEL_PATH.exists():
        print("No trained model found - training now.\n")
        from src.train import train

        train()
    else:
        print(f"Using existing model at {config.MODEL_PATH}\n")
    run_demo()


if __name__ == "__main__":
    main()
