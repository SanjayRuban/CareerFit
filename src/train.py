"""Train the recommender and export the artefact to models/."""
from __future__ import annotations

import argparse
import json
import time

from sklearn.model_selection import train_test_split

from src import config
from src.data import load_jobs
from src.evaluate import evaluate, save_metrics
from src.model import JobRecommender


def train(data_path=None, model_path=None, eval_sample: int = 500,
          top_k: int = config.DEFAULT_TOP_K, skip_eval: bool = False) -> dict:
    started = time.time()
    print(f"[1/5] Loading data from {data_path or config.DATA_PATH}")
    df = load_jobs(data_path)
    print(f"      {len(df):,} postings, {df[config.COL_TITLE].nunique():,} distinct titles")

    print("[2/5] Splitting train / held-out probe set")
    train_df, holdout_df = train_test_split(
        df, test_size=0.1, random_state=config.RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    print(f"      train={len(train_df):,}  holdout={len(holdout_df):,}")

    print("[3/5] Fitting vectorizers and nearest-neighbour index")
    model = JobRecommender().fit(train_df)
    print(f"      feature space: {model.matrix_.shape[1]:,} dimensions")

    metrics = {"skipped": True}
    if not skip_eval:
        print(f"[4/5] Evaluating on held-out profiles (k={top_k}, sample={eval_sample})")
        metrics = evaluate(model, holdout_df, k=top_k, sample=eval_sample)
        save_metrics(metrics)
        print("      " + json.dumps(metrics))
    else:
        print("[4/5] Evaluation skipped")

    print("[5/5] Exporting model")
    # Refit on the full catalogue so every posting is recommendable in production.
    final = JobRecommender().fit(df)
    path = final.save(model_path)
    print(f"      saved -> {path}")
    print(f"Done in {time.time() - started:.1f}s")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train the job recommender")
    p.add_argument("--data", default=None, help="path to the CSV dataset")
    p.add_argument("--out", default=None, help="path for the exported model")
    p.add_argument("--eval-sample", type=int, default=500)
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    train(
        data_path=args.data,
        model_path=args.out,
        eval_sample=args.eval_sample,
        top_k=args.top_k,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
