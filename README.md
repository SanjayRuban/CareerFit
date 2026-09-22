# Job Recommendation System

A content-based job recommender built on 50,000 job postings. It embeds postings
and candidate profiles into the same weighted vector space (skills, title,
industry, location, seniority, salary) and retrieves the closest matches with
cosine nearest neighbours. Training, evaluation and model export run in one
command; inference is available as a CLI and a REST API.

## Quickstart

One command — creates a virtualenv, installs dependencies, trains, exports the
model and prints sample recommendations:

```bash
bash run.sh
```

Already have the dependencies? Then:

```bash
pip install -r requirements.txt
python main.py            # trains if no model exists, then runs the demo
```

Training on the full 50k rows takes a few seconds and writes:

```
models/job_recommender.joblib   # vectorizers + index + job catalogue
models/metrics.json             # held-out evaluation scores
```

## Web UI

A Streamlit app is included — one file, `app.py`. It loads the trained model
directly (no separate backend needed), lets you build a profile in a sidebar
form, and for every recommendation shows *why* it was suggested: matched
skills, missing skills, and whether location/industry lined up.

```bash
python app.py            # launches Streamlit for you, http://localhost:8501
# or, the standard way:
streamlit run app.py
```

It also has a "find jobs similar to a specific posting" panel (item-to-item,
by `job_id`) and a raw scored table you can expand for the underlying numbers.
The model must be trained first (`python main.py train` / `bash run.sh` does
this); the page tells you if `models/job_recommender.joblib` is missing.

## Usage

```bash
# retrain and re-export
python main.py train

# recommendations for a candidate profile
python main.py recommend \
  --skills "Python, SQL, Machine Learning" \
  --title "Data Scientist" \
  --location Bangalore \
  --experience "Mid Level" \
  --industry Software \
  --expected-salary 120000 \
  --top-k 10

# hard constraints (applied after scoring) and JSON output
python main.py recommend --skills "Nursing, Patient Care" \
  --filter-location London --min-salary 90000 --json

# "more jobs like this one"
python main.py recommend --similar-to 42

# REST API on http://127.0.0.1:8000
python main.py serve
```

API:

```bash
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' -d '{
  "skills": ["Python", "SQL"],
  "location": "Bangalore",
  "experience_level": "Mid Level",
  "top_k": 5,
  "filters": {"location": "Bangalore", "min_salary": 100000}
}'

curl localhost:8000/similar/42?top_k=5
curl localhost:8000/meta      # valid locations, industries, skills
curl localhost:8000/health
```

Use in your own Python code:

```python
from src.model import JobRecommender

model = JobRecommender.load()
recs = model.recommend({"skills": ["SEO", "Content Writing"], "location": "London"}, top_k=5)
print(recs)
```

## How it works

| Block | Representation | Weight |
|---|---|---|
| Required skills | TF-IDF over comma-separated skill tokens | 0.50 |
| Job title | Word 1–2 gram TF-IDF | 0.20 |
| Industry | One-hot | 0.15 |
| Location | One-hot | 0.07 |
| Experience level | One-hot | 0.05 |
| Salary | Min-max scaled | 0.03 |

Each block is L2-normalised and multiplied by its weight, then horizontally
stacked into one sparse matrix, so similarity is effectively a weighted cosine.
Retrieval uses `NearestNeighbors(metric="cosine")`. Hard filters (city, salary
floor, seniority, industry) are applied to an over-fetched candidate pool, which
keeps the ranking intact while guaranteeing the constraints. Every result comes
with an explanation: which of your skills matched and which the posting wants
that you don't list.

Weights live in `src/config.py` — tune them there and retrain.

## Evaluation

The dataset has no click or application log, so there is no ground truth for
"the user liked this job". `src/evaluate.py` uses held-out proxy relevance: 10%
of postings are held out, each is turned into a candidate profile, and a
recommendation counts as relevant if it shares the industry **and** at least one
skill with the held-out posting. Scores from the last run (k=10):

| Metric | Model | Random baseline |
|---|---|---|
| Precision@10 | 0.99 | 0.12 |
| MAP@10 | 0.99 | – |
| NDCG@10 | 1.00 | – |
| Hit rate@10 | 1.00 | – |
| Mean skill Jaccard | 0.99 | 0.07 |

A caveat worth keeping in mind: these numbers are very high because this dataset
is synthetic — skills are sampled from a small per-industry pool, so thousands of
postings share identical skill sets and job titles are uncorrelated with them
(you will see a "Chiropractor" tagged Software with Python). The metrics show
that retrieval is internally consistent, not that a real user would be happy.
On real postings expect materially lower scores, and swap this module for
log-based evaluation (CTR, application rate, A/B tests) once interaction data
exists.

## Project structure

```
job-recommender/
├── main.py                  # single entrypoint: train | recommend | serve | demo
├── app.py                   # Streamlit web UI (python app.py or streamlit run app.py)
├── run.sh                   # one-command setup + train + demo
├── Makefile                 # make train / demo / serve / test
├── requirements.txt
├── data/
│   └── job_recommendation_dataset.csv
├── models/                  # generated artefacts (gitignored)
├── src/
│   ├── config.py            # paths, column names, feature weights
│   ├── data.py              # loading, cleaning, deduplication
│   ├── features.py          # vectorizers for jobs and profiles
│   ├── model.py             # JobRecommender: fit / recommend / similar / save / load
│   ├── evaluate.py          # held-out proxy metrics vs random baseline
│   ├── train.py             # training pipeline + export
│   ├── recommend.py         # CLI inference
│   └── api.py               # Flask REST API
└── tests/
    └── test_pipeline.py     # pytest smoke tests
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Push to git

```bash
cd job-recommender
git init
git add .
git commit -m "Content-based job recommendation system"
git branch -M main
git remote add origin https://github.com/<you>/job-recommender.git
git push -u origin main
```

`.gitignore` excludes `models/*.joblib` — the artefact is ~2.5 MB and anyone who
clones the repo regenerates it in seconds with `python main.py train`. If you
want the trained model in the repo, delete that line from `.gitignore` (or track
it with Git LFS).

## Possible next steps

- Collaborative filtering or a two-tower model once application logs exist.
- Sentence embeddings for job titles so that related titles match without exact
  word overlap.
- Diversity re-ranking (MMR) so one company doesn't fill the whole list.
- FAISS or ScaNN instead of brute-force search beyond a few million postings.
