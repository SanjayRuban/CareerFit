"""Central configuration for the job recommendation system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "job_recommendation_dataset.csv"
MODEL_PATH = MODEL_DIR / "job_recommender.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Column names in the source CSV
COL_TITLE = "Job Title"
COL_COMPANY = "Company"
COL_LOCATION = "Location"
COL_EXPERIENCE = "Experience Level"
COL_SALARY = "Salary"
COL_INDUSTRY = "Industry"
COL_SKILLS = "Required Skills"

# Relative importance of each block of features in the similarity space.
# They are applied as multipliers on L2-normalised feature blocks, so the
# numbers behave like weights in a weighted cosine similarity.
FEATURE_WEIGHTS = {
    "skills": 0.50,
    "title": 0.20,
    "industry": 0.15,
    "location": 0.07,
    "experience": 0.05,
    "salary": 0.03,
}

# Number of neighbours the ANN index keeps ready (over-fetch so that hard
# filters such as location still leave enough candidates).
INDEX_NEIGHBOURS = 200
DEFAULT_TOP_K = 10
RANDOM_STATE = 42
