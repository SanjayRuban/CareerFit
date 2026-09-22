#!/usr/bin/env bash
# One command to set everything up: deps -> train -> export -> demo.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

if [ ! -d ".venv" ]; then
  echo ">> creating virtual environment"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ">> training + exporting model"
python main.py train

echo ">> demo recommendations"
python main.py demo

echo
echo "Model exported to models/job_recommender.joblib"
echo "Next:"
echo "  source .venv/bin/activate && python app.py        # web UI  (http://localhost:8501)"
echo "  source .venv/bin/activate && python main.py serve # REST API (http://localhost:8000)"
