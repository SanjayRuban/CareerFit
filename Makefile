.PHONY: install train demo serve ui test clean

install:
	pip install -r requirements.txt

train:
	python main.py train

demo:
	python main.py demo

serve:
	python main.py serve

ui:
	python app.py

test:
	python -m pytest -q

clean:
	rm -rf models/*.joblib models/metrics.json __pycache__ src/__pycache__ .pytest_cache
