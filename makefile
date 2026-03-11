.PHONY: install setup run

install:
	pip install -r requirements.txt

setup:
	python -m venv .venv
	.venv/bin/activate && pip install -r requirements.txt

run:
	.venv/bin/activate && uvicorn main:app --reload

clean:
	rm -rf .venv
