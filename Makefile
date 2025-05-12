.PHONY: install train test

install:
	pip install -r requirements.txt   # Esto debe tener una tabulación antes

train:
	python src/train.py   # Esto debe tener una tabulación antes

test:
	pytest tests/   # Esto debe tener una tabulación antes