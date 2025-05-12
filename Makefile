# Makefile

.PHONY: install train test

install:
    pip install -r requirements.txt

train:
    python src/train.py

test:
    pytest tests/