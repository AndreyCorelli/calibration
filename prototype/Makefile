VENV := venv/bin

.PHONY: test run

test:
	$(VENV)/pytest tests/ -v -s

run:
	$(VENV)/streamlit run app.py
