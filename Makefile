# Thin wrapper: every target calls the cross-platform runner `python -m aml.cli`.
PY ?= python

.PHONY: data features detect train app test

data features detect train app test:
	$(PY) -m aml.cli $@
