PYTHON = python3
SOURCES = client/main.py
TESTS = test_x

run:
	python3 -m client.main

test:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test);)

test_verbose:
	$(foreach test,$(TESTS),python3 -m unittest tests.$(test) -v;)

demo:
	@echo TODO

conda_install:
	conda env create -f environment.yml

conda_remove:
	conda remove --name social-group-clustering --all

.PHONY: run