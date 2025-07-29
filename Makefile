.DEFAULT_GOAL := init
.PHONY: install-dev reference-img help lint tests pyclean doc release
VENV = ".milespy"

define PROJECT_HELP_MSG

Usage:\n
	\n
    make help\t\t\t             show this message\n
	\n
	-------------------------------------------------------------------------\n
	\t\tInstallation\n
	-------------------------------------------------------------------------\n
	make\t\t\t\t                Install milespy in the current environment\n
	\n
	-------------------------------------------------------------------------\n
	\t\tDevelopment\n
	-------------------------------------------------------------------------\n
	make install-dev\t\t 		Install milespy for development purpose\n
	make reference-img\t\t     Generate reference images for tests\n
	make tests\t\t\t             Run units and integration tests\n
	\n
	make doc\t\t\t 				Generate the documentation\n
	\n
	make release\t\t\t 			Build the distribution files\n
	\n
	make pyclean\t\t\t		Clean .pyc files and __pycache__ directories\n
	\n
	make envclean\t\t\t		Remove the local development environment\n
	\n
	-------------------------------------------------------------------------\n
	\t\tOthers\n
	-------------------------------------------------------------------------\n
	make lint\t\t\t			Lint\n

endef
export PROJECT_HELP_MSG


#Show help
#---------
help:
	echo $$PROJECT_HELP_MSG

init:
	python3 -m pip install .

install-dev:
	uv lock && uv sync --all-groups

lint:  ## Lint and static-check
	uv run ruff check milespy

format:  ## Lint and static-check
	uv run ruff format milespy

reference-img:  ## Generate reference images for the tests
	uv run pytest --mpl-generate-path=test/baseline

tests:  ## Run tests
	MILESPY_LOG=DEBUG uv run coverage run -m pytest --html=test_results/report.html --self-contained-html --mpl --mpl-generate-summary=basic-html --mpl-results-path=test_results
	uv run coverage html

doc:
	rm -rf _build
	MILESPY_AUTO_DOWNLOAD=1 uv run sphinx-build -W --keep-going -b html docs/ _build/

release:
	uv build

version:
	uv version -s

pyclean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

envclean:
	rm -r .venv
