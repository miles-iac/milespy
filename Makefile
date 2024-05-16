.DEFAULT_GOAL := init
.PHONY: add_major_version add_minor_version add_patch_version add_premajor_version add_preminor_version add_prepatch_version add_prerelease_version prepare-dev install-dev data help lint reference-plot tests upload-prod-pypi upload-test-pypi update_req update_req_dev pyclean doc doc-pdf visu-doc-pdf visu-doc licences
VENV = ".pymiles"

define PROJECT_HELP_MSG

Usage:\n
	\n
    make help\t\t\t             show this message\n
	\n
	-------------------------------------------------------------------------\n
	\t\tInstallation\n
	-------------------------------------------------------------------------\n
	make\t\t\t\t                Install pymiles in the system (root)\n
	make user\t\t\t 			Install pymiles for non-root usage\n
	\n
	-------------------------------------------------------------------------\n
	\t\tDevelopment\n
	-------------------------------------------------------------------------\n
	make prepare-dev\t\t 		Prepare Development environment\n
	make install-dev\t\t 		Install COTS and pymiles for development purpose\n
	make reference-img\t\t     Generate reference images for tests\n
	make tests\t\t\t             Run units and integration tests\n
	\n
	make doc\t\t\t 				Generate the documentation\n
	make doc-pdf\t\t\t 			Generate the documentation as PDF\n
	make visu-doc-pdf\t\t 		View the generated PDF\n
	make visu-doc\t\t\t			View the generated documentation\n
	\n
	make release\t\t\t 			Release the package as tar.gz\n
	\n
	make update_req\t\t			Update the version of the packages in pyproject.toml\n
	\n
	make pyclean\t\t\t		Clean .pyc files and __pycache__ directories\n
	\n
	-------------------------------------------------------------------------\n
	\t\tVersion\n
	-------------------------------------------------------------------------\n
	make version\t\t\t		Display the version\n
	make add_major_version\t\t	Add a major version\n
	make add_minor_version\t\t	Add a major version\n
	make add_patch_version\t\t	Add a major version\n
	make add_premajor_version\t	Add a pre-major version\n
	make add_preminor_version\t	Add a pre-minor version\n
	make add_prepatch_version\t	Add a pre-patch version\n
	make add_prerelease_version\t	Add a pre-release version\n
	\n
	-------------------------------------------------------------------------\n
	\t\tOthers\n
	-------------------------------------------------------------------------\n
	make licences\t\t\t		Display the list of licences\n
	make lint\t\t\t			Lint\n

endef
export PROJECT_HELP_MSG


#Show help
#---------
help:
	echo $$PROJECT_HELP_MSG


#
# Sotware Installation in the system (need root access)
# -----------------------------------------------------
#
init:
	poetry install --no-dev

#
# Sotware Installation for user
# -----------------------------
# This scheme is designed to be the most convenient solution for users
# that don’t have write permission to the global site-packages directory or
# don’t want to install into it.
#
user:
	poetry install --no-dev

prepare-dev:
	git config --global init.defaultBranch main && git init && echo "python3 -m venv pymiles-env && export PYTHONPATH=. && export PATH=`pwd`/pymiles-env/bin:${PATH}" > ${VENV} && echo "source \"`pwd`/pymiles-env/bin/activate\"" >> ${VENV} && echo "\nnow source this file: \033[31msource ${VENV}\033[0m"

install-dev:
	poetry install && poetry run pre-commit install

lint:  ## Lint and static-check
	poetry run flake8 --ignore=E203,E266,E501,W503,F403,F401 --max-line-length=79 --max-complexity=18 --select=B,C,E,F,W,T4,B9 pymiles
	poetry run pylint pymiles
	poetry run mypy --install-types --non-interactive pymiles


reference-img:  ## Generate reference images for the tests
	poetry run pytest --mpl-generate-path=test/baseline

tests:  ## Run tests
	poetry run pytest --mpl

doc:
	make licences
	poetry run pytest -ra --html=docs/source/_static/report.html
	make html -C docs

doc-pdf:
	make doc && make latexpdf -C docs

visu-doc-pdf:
	acroread docs/build/latex/pymiles.pdf

visu-doc:
	firefox docs/build/html/index.html

release:
	poetry build

version:
	poetry version -s

add_major_version:
	poetry version major
	poetry run git tag $(shell poetry version -s)

add_minor_version:
	poetry version minor
	poetry run git tag $(shell poetry version -s)

add_patch_version:
	poetry version patch
	poetry run git tag $(shell poetry version -s)

add_premajor_version:
	poetry version premajor

add_preminor_version:
	poetry version preminor

add_prepatch_version:
	poetry version prepatch

add_prerelease_version:
	poetry version prerelease

update_req:
	poetry update

pyclean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
