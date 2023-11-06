#!make

PYTHON_VERSION := 3.11.5
PROJECT_NAME := bids23hackathon
PYTHON_VERSION_NAME := py-${PYTHON_VERSION}-${PROJECT_NAME}

prep-local-env:
	@brew install virtualenv pyenv
	@pyenv shell ${PYTHON_VERSION_NAME}

install-local-env:
	@pyenv install -s ${PYTHON_VERSION}
	@echo ${PYTHON_VERSION} ${PYTHON_VERSION_NAME}
	@pyenv virtualenv ${PYTHON_VERSION} ${PYTHON_VERSION_NAME}
	@pyenv local ${PYTHON_VERSION_NAME} && pip install --upgrade pip
	@pyenv local ${PYTHON_VERSION_NAME} && pip install poetry
	@ll ~/.pyenv/versions/${PYTHON_VERSION_NAME}/bin/python

run-stac-app:
	@cd stac-app
	@./run_app.sh