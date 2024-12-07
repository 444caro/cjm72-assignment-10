ENV_NAME := flask_env
APP_FILE := app.py
REQUIREMENTS := requirements.txt
UPLOADS_DIR := static/uploads

.PHONY: all create_env install_deps run clean

all: run

create_env:
	@echo "Creating virtual environment..."
	@test -d $(ENV_NAME) || python3 -m venv $(ENV_NAME)

install_deps: create_env
	@echo "Installing dependencies..."
	@$(ENV_NAME)/bin/pip install --upgrade pip
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

run: install_deps
	@echo "Starting Flask app..."
	@$(ENV_NAME)/bin/python $(APP_FILE)

clean:
	@echo "Cleaning up..."
	@rm -rf $(ENV_NAME)
	@find . -type d -name '__pycache__' -exec rm -r {} +
	@rm -rf $(UPLOADS_DIR)/*
	@echo "Cleanup complete."
