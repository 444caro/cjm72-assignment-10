ENV_NAME := flask_env
APP_FILE := app.py
REQUIREMENTS := requirements.txt
UPLOADS_DIR := static/uploads


all: run

create:
	@echo "Creating virtual environment..."
	@test -d $(ENV_NAME) || python3 -m venv $(ENV_NAME)

install: create
	@echo "Installing dependencies..."
	@$(ENV_NAME)/bin/pip install --upgrade pip
	@$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

run: install
	@echo "Starting Flask app..."
	@$(ENV_NAME)/bin/python $(APP_FILE)

clean:
	@echo "Cleaning up..."
	@rm -rf $(ENV_NAME)
	@find . -type d -name '__pycache__' -exec rm -r {} +
	@rm -rf $(UPLOADS_DIR)/*
	@echo "Cleanup complete."
