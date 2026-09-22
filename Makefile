.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT := App/Spektrafilm.xcodeproj
SCHEME  := Spektrafilm
SIM     := 'platform=iOS Simulator,name=iPhone 17 Pro'

.PHONY: help
help: ## List targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

.PHONY: build
build: ## Build the engine package
	swift build

.PHONY: test
test: ## Run engine + parity tests
	swift test --parallel

.PHONY: format
format: ## Format Swift sources in place
	swift format --in-place --recursive Sources Tests App/Spektrafilm Package.swift

.PHONY: lint
lint: ## Check formatting without writing
	swift format lint --strict --recursive Sources Tests App/Spektrafilm Package.swift

.PHONY: project
project: ## Generate App/Spektrafilm.xcodeproj from project.yml
	cd App && xcodegen generate

.PHONY: app
app: project ## Build the iOS app for the simulator
	set -o pipefail; xcodebuild build -project $(PROJECT) -scheme $(SCHEME) \
		-destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO | tail -30

.PHONY: run
run: project ## Build and launch the app in the simulator
	set -o pipefail; xcodebuild build -project $(PROJECT) -scheme $(SCHEME) \
		-destination $(SIM) CODE_SIGNING_ALLOWED=NO | tail -20

.PHONY: oracle
oracle: ## Create the pinned Python oracle venv under Tools/parity/oracle
	Tools/parity/setup_oracle.sh

.PHONY: tables
tables: ## Regenerate Sources/SpektraFilm/Generated from colour-science
	Tools/parity/oracle/.venv/bin/python Tools/parity/extract_colour_tables.py

.PHONY: goldens
goldens: ## Regenerate parity goldens from the pinned Python oracle
	Tools/parity/oracle/.venv/bin/python Tools/parity/generate_goldens.py

.PHONY: docs
docs: ## Check documentation consistency
	Tools/check_docs.sh

.PHONY: clean
clean: ## Remove build products
	rm -rf .build App/Spektrafilm.xcodeproj
