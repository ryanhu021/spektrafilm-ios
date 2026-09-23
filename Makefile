.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT := App/SpektrafilmApp.xcodeproj
SCHEME  := Spektrafilm
BUNDLE  := dev.ryanhu.spektrafilm
APP     := .build/xcode/Build/Products/Release-iphonesimulator/SpektrafilmApp.app

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
	swift format --in-place --recursive Sources Tests App/Spektrafilm Tools Package.swift

.PHONY: lint
lint: ## Check formatting without writing
	swift format lint --strict --recursive Sources Tests App/Spektrafilm Tools Package.swift

.PHONY: project
project: ## Generate App/SpektrafilmApp.xcodeproj from project.yml
	cd App && xcodegen generate

.PHONY: app
app: project ## Build the iOS app for the simulator
	xcodebuild build -quiet -project $(PROJECT) -scheme $(SCHEME) \
		-destination 'generic/platform=iOS Simulator' -derivedDataPath .build/xcode

# Release, because Debug keeps bounds checks on every buffer access and renders 40x slower.
.PHONY: run
run: project ## Build in release and launch on the booted simulator, with the sample scene
	xcodebuild build -quiet -project $(PROJECT) -scheme $(SCHEME) -configuration Release \
		-destination 'generic/platform=iOS Simulator' -derivedDataPath .build/xcode
	xcrun simctl install booted $(APP)
	xcrun simctl launch booted $(BUNDLE) -sample

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
	Tools/check_docs.py

.PHONY: clean
clean: ## Remove build products
	rm -rf .build App/SpektrafilmApp.xcodeproj
