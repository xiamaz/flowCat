# Basic settings
BUCKET = mll-flow-classification
DATA = output
REPORT = $(DATA)/report
EXTRA_TAG = _dedup

# use a specific experiment setup
TEMPLATE = experiments/template.mk
include $(TEMPLATE)
EXP ?= experiments/test.mk
include $(EXP)

# set to name of included experiment
NAME := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))$(EXTRA_TAG)

# input settings
INPUT = clustering/$(SET)
# output data options
OUTPUT = classification


# Build command strings
BIN = python3 classify.py

OPTIONS = --note "$(NOTE)" \
		  --pattern $(PATTERN) \
		  --method $(METHOD) \
		  --tubes "$(TUBES)" \
		  -i $(DATA)/$(INPUT) \
		  -o $(DATA)/$(OUTPUT) \
		  $(GROUPS) \
		  $(MODIFIERS) \
		  $(SIZE) \
		  $(INFILTRATION) \


# Targets
.PHONY: run sync download upload report
run: download
	# always use newest results
	$(BIN) $(OPTIONS) $(TAG)/$(NAME)

sync: # synchronize output folder containg all results
	aws s3 sync s3://$(BUCKET) output
	# aws s3 sync output s3://$(BUCKET)

download:
	mkdir -p $(DATA)/$(INPUT)
	aws s3 sync --exclude "*" --include "*$(PATTERN)*" s3://$(BUCKET)/$(INPUT) $(DATA)/$(INPUT)

upload:
	aws s3 sync --exclude "*" --include "*$(NAME)*" $(DATA)/$(OUTPUT)/$(TAG) s3://$(BUCKET)/$(OUTPUT)/$(TAG)

report: # regenerate all figures and tables
	python3 -m report $(REPORT)
	cp templates/report.tex $(REPORT)
	cd $(REPORT) && latexmk -pdf report.tex && latexmk -c
