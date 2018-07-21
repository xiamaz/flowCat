# Basic settings
BUCKET = mll-flow-classification

DATA = output
# input settings
INPUT = $(DATA)/clustering

# output data options
OUTPUT = $(DATA)/classification


REPORT = $(DATA)/report

EXTRA_TAG = _dedup

# use a specific experiment setup
TEMPLATE = experiments/template.mk
include $(TEMPLATE)

EXP ?= experiments/test.mk
include $(EXP)

# set to name of included experiment
NAME := $(TAG)$(basename $(notdir $(lastword $(MAKEFILE_LIST))))$(EXTRA_TAG)

BIN = python3 classify.py

OPTIONS = --note "$(NOTE)" \
		  --pattern $(PATTERN) \
		  --method $(METHOD) \
		  --tubes "$(TUBES)" \
		  -i $(INPUT) \
		  -o $(OUTPUT) \
		  $(GROUPS) \
		  $(MODIFIERS) \
		  $(SIZE) \
		  $(INFILTRATION) \

.PHONY: run sync download upload report

run: download
	# always use newest results
	$(BIN) $(OPTIONS) $(NAME)

sync: # synchronize output folder containg all results
	aws s3 sync s3://$(BUCKET) output
	aws s3 sync output s3://$(BUCKET)

download:
	mkdir -p $(INPUT)
	aws s3 sync --exclude "*" --include "*$(PATTERN)*.csv" s3://$(BUCKET)/clustering $(INPUT)

upload:
	aws s3 sync --exclude "*" --include "*$(NAME)*" $(OUTPUT) s3://$(BUCKET)/classification

report: # regenerate all figures and tables
	python3 -m report $(REPORT)
	cp templates/report.tex $(REPORT)
	cd $(REPORT) && latexmk -pdf report.tex && latexmk -c
