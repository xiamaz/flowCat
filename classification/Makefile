# Basic settings
BUCKET = mll-flow-classification

DATA = output
# input settings
INPUT = $(DATA)/clustering

# output data options
OUTPUT = $(DATA)/classification

# use a specific experiment setup
TEMPLATE = experiments/template.mk
include $(TEMPLATE)

EXP ?= experiments/test.mk
include $(EXP)

# set to name of included experiment
NAME := $(TAG)$(basename $(notdir $(lastword $(MAKEFILE_LIST))))_dedup

BIN = python3 classify.py

OPTIONS = -n "$(NOTE)" \
		  -g "$(GROUPS)" \
		  -f "$(FILTERS)" \
		  -i $(INPUT) \
		  --pattern $(PATTERN) \
		  -o $(OUTPUT) \
		  -m $(METHOD) \
		  --tubes "$(TUBES)"

.PHONY: run sync download upload

run: download
	# always use newest results
	$(BIN) $(OPTIONS) $(NAME)

sync: # synchronize output folder containg all results
	aws s3 sync s3://$(BUCKET) output
	aws s3 sync output s3://$(BUCKET)

download:
	mkdir -p $(INPUT)
	aws s3 sync --exclude "*" --include "*$(PATTERN)*.csv" s3://$(BUCKET)/clustering $(INPUT)

upload: run
	aws s3 sync $(OUTPUT) s3://$(BUCKET)/classification
