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
NAME := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))

BIN = python3 classify.py

OPTIONS = -n "$(NOTE)" \
		  -g "$(GROUPS)" \
		  -f "$(FILTERS)" \
		  -i $(INPUT) \
		  --pattern $(PATTERN) \
		  -o $(OUTPUT) \
		  -m $(METHOD) \
		  --tubes "$(TUBES)"

.PHONY: all run clean

run:
	# always use newest results
	$(BIN) $(OPTIONS) $(NAME)

upload: # run
	aws s3 cp --recursive $(OUTPUT) s3://$(BUCKET)/classification
	# aws s3 cp --recursive $(INDATA) s3://$(BUCKET)/clustering

sync: # synchronize output folder containg all results
	aws s3 sync s3://$(BUCKET) output

download:
	aws s3 cp --recursive --exclude "*" --include "*.csv" s3://$(BUCKET) $(DATA)
