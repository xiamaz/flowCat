CONTAINER_NAME = flow-classification
REGISTRY = 463789428463.dkr.ecr.eu-central-1.amazonaws.com
DOCKERFILE = docker/Dockerfile

# Basic settings
BUCKET = mll-flow-classification

DATA = output
# input settings
INPUT = $(DATA)/clustering
INPUT_TYPE = output_selected

# output data options
OUTPUT = $(DATA)/classification

# use a specific experiment setup
EXP = experiments/elu_tube12_large_only.mk
include $(EXP)

# set to name of included experiment
NAME := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))

BIN = python3 classify.py

OPTIONS = -n "$(NOTE)" \
		  -g "$(GROUPS)" \
		  -f "$(FILTERS)" \
		  -i $(INPUT) \
		  -o $(OUTPUT) \
		  -m $(METHOD) \
		  --tubes "$(TUBES)"

.PHONY: all run clean image image-run image-upload


run:
	# always use newest results
	$(eval INPUT := $(INPUT)/$(shell ls $(INPUT) | grep "$(INPUT_TYPE)" | sort -r | head -n 1))
	$(BIN) $(OPTIONS) $(NAME)

upload: # run
	aws s3 cp --recursive $(OUTPUT) s3://$(BUCKET)/classification
	# aws s3 cp --recursive $(INDATA) s3://$(BUCKET)/clustering

download:
	aws s3 cp --recursive --exclude "*" --include "*.csv" s3://$(BUCKET) $(DATA)

include ../docker.mk
