CONTAINER_NAME = flow-classification
REGISTRY = 463789428463.dkr.ecr.eu-central-1.amazonaws.com
DOCKERFILE = docker/Dockerfile

# Program settings
NAME = containern
NOTE = Cloud deployment testing

BUCKET = mll-flow-classification
DATA = output
INDATA = $(DATA)/clustering
OUTPUT = $(DATA)/classification

GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;normal
# GROUPS = CLL;normal

METHOD = kfold:5
FILTERS = smallest

INPUT = output/clustering

BIN = python3 classify.py

.PHONY: all run clean image image-run image-upload


run: download
	# always use newest results
	$(eval INPUT := $(INDATA)/$(shell ls $(INDATA) | grep "output_selected" | sort -r | head -n 1))
	$(BIN) -n "$(NOTE)" -g "$(GROUPS)" -f "$(FILTERS)" -i $(INPUT) -o $(OUTPUT) -m $(METHOD) $(NAME)

upload: # run
	aws s3 cp --recursive $(OUTPUT) s3://$(BUCKET)/classification
	# aws s3 cp --recursive $(INDATA) s3://$(BUCKET)/clustering

download:
	aws s3 cp --recursive --exclude "*" --include "*.csv" s3://$(BUCKET) $(DATA)

include ../docker.mk
