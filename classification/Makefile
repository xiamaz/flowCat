NAME = containern

NOTE = Cloud deployment testing

BUCKET = mll-flow-classification
DATA = output
INDATA = $(DATA)/preprocess
OUTPUT = $(DATA)/classification

GROUPS = LPLM:LPL,Marginal;MCH:Mantel,CLLPL;CM:CLL,MBL;FL;normal;HZL

METHOD = kfold:5
FILTERS = smallest

BIN = python3 classify.py


.PHONY: all
all: upload

upload: run
	aws s3 cp --recursive $(OUTPUT) s3://$(BUCKET)/classification

run: download
	# always use newest results
	$(eval INPUT := $(INDATA)/$(shell ls $(INDATA) | sort -r | head -n 1))
	$(BIN) -n "$(NOTE)" -g "$(GROUPS)" -f "$(FILTERS)" -i $(INPUT) -o $(OUTPUT) -m $(METHOD) $(NAME)

download:
	aws s3 cp --recursive --exclude "*" --include "*.csv" s3://$(BUCKET) $(DATA)

image:
	$(MAKE) -C docker
