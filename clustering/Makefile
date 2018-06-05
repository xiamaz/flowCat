NAME = flow-clustering

BUCKET = mll-flowdata

BUCKET_URL = s3://$(BUCKET)

TEMP = tmp

CASEINFO = $(BUCKET_URL)/case_info.json

REFCASES = selected_hashed_ids.json

PLOTDIR = plots

# CUSTOM config options, offloaded in config
EXP = experiments/template.mk

include $(EXP)

OUTNAME := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))

BIN = python3 -m clustering

USROPTS = \
	   --tubes $(TUBES) \
	   --num $(NUM) \
	   --upsampled $(UPSAMPLING_NUM) \
	   --pipeline $(METHOD) \
	   --groups $(GROUPS) \
	   --prefit $(PREMETHOD) \
	   --pretrans $(TRANSMETHOD) \
	   $(PLOT)

OPTS = \
	   --temp $(TEMP) \
	   --bucketname $(BUCKET) \
	   --input $(CASEINFO) \
	   --plotdir $(PLOTDIR) \
	   $(USROPTS)

# OUTPUT = s3://mll-flow-classification/clustering/output
OUTPUT = ../classification/output/clustering/$(OUTNAME)

.PHONY: run run-selected tests image

run:
	$(BIN) $(OPTS) --output $(OUTPUT)

run-selected: $(REFCASES)
	$(BIN) $(OPTS) --output $(OUTPUT)_selected --refcases $(REFCASES)

tests:
	python3 -m unittest discover

$(REFCASES):
	aws s3 cp $(BUCKET_URL)/$(REFCASES) .

image:
	docker build -t $(NAME) .
