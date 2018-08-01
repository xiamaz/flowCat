NAME = clustering

BUCKET = mll-flowdata

BUCKET_URL = s3://$(BUCKET)

TEMP = /data/tmp

INPUT = $(BUCKET_URL)/CLL-9F

REFCASES = $(BUCKET_URL)/meta/selected_cases.txt

PLOTDIR = plots

# CUSTOM config options, offloaded in config
TEMPLATE = experiments/template.mk

include $(TEMPLATE)

EXP_NAME ?= test.mk
TAG_NAME ?= default

EXP = experiments/$(EXP_NAME)

include $(EXP)

OUTNAME := $(TAG_NAME)/$(basename $(notdir $(lastword $(MAKEFILE_LIST))))$(SUFFIX)

BIN = python3 -m clustering

USROPTS = \
	   --tubes $(TUBES) \
	   --fitnum $(NUM) \
	   --fitgroups $(FITGROUPS) \
	   --transnum $(UPSAMPLING_NUM) \
	   --transgroups $(TRANSGROUPS) \
	   --pipeline $(METHOD) \
	   --prefit $(PREMETHOD) \
	   --pretrans $(TRANSMETHOD) \
	   $(FLAGS)

OPTS = \
	   --temp $(TEMP) \
	   --input $(INPUT) \
	   --plotdir $(PLOTDIR) \
	   $(USROPTS)

OUTPUT = s3://mll-flow-classification/clustering/$(OUTNAME)

.PHONY: run
run:
	$(BIN) $(OPTS) --output $(OUTPUT)

# use selected cases provided in reference json file for consensus SOM
.PHONY: run-selected
run-selected:
	$(BIN) $(OPTS) --output $(OUTPUT)_selected --refcases $(REFCASES)

# run the experiment locally and output to a local folder
.PHONY: run-local
run-local:
	$(eval TEMP:=tmp)
	$(BIN) $(OPTS) --output output/clustering/$(OUTNAME)

.PHONY: run-image
run-image: image
	docker run --runtime nvidia --mount type=bind,source=/data,target=/data $(NAME) EXP=experiments/miniseq.mk
