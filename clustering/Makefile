NAME = clustering

BUCKET = mll-flowdata

BUCKET_URL = s3://$(BUCKET)

TEMP = /data/tmp

CASEINFO = $(BUCKET_URL)/case_info.json

REFCASES = selected_hashed_ids.json

PLOTDIR = plots

# CUSTOM config options, offloaded in config
TEMPLATE = experiments/template.mk

include $(TEMPLATE)

EXP_NAME ?= test.mk
TAG_NAME ?= default

EXP = experiments/$(EXP_NAME)

include $(EXP)

OUTNAME := $(TAG_NAME)_$(basename $(notdir $(lastword $(MAKEFILE_LIST))))

BIN = python3 -m clustering

USROPTS = \
	   --tubes $(TUBES) \
	   --num $(NUM) \
	   --upsampled $(UPSAMPLING_NUM) \
	   --pipeline $(METHOD) \
	   --prefit $(PREMETHOD) \
	   --pretrans $(TRANSMETHOD) \
	   --groups $(GROUPS) \
	   $(FLAGS)

OPTS = \
	   --temp $(TEMP) \
	   --bucketname $(BUCKET) \
	   --input $(CASEINFO) \
	   --plotdir $(PLOTDIR) \
	   $(USROPTS)

OUTPUT = s3://mll-flow-classification/clustering/$(OUTNAME)
# OUTPUT = ../classification/output/clustering/$(OUTNAME)

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

run-image: image
	docker run --runtime nvidia --mount type=bind,source=/data,target=/data $(NAME) EXP=experiments/miniseq.mk

ecr: image
	@eval $$(aws ecr get-login --no-include-email --region eu-central-1)
	docker tag clustering:latest 463789428463.dkr.ecr.eu-central-1.amazonaws.com/clustering:latest
	docker push 463789428463.dkr.ecr.eu-central-1.amazonaws.com/clustering:latest
