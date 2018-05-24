NAME = flow-clustering

BUCKET = mll-flowdata

TEMP = /data/flow
OUTPUT = s3://mll-flow-classification/clustering/output

CASEINFO = s3://mll-flowdata/case_info.json

REFCASES = ./selected_hashed_ids.json

TUBES = "1;2"

BIN = python3 -m clustering
OPTS = --temp $(TEMP) --bucketname $(BUCKET) --tubes $(TUBES) --input $(CASEINFO)

run:
	$(BIN) $(OPTS) --output $(OUTPUT)

run-selected:
	$(BIN) $(OPTS) --output $(OUTPUT)_selected --refcases $(REFCASES)

image:
	docker build -t $(NAME) .
