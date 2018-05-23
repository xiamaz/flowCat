NAME = flow-clustering

BUCKET = mll-flowdata

TEMP = /data/flow
OUTPUT = s3://mll-flow-classification/output

CASEINFO = s3://mll-flowdata/case_info.json

REFCASES = ./selected_hashed_ids.json

BIN = python3 -m clustering

run:
	$(BIN) --temp $(TEMP) --input $(CASEINFO) --output $(OUTPUT) --bucketname $(BUCKET)

run-selected:
	$(BIN) --temp $(TEMP) --input $(CASEINFO) --output $(OUTPUT)_selected --bucketname $(BUCKET) --refcases $(REFCASES)

image:
	docker build -t $(NAME) .
