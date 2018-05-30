NAME = flow-clustering

BUCKET = mll-flowdata

BUCKET_URL = s3://$(BUCKET)

TEMP = /data/flow
# OUTPUT = s3://mll-flow-classification/clustering/output
OUTPUT = output

CASEINFO = $(BUCKET_URL)/case_info.json

REFCASES = selected_hashed_ids.json

TUBES = "1;2"
NUM = 5
UPSAMPLING_NUM = 100

BIN = python3 -m clustering
OPTS = --temp $(TEMP) --bucketname $(BUCKET) --tubes $(TUBES) --input $(CASEINFO) --num $(NUM) --upsampled $(UPSAMPLING_NUM)

run:
	$(BIN) $(OPTS) --output $(OUTPUT)

run-selected: $(REFCASES)
	$(BIN) $(OPTS) --output $(OUTPUT)_selected --refcases $(REFCASES)

$(REFCASES):
	aws s3 cp $(BUCKET_URL)/$(REFCASES) .

image:
	docker build -t $(NAME) .
