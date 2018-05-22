BUCKET = mll-flowdata

CACHE = /data/flow

all: tube1.csv tube2.csv

%.csv: case_info.json
	python3 clustering.py $< $(CACHE)

case_info.json:
	aws s3 cp s3://$(BUCKET)/$@ $@
