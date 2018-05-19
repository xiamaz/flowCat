BUCKET = mll-flowdata

TUBE = 1

test: cases
	python3 example.py input.lst

cases: input.lst
	cat $< | xargs -i aws s3 cp s3://$(BUCKET)/{} $@/

input.lst: case_info.json
	./select_fcs.py $(TUBE)

case_info.json:
	aws s3 cp s3://$(BUCKET)/$@ $@
