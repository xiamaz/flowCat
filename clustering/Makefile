BUCKET = mll-flowdata

TUBE = 1

test: cases
	./test.py $<

cases: input.lst
	cat $< | xargs -i aws s3 cp s3://$(BUCKET)/{} $@/

input.lst: case_info.json
	./select_fcs.py $(TUBE)

case_info.json:
	aws s3 cp s3://$(BUCKET)/$@ $@
