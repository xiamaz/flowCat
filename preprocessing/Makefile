CREATE_SOM = ./create_upsampling.R

OUTPUT = s3://mll-flow-classification/preprocess
INPUT = s3://mll-flowdata
TEMP = output/cache
SIZE = 5
GROUP = 500

PROC = 4

RUN = AWSTEST

.PHONY: run clean image
run:
	Rscript $(CREATE_SOM) -o $(OUTPUT) -i $(INPUT) \
		-s $(SIZE) -g $(GROUP) -t $(TEMP) -p $(PROC) $(RUN)

clean:
	rm -rf output/preprocess/$(NUM)_$(RUN)

image:
	$(MAKE) -C docker
