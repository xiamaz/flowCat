CREATE_SOM = ./create_upsampling.R

OUTPUT = s3://mll-flow-classification/preprocess
INPUT = s3://mll-flowdata
TEMP = output/cache
SIZE = 1
GROUP = 10

RUN = AWSTEST

.PHONY: run clean
run:
	Rscript $(CREATE_SOM) -o $(OUTPUT) -i $(INPUT) \
		-s $(SIZE) -g $(GROUP) -t $(TEMP) $(RUN)

clean:
	rm -rf output/preprocess/$(NUM)_$(RUN)
