CREATE_SOM = ./create_upsampling.R

OUTPUT = output/preprocess
SIZE = 1

RUN = AWSTEST
NUM = 0

.PHONY: run clean
run:
	Rscript $(CREATE_SOM) -o $(OUTPUT) -s $(SIZE) $(RUN) $(NUM)

clean:
	rm -rf output/preprocess/$(NUM)_$(RUN)
