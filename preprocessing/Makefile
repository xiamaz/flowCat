CREATE_SOM = ./create_upsampling.R

OUTPUT = s3://mll-flow-classification/preprocess
INPUT = s3://mll-flowdata
TEMP = output/cache
SIZE = 1
GROUP = 10

RUN = AWSTEST

AWS_ID = AKIAIFPSJDV55AEVQREA
AWS_SECRET = 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

.PHONY: run clean
run:
	AWS_ACCESS_KEY_ID=$(AWS_ID) \
					  AWS_SECRET_ACCESS_KEY=$(AWS_SECRET) \
					  Rscript $(CREATE_SOM) -o $(OUTPUT) -i $(INPUT) \
					  -s $(SIZE) -s $(GROUP) -t $(TEMP) $(RUN)

clean:
	rm -rf output/preprocess/$(NUM)_$(RUN)
