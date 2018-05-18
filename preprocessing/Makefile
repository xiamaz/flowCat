CONTAINER_NAME = flow-preprocessing

REGISTRY = 463789428463.dkr.ecr.eu-central-1.amazonaws.com

DOCKERFILE = docker/Dockerfile

CREATE_SOM = ./create_upsampling.R

OUTPUT = s3://mll-flow-classification/preprocess
INPUT = s3://mll-flowdata
TEMP = output/cache
SIZE = 5
GROUP = 500

PROC = 4

RUN = AWSTEST

.PHONY: run image clean image-run image-upload

run:
	@Rscript $(CREATE_SOM) -o $(OUTPUT) -i $(INPUT) \
		-s $(SIZE) -g $(GROUP) -t $(TEMP) -p $(PROC) $(RUN)
clean:
	@rm -rf output/preprocess/$(NUM)_$(RUN)

image-run: image
	@sudo docker run -it --rm $(CONTAINER_NAME)

image-upload: image
	@docker tag flow-preprocessing:latest $(REGISTRY)/$(CONTAINER_NAME):latest
	@docker push $(REGISTRY)/$(CONTAINER_NAME):latest

image: $(DOCKERFILE)
	@sudo docker build -f $(DOCKERFILE) -t $(CONTAINER_NAME) ../
