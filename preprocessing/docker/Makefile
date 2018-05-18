CONTAINER_NAME = flow-preprocessing

REGISTRY = 463789428463.dkr.ecr.eu-central-1.amazonaws.com

DOCKERFILE = Dockerfile

.PHONY: run image clean

image: install_base.R
	@sudo docker build -f $(DOCKERFILE) -t $(CONTAINER_NAME) ../

run:
	@sudo docker run -it --rm $(CONTAINER_NAME)

upload:
	docker tag flow-preprocessing:latest $(REGISTRY)/$(CONTAINER_NAME):latest
	docker push $(REGISTRY)/$(CONTAINER_NAME):latest

clean:
	@rm install.R
