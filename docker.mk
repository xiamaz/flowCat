.PHONY: image image-run image-upload

image: $(DOCKERFILE)
	@sudo docker build -f $(DOCKERFILE) -t $(CONTAINER_NAME) ../

image-run: image
	@sudo docker run -it --rm --env-file .env $(CONTAINER_NAME)

image-upload: image
	sudo sh -c "$$(aws ecr get-login --no-include-email --region eu-central-1)"
	sudo docker tag $(CONTAINER_NAME):latest $(REGISTRY)/$(CONTAINER_NAME):latest
	sudo docker push $(REGISTRY)/$(CONTAINER_NAME):latest
