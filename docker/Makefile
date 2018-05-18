CONTAINER_NAME = flow-preprocessing
BIOC_VERSION = 3.7

DOCKERFILE = Dockerfile

.PHONY: run image clean

image: install_base.R
	@sudo docker build -f $(DOCKERFILE) -t $(CONTAINER_NAME) ../

install_base.R: install_base.R.in
	@sed "s/BIOC_VERSION/$(BIOC_VERSION)/g" $< > $@

run:
	@sudo docker run -it --rm $(CONTAINER_NAME)

clean:
	@rm install.R
