.PHONY: image preprocessing
image: preprocessing

preprocessing:
	$(MAKE) -C docker
