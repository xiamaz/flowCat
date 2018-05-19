.PHONY: image preprocessing classification
image: preprocessing classification

preprocessing:
	$(MAKE) -C preprocessing image

classification:
	$(MAKE) -C classification image
