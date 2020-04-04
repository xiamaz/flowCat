LAST_VERSION := $(shell git tag --sort=-creatordate | head -n 1)
DEV_VERSION := $(shell python setup.py --version)
NEXT_VERSION := $(shell echo $(DEV_VERSION) | sed 's/\.dev.*//')
WHEEL_FILE := ./dist/flowcat-$(NEXT_VERSION)-py3-none-any.whl

# Build packaged wheel for flowCat.
.PHONY: build
build:
	python3 setup.py sdist bdist_wheel
	cp ./dist/flowcat-$(DEV_VERSION)-py3-none-any.whl $(WHEEL_FILE)
	@$(shell echo "Version $(NEXT_VERSION)\n\n$$(git shortlog $(LAST_VERSION)..)" > ./dist/$(NEXT_VERSION).chglog)

# Will load the current project tree as a python library. Useful for making
# changes to flowCat.
.PHONY: devel
devel:
	python3 setup.py devel

# Remove build directories created by build.
.PHONY: clean
clean:
	rm -r build
	rm -r dist

# create a github release from current head and include the changelog
.PHONY: release
release: build
	@echo $(DEV_VERSION)
	@echo $(NEXT_VERSION)
	hub release create -a $(WHEEL_FILE) -F ./dist/$(NEXT_VERSION).chglog $(NEXT_VERSION)
