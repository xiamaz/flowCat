# Build packaged wheel for flowCat.
build:
	python3 setup.py sdist bdist_wheel

# Will load the current project tree as a python library. Useful for making
# changes to flowCat.
devel:
	python3 setup.py devel

# Remove build directories created by build.
clean:
	rm -r build
	rm -r dist
