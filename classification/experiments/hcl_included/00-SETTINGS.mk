# experiment set name
TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2
SET = abstract

SIZE = --size 3000
