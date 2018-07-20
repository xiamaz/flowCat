TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))_

GROUPS = CLL;normal
METHOD = kfold:10
FILTERS = 
TUBES = 1;2
