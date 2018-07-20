TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))_

GROUPS = CLL;normal
METHOD = kfold:10
FILTERS = smallest
TUBES = 1;2
