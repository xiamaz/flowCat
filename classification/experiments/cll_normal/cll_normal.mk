TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))_

GROUPS = CLL:normal
METHOD = kfold:10
FILTERS = max_size:100
TUBES = 1;2
