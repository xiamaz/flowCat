TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
TUBES = 1;2
METHOD = kfold:10

GROUPS = --group "CLL;normal"
MODIFIERS =  --modifiers smallest
