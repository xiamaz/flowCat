TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2
SET = mblcll

GROUPS = --group "CM:CLL,MBL;normal"
SIZE = 
FILTERS = 
