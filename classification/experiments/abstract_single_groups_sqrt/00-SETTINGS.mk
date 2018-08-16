TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "HCL;FL;LPL;MZL;MCL;PL;CLL;MBL;normal"
SIZE = --size 3000

SET = abstract
EXTRA_ARGS = --transform sqrt
