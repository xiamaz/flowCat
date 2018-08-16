TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "LMg:LPL,MZL;MtCp:MCL,PL;CM:CLL,MBL;FL;HCL;normal"
SIZE = --size 3000

SET = abstract
EXTRA_ARGS = --transform cbrt
