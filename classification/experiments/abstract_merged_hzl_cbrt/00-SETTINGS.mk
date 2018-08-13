TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;HZL;normal"
SIZE = --size 3000

SET = abstract
EXTRA_ARGS = --transform cbrt
