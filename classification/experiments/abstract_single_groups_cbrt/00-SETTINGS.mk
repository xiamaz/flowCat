TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "HZL;FL;LPL;Marginal;Mantel;CLLPL;CLL;MBL;normal"
SIZE = --size 3000

SET = abstract
EXTRA_ARGS = --transform cbrt
