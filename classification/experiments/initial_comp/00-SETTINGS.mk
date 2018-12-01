TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "LMg:LPL,MZL;MtCp:MCL,PL;CM:CLL,MBL;normal"
SIZE = --size 2000
