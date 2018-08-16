TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "HCL;FL;LPL;MZL;MCL;PL;CLL;MBL;normal"
GROUPS = --group "LMg:LPL,MZL;MtCp:MCL,PL;CM:CLL,MBL;FL;normal"
SIZE = --size 3000

SET = abstract
