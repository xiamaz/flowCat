TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))_

GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal
METHOD = kfold:10
FILTERS = max_size:2000
TUBES = 1;2

PATTERN := comp_pregating
