TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal"
FILTERS = --size 2000
