TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
PATTERN := comp_pregating
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal"
SIZE = --size 2000
MODIFIERS = 
