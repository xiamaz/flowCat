TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "HZL;FL;LPL;Marginal;Mantel;CLLPL;CLL;MBL;normal"
GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;normal"
SIZE = --size 3000

SET = abstract
