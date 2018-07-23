TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "FL;HZL;HZLv;LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal"
SIZE = --size 2000
