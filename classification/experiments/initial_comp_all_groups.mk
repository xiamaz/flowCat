TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "FL;HZL;HZLv;LPL;Marginal;Mantel;CLLPL;CLL;MBL;normal"
MODIFIERS = 
SIZE = --size 2000
