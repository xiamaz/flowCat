TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))_

GROUPS = FL;HZL;HZLv;LPL;Marginal;Mantel;CLLPL;CLL;MBL;normal
METHOD = kfold:10
FILTERS = max_size:2000
TUBES = 1;2
