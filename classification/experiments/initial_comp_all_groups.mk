TAG := $(basename $(notdir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2

GROUPS = --group "FL;HCL;HZLv;LPL;MZL;MCL;PL;CLL;MBL;normal"
MODIFIERS = 
SIZE = --size 2000
