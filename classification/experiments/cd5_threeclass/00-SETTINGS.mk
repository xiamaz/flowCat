# experiment set name
TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))_

GROUPS = CD5pos:CLL,MBL,Mantel,CLLPL;CD5neg:FL,LPL,Marginal;normal
METHOD = kfold:10
FILTERS = max_size:2000
TUBES = 1;2

PATTERN := initial_comp_normal_2018
