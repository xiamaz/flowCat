# experiment set name
TAG := $(shell basename $(dir $(lastword $(MAKEFILE_LIST))))
METHOD = kfold:10
TUBES = 1;2
PATTERN := initial_comp_normal_2018

GROUPS = --group "CD5pos:CLL,MBL,MCL,PL;CD5neg:FL,LPL,MZL;normal"
SIZE = --size 2000
MODIFIERS = 
