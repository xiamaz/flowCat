DATADIR = /data/flowcat-data/mll-flowdata/decCLL-9F
DATAMETA = /data/flowcat-data/mll-flowdata/decCLL-9F/case_info_2018-12-15
OUTDIR = ~/flowCat/output/test-2019-08

DSNAME = dataset

ALLDATA = --input $(DATADIR) --meta $(DATAMETA)
TRAINDATA = --input $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/train
TESTDATA = --input $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/test

GROUPS = CLL,MBL,MCL,PL,LPL,MZL,FL,HCL,normal


.PHONY: dataset
.ONESHELL:
dataset:
	@echo "Executing 01a"
	./00a_dataset_prepare.py $(ALLDATA) $(OUTDIR)/$(DSNAME)
	echo "Executing 01b"
	./00b_select_ref_cases.py $(TRAINDATA) --sample 3 --groups $(GROUPS) --infiltration 40,None $(OUTDIR)/$(DSNAME)/reference.json
