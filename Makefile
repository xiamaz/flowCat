DATADIR = /data/flowcat-data/mll-flowdata/decCLL-9F
DATAMETA = /data/flowcat-data/mll-flowdata/decCLL-9F/case_info_2018-12-15
OUTDIR = ~/flowCat/output/test-2019-08

DSNAME = dataset

ALLDATA = --input $(DATADIR) --meta $(DATAMETA)
TRAINDATA = --input $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/train
TESTDATA = --input $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/test

REFERENCE_DATA = $(OUTDIR)/$(DSNAME)/reference.json
REFERENCE_SOM = $(OUTDIR)/reference

SOM_DATASET = $(OUTDIR)/som

GROUPS = CLL,MBL,MCL,PL,LPL,MZL,FL,HCL,normal


.PHONY: dataset
.ONESHELL:
dataset:
	@echo "Executing 01a"
	./00a_dataset_prepare.py $(ALLDATA) $(OUTDIR)/$(DSNAME)
	echo "Executing 01b"
	./00b_select_ref_cases.py $(TRAINDATA) --sample 1 --groups $(GROUPS) --infiltration 40,None $(REFERENCE_DATA)

.PHONY: reference
.ONESHELL:
reference: reference.ref

.PHONY: %.ref
.ONESHELL:
%.ref:
	./01a_create_ref_som.py $(TRAINDATA) --tensorboard $(REFERENCE_DATA) $(OUTDIR)/$*

.PHONY: som
.ONESHELL:
som:
	@./01b_create_soms.py $(TRAINDATA) $(REFERENCE_SOM) $(SOM_DATASET)

.PHONY: %.som
.ONESHELL:
%.som:
	@./01b_create_soms.py --tensorboard $(TRAINDATA) $(REFERENCE_SOM) $(OUTDIR)/epoch_cmp/$*
