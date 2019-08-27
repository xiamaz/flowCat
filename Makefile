DATADIR = /data/flowcat-data/mll-flowdata/decCLL-9F
DATAMETA = /data/flowcat-data/mll-flowdata/decCLL-9F/case_info_2018-12-15
OUTDIR = ~/flowCat/output/0-munich-data

DSNAME = dataset

TRAINDATA = --data $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/train.json
TESTDATA = --data $(DATADIR) --meta $(OUTDIR)/$(DSNAME)/test.json

REFERENCE_DATA = $(OUTDIR)/$(DSNAME)/reference.json
REFERENCE_SOM = $(OUTDIR)/reference

SOM_DATASET = $(OUTDIR)/som

GROUPS = CLL,MBL,MCL,PL,LPL,MZL,FL,HCL,normal


.PHONY: dataset
.ONESHELL:
dataset:
	@echo "Executing 01a"
	./00a_dataset_prepare.py --data /data/flowcat-data/mll-flowdata/decCLL-9F --meta /data/flowcat-data/mll-flowdata/decCLL-9F/case_info_2018-12-15.json $(OUTDIR)/$(DSNAME)
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
som: som.som

.PHONY: %.som
.ONESHELL:
%.som:
	@./01b_create_soms.py $(TRAINDATA) $(REFERENCE_SOM) $(OUTDIR)/$*

.PHONY: %.somtb
.ONESHELL:
%.somtb:
	@./01b_create_soms.py --tensorboard $(TRAINDATA) $(REFERENCE_SOM) $(OUTDIR)/$*
