# Basic settings
DATA = ../output
REPORT = $(DATA)/report

.PHONY: report
report: # regenerate all figures and tables
	python3 -m report $(REPORT)
	cp templates/report.tex $(REPORT)
	cd $(REPORT) && latexmk -pdf report.tex && latexmk -c

.PHONY: numbers
numbers:
	python3 ./data_numbers.py --tubes 1,2 --plotting --plotdir figures/numbers s3://mll-flowdata/CLL-9F
	python3 ./experiment_numbers.py

.PHONY: figures
figures: # numbers
	cd figures && latexmk -pdf $@.latex && latexmk -pdf -c $@.latex
