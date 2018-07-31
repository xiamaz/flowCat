# Basic settings
DATA = ../output
REPORT = $(DATA)/report

.PHONY: report
report: # regenerate all figures and tables
	python3 -m report $(REPORT)
	cp templates/report.tex $(REPORT)
	cd $(REPORT) && latexmk -pdf report.tex && latexmk -c
