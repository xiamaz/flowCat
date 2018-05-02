#!/bin/env/Rscript
install.packages("devtools")
# install bioconductor
source("https://bioconductor.org/biocLite.R")
biocLite()

# install flowcore
biocLite("flowCore")
biocLite("FlowSOM")

# install flowproc
devtools::install_github("xiamaz/flowProc")
