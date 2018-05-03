#!/bin/env/Rscript
install.packages("devtools")
install.packages("ggplot2")
install.packages("optparse")
# install bioconductor
source("https://bioconductor.org/biocLite.R")
biocLite()

# install flowcore
biocLite("flowCore")
biocLite("FlowSOM")
biocLite("flowDensity")

# install flowproc
devtools::install_github("xiamaz/flowProc")
