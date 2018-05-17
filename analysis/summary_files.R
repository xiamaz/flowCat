#!/bin/env Rscript
library(flowProc)
library(ggplot2)

kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
filters <- list(material = kMaterialSelection)

# datas <- ReadDatasets("../mll_data", thread.num = 4, filters = filters)
datas <- ReadDatasets("../mll_data", thread.num = 4)

groups <- GroupBy(datas, "group")

test <- lapply(names(groups), function(gname){
         message(gname)
         # message("Length all: ", length(groups[[gname]]))
         tubegroup <- groups[[gname]][FilterEntries(groups[[gname]], list(tube_set=c(1)))]
         message("In one tube: ", length(tubegroup))
         materials <- GroupBy(tubegroup, "material")
         test <- lapply(names(materials), function(x) {
                          xstr <- paste(x, ":", length(materials[[x]]))
                          message(xstr)
})
})
