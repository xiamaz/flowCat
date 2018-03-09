library(methods)
library(parallel)
library(data.table)
library(flowProc)

source("lib/fsom.R")

margin.list <- list(MarginUpperLower = c(T, T), MarginUpper = c(T, F), MarginLower = c(F, T))

for (kRunName in c("MarginLower")) {
# identification settings
kRunNumber <- 1
# kRunName <- "RemoveMarginUpperLower"

# number of cpu threads for parallel processes
kThreads <- 12
# directory containing files to be processed
kPath <- "/data/ssdraid/Genetik/Moredata"
# specify and create the output path containing plots and output data
kOutputPath <- sprintf("../%s_%d_output", kRunName, kRunNumber)
dir.create(kOutputPath, recursive = T, showWarnings = F)

# group size for inclusion in the flowSOM
kThresholdGroupSize <- 100
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
# number of metaclusters in the flowSOM - currently unused
# kMetaNumber <- 10

all.files <- CreateFileInfo(kPath,
                            thread.num = kThreads,
                            material = kMaterialSelection)

margins <- margin.list[[kRunName]]

loader <- function(x, selected) {
  return(flowProc::ProcessSingle(x, selected, trans = "log",
                                 remove_margins = T, upper = margins[1], lower = margins[2]))
}

tube.matrix.1 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(1)), load.func = loader)
SaveMatrix(kOutputPath, "native_tube1.csv", tube.matrix.1)

tube.matrix.2 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(2)), load.func = loader)
SaveMatrix(kOutputPath, "native_tube2.csv", tube.matrix.2)
}
