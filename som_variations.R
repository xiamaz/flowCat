#!/usr/bin/env Rscript
library(methods)
library(parallel)
library(data.table)
library(flowProc)

source("lib/fsom.R")
source("lib/utils.R")
source("lib/visualization.R")

kTextNote <- c(
               "Show effects of different number of cases in a generated SOM."
               )
kTextNote <- paste(kTextNote, sep = "")

# identification settings
kRunName <- "som_size_effects"
kRunNumber <- 1

# number of cpu threads for parallel processes
kThreads <- 12
# directory containing files to be processed
kPath <- "../mll_data"
# specify and create the output path containing plots and output data
kOutputPath <- file.path("output/visualization", sprintf("%d_%s", kRunNumber, kRunName))

# group size for inclusion in the flowSOM
kThresholdGroupSize <- 30

# selectors for filtering of cases
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
kGroupSelection <- c("CLL", "MBL", "normal", "Marginal", "CLLPL", "LPL", "HZL", "Mantel")

# general filters to all files
filters <- list(material = kMaterialSelection, group = kGroupSelection)

all.files <- ReadDatasets(kPath, thread.num = kThreads, filters = filters)

# randomly sample the larger cohort down to the smaller one
tube.files <- all.files[FilterEntries(all.files, list(tube_set=c(1)))]
group.files <- GroupBy(tube.files, "group", thread.num = kThreads)

group.files <- cGroupBy(tube.files, "group", vector())

# CREATE output directory
# annoying by design to prevent overwriting previous results
# save logging information
CreateOutputDirectory(kOutputPath, kTextNote)

for (minimal.size in c(10, 100)) {
  selected.files <- SampleGroups(group.files, sample.size = minimal.size)

  result <- FilterChannelMajority(selected.files, threshold = 0.8)
  filtered.files <- result$entries
  selected.channels <- result$markers

  l.func <- function(x) {
    LoadFunction(x, selected.channels)
  }

  all.filtered <- LoadCases(filtered.files, l.func, thread.num = 1)

  consensus.fsom <- FsomFromEntries(all.filtered)
  consensus.fsom <- FlowSOM::BuildMST(consensus.fsom, tSNE=T)
  meta.fsom <- CreateMetaclustering(consensus.fsom)

  name <- sprintf("%s/fsom_groupsize_%d", kOutputPath, minimal.size)
  PlotClustering(name, consensus.fsom, meta.fsom)
}
