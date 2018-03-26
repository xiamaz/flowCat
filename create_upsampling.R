#!/usr/bin/env Rscript
library(methods)
library(parallel)
library(data.table)
library(flowProc)
library(optparse)

source("lib/fsom.R")
source("lib/utils.R")

kTextNote <- c(
               "Three class classification with CLL MBL and normal.",
               "This serves as comparison to the binary classifications between mbl, cll and normal results."
               )
kTextNote <- paste(kTextNote, sep = "")

threads.option <- make_option(c("--threads", "-t"), type = "numeric", default = 1,
                              help = "Number of threads", metavar = "threads")
dataset.option <- make_option(c("--dataset", "-d"), action = "store_true", default = F, help = "Read single dataset.")
inputpath.option <- make_option(c("--input", "-i"), type = "character",
                                default = "../Moredata", help = "Input directory", metavar = "input")
outputpath.option <- make_option(c("--output", "-o"), type = "character",
                                default = "output/preprocess", help = "Output directory", metavar = "output")
groupsize.option <- make_option(c("--groupsize", "-s"), type = "numeric",
                                default = 20, help = "Threshold group size for fSOM generation.",
                                metavar = "groupsize")

option.list <- list(threads.option, inputpath.option, dataset.option, outputpath.option, groupsize.option)

parser <- OptionParser(usage = "%prog [options] run.name run.number", option_list = option.list)
arguments <- parse_args(parser, positional_arguments = 2)
parsed.options <- arguments$options
parsed.args <- arguments$args

# identification settings
kRunName <- parsed.args[[1]]
kRunNumber <- as.integer(parsed.args[[2]])

# number of cpu threads for parallel processes
kThreads <- parsed.options$threads
# directory containing files to be processed
kPath <- parsed.options$input
# specify and create the output path containing plots and output data
kOutputPath <- file.path(parsed.options$output, sprintf("%d_%s", kRunNumber, kRunName))

# group size for inclusion in the flowSOM
kThresholdGroupSize <- parsed.options$groupsize

# selectors for filtering of cases
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
kGroupSelection <- c("CLL", "MBL", "normal")

# general filters to all files
filters <- list(material = kMaterialSelection, group = kGroupSelection)

data.reader <- ifelse(parsed.options$dataset, ReadDataset, ReadDatasets)
all.files <- data.reader(kPath, thread.num = kThreads, filters = filters)

# randomly sample the larger cohort down to the smaller one
group.files <- GroupBy(all.files, "group", num.threads = kThreads)

group.files <- lapply(group.files, function(group){
                        group <- cGroupBy(group, "label", c(1, 2))
                        return(group)
                                })
minimal.size <- min(sapply(group.files, length))
group.files <- lapply(group.files, function(group) {
                        group <- sample(group, minimal.size)
                        # flatten list
                        group <- do.call(c, group)
                        return(group)
                                })
all.files <- do.call(c, group.files)

# CREATE output directory
# annoying by design to prevent overwriting previous results
# save logging information
CreateOutputDirectory(kOutputPath, kTextNote)

tube.matrix.1 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(1)),
                               output.dir = kOutputPath, name = "tube1", sample.size = kThresholdGroupSize)
SaveMatrix(kOutputPath, "tube1.csv", tube.matrix.1)

tube.matrix.2 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(2)),
                               output.dir = kOutputPath, name = "tube2", sample.size = kThresholdGroupSize)
SaveMatrix(kOutputPath, "tube2.csv", tube.matrix.2)
