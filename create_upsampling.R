#!/usr/bin/env Rscript
library(methods)
library(parallel)
library(data.table)
library(flowProc)
library(optparse)

source("lib/fsom.R")

threads.option <- make_option(c("--threads", "-t"), type = "numeric", default = 1,
                              help = "Number of threads", metavar = "threads")
dataset.option <- make_option(c("--dataset", "-d"), action = "store_true", default = F,
                              metavar = "dataset", help = "Read single dataset.")
inputpath.option <- make_option(c("--input", "-i"), type = "character",
                                default = "../Moredata", help = "Input directory", metavar = "input")
outputpath.option <- make_option(c("--output", "-o"), type = "character",
                                default = "output/preprocess", help = "Output directory", metavar = "output")
groupsize.option <- make_option(c("--groupsize", "-s"), type = "numeric",
                                default = 20, help = "Threshold group size for fSOM generation.",
                                metavar = "groupsize")

option.list <- list(threads.option, inputpath.option, outputpath.option, groupsize.option)

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
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")

print(kOutputPath)
stop()

# annoying by design to prevent overwriting previous results
if (dir.exists(kOutputPath)){
  stop(paste(kOutputPath, "already exists. Move or delete it."))
}
dir.create(kOutputPath, recursive = T, showWarnings = T)

all.files <- ifelse(parsed.options$dataset,
                    ReadDatasets(kPath, thread.num = kThreads, material = kMaterialSelection),
                    ReadDataset(kPath, thread.num = kThreads, material = kMaterialSelection))

tube.matrix.1 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(1)),
                               output.dir = kOutputPath, name = "tube1", sample.size = 10)#, load.func = loader)
SaveMatrix(kOutputPath, "tube1.csv", tube.matrix.1)

tube.matrix.2 <- CasesToMatrix(all.files, kThreads, filters = list(tube_set = c(2)),
                               output.dir = kOutputPath, name = "tube2", sample.size = 10)#, load.func = loader)
SaveMatrix(kOutputPath, "tube2.csv", tube.matrix.2)
