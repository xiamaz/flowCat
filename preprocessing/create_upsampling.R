#!/usr/bin/env Rscript
library(methods)
library(parallel)
library(data.table)
library(flowProc)
library(optparse)

source("lib/fsom.R")
source("lib/utils.R")

kTextNote <- c(
               "Cloud adaptations for running in AWS."
               )
kTextNote <- paste(kTextNote, sep = "")

threads.option <- make_option(c("--processes", "-p"), type = "numeric", default = 1,
                              help = "Number of threads", metavar = "processes")
dataset.option <- make_option(c("--dataset", "-d"), action = "store_true", default = F, help = "Read single dataset.")
inputpath.option <- make_option(c("--input", "-i"), type = "character",
                                default = "../Moredata", help = "Input directory", metavar = "input")
outputpath.option <- make_option(c("--output", "-o"), type = "character",
                                default = "output/preprocess", help = "Output directory", metavar = "output")
temppath.option <- make_option(c("--temp", "-t"), type = "character",
                              default = "output/s3cache", help = "Temp directory for caching", metavar = "temp")
groupsize.option <- make_option(c("--somnum", "-s"), type = "numeric",
                                default = 7, help = "Threshold group size for fSOM generation for each cohort.",
                                metavar = "somsize")
limitsize.option <- make_option(c("--groupsize", "-g"), type = "numeric",
                                default = 100, help = "Threshold group size for upsampling.",
                                metavar = "groupsize")

option.list <- list(threads.option, inputpath.option, dataset.option, outputpath.option, groupsize.option, temppath.option, limitsize.option)

parser <- OptionParser(usage = "%prog [options] run.name", option_list = option.list)
arguments <- parse_args(parser, positional_arguments = 1)
parsed.options <- arguments$options
parsed.args <- arguments$args

# identification settings
kRunName <- parsed.args[[1]]

# number of cpu threads for parallel processes
kThreads <- parsed.options$processes
# directory containing files to be processed
kInputPath <- parsed.options$input

kTempPath <- parsed.options$temp

kInfoFile <- "case_info.json"

# group size for inclusion in the flowSOM
kThresholdGroupSize <- parsed.options$somsize
kUpsamplingGroupSize <- parsed.options$groupsize

# selectors for filtering of cases
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
kGroupSelection <- c("CLL", "MBL", "normal", "Marginal", "CLLPL", "LPL", "HZL", "Mantel", "FL", "DLBCL")

# general filters to all files
filters <- list(material = kMaterialSelection, group = kGroupSelection)

# get case descriptions from json file
info.path <- GetFile(kInfoFile, kInputPath, kTempPath)
group.files <- ReadDatasetJson(info.path, kInputPath, kTempPath)

# set high cutoff to reduce runtime for large cohorts
group.files <- lapply(group.files, function(group) {
                        group <- sample(group, min(length(group), kUpsamplingGroupSize))
                        # flatten list
                        group <- do.call(c, group)
                        return(group)
                                })

all.files <- do.call(c, group.files)

# specify and create the output path containing plots and output data
kRunNumber <- strftime(Sys.time(), "%Y%m%d_%H%M")
kOutputPath <- file.path(parsed.options$output, sprintf("%s_%s", kRunName, kRunNumber))

CreateOutputDirectory(kTextNote, kOutputPath, kTempPath)

# create som csvs
tube.matrix.1 <- CasesToMatrix(all.files, thread.num = kThreads, filters = list(tube_set = c(1)),
                               output.dir = kOutputPath, name = "tube1",
                               temp.dir = kTempPath, sample.size = kThresholdGroupSize)
PutFile(tube.matrix.1, "tube1.csv", kOutputPath, SaveMatrix, kTempPath)

tube.matrix.2 <- CasesToMatrix(all.files, thread.num = kThreads, filters = list(tube_set = c(2)),
                               output.dir = kOutputPath, name = "tube2",
                               temp.dir = kTempPath, sample.size = kThresholdGroupSize)
PutFile(tube.matrix.2, "tube2.csv", kOutputPath, SaveMatrix, kTempPath)
