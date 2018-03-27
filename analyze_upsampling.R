#!/usr/bin/env Rscript
library(flowProc)
library(optparse)
source("lib/fsom.R")
source("lib/visualization.R")

GetCorrectChannels <- function(fcs.files, som.names) {
  fcs.channels <- sapply(fcs.files, function(f){
                           n <- flowCore::colnames(ProcessSingle(f)@fcs)
                           all(som.names %in% n)
  })
  return(fcs.files[fcs.channels][[1]])
}

input.som.file <- make_option(c("--fsom", "-f"), type = "character", help = "Input som file")

input.fcs.directory <- make_option(c("--input", "-i"), type = "character", help = "Input data directory",
                                   default = "../mll_data")
output.directory <- make_option(c("--output", "-o"), type = "character", help = "Output directory for plots",
                                default = "output/visualization")
parser <- OptionParser(usage = "%prog [options] label",
                       option_list = list(input.som.file, input.fcs.directory, output.directory))
arguments <- parse_args(parser, positional_arguments = 1)
parsed.options <- arguments$options
label <- arguments$args[[1]]

all.files <- ReadDatasets(parsed.options$input, thread.num = 12)

# get all files with specified label
all.files <- all.files[FilterEntries(all.files, list(label = label))]

som.object <- readRDS(parsed.options$fsom)

som.names <- names(som.object$prettyColnames)
som.file <- GetCorrectChannels(all.files, som.names)

som.file <- LoadFunction(som.file, som.names)

upsampled <- FlowSOM::NewData(som.object, som.file@fcs)
print(upsampled)
