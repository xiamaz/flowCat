library(jsonlite)
library(flowCore)
library(FlowSOM)
library(ggcyto)

#' Return filepaths substructure from JSON metadata object
get_id_filepaths <- function(id, metadata) {
  if (is.character(id)) {
    idx <- match(id, metadata$id)
  } else {
    idx <- id
  }
  metadata[idx, "filepaths"][[1]]
}

#' Get Dataframe with label and columns of tubes
#'
#' @example
#' get_samples("output/missing")
get_samples <- function(directory, metadata) {
  if (missing(metadata)) {
    metadata <- file.path(directory, "metadata.json")
  }
  fromJSON(metadata)
}

get_dataset <- function(path, ...) {
  ds <- list(path=path)
  ds$meta <- get_samples(path, ...)
  ds
}

get_label <- function(idx, dataset) {
  dataset$meta[idx, "id"][[1]]
}

get_fcs_path <- function(id, tube, dataset) {
  filepaths <- get_id_filepaths(id, dataset$meta)
  file.path(dataset$path, filepaths["fcs"][[1]][["path"]][tube])
}

#' Read an fcs file for a single id from the provided metadata
#'
#' @param id Patient id
#' @param tube Tube number for the file
#' @param dataset List with $meta dataframe and $path for path to fcs files
#' @param ... Forwarded to read.FCS function
get_fcs_data <- function(id, tube, dataset, ...) {
  filepaths <- get_id_filepaths(id, dataset$meta)
  fcspath <- file.path(dataset$path, filepaths["fcs"][[1]][["path"]][tube])
  read.FCS(fcspath, dataset=1, transformation=F, ...)
  # ReadInput(fcspath, compensate=F, transform=F, scale=T)
}

fcs_to_som_weights <- function(fcs) {
  fsom_ss <- ReadInput(fcs, transform=F, scale=T)
  fsom_ss <- BuildSOM(fsom_ss)
  weights_df <- fsom_ss$map$codes
  colnames(weights_df) <- markernames(fcs)
  weights_df
}

som_weights_csv <- function(weights_df, path) {
  write.csv(weights_df, file=path)
}

plot_dir <- "output/4-flowsom-cmp/flowsom-samples"
dir.create(plot_dir, recursive=T)

dataset <- get_dataset("output/4-flowsom-cmp/samples")

# generate some test images with case 1 and tube 1
# pfcs_mi <- get_fcs_path(1, 1, dataset)
# cat(pfcs_mi)
# 
# fcs_mi <- read.FCS(pfcs_mi, dataset=1, transformation=F)
# fcs_ss <- read.FCS(pfcs_ss, dataset=1, transformation=F)
# plt <- autoplot(fcs_mi, "CD45-KrOr", "SS INT LIN", bins=64)
# ggsave(file.path(plot_dir, "test_missing.png"), plot=plt)
# plt <- autoplot(fcs_ss, "CD45-KrOr", "SS INT LIN", bins=64)
# ggsave(file.path(plot_dir, "test_sample01.png"), plot=plt)

# -------------------------

# load reference weights

num_cases <- nrow(dataset$meta)
for (case_index in 1:num_cases) {
  for (tube in c(1, 2, 3)) {
    label <- get_label(case_index, dataset)
    out_path <- file.path(plot_dir, sprintf("%s_t%d.csv", label, tube))
    fcs <- get_fcs_data(case_index, tube, dataset)
    weights <- fcs_to_som_weights(fcs)
    som_weights_csv(weights, out_path)
  }
}

# create a simple SOM and extract the weights
# weights_ss <- fcs_to_som_weights(fcs_ss)
# som_weights_csv(weights_ss, file.path(plot_dir, "sample_t1.csv"))

# -> further processing and comparison now in python
