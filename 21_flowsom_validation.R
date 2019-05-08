library(jsonlite)
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
    metadata <- file.path(directory, "case_info.json")
  }
  fromJSON(metadata)
}

get_dataset <- function(path, ...) {
  ds <- list(path=path)
  ds$meta <- get_samples(path, ...)
  ds
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
  # read.FCS(fcspath, dataset=1, ...)
  ReadInput(fcspath, compensate=F, transform=F, scale=T)
}

plot_dir <- "output/21-flowsom-validation"
dir.create(plot_dir, recursive=T)

ds_missing <- get_dataset("output/missing")
ds_subsample <- get_dataset("output/subsample")

# generate some test images
pfcs_mi <- get_fcs_path(1, 1, ds_missing)
pfcs_ss <- get_fcs_path(1, 1, ds_subsample)

fcs_mi <- read.FCS(pfcs_mi, dataset=1, transformation=F)
fcs_ss <- read.FCS(pfcs_ss, dataset=1, transformation=F)
plt <- autoplot(fcs_mi, "CD45-KrOr", "SS INT LIN", bins=64)
ggsave(file.path(plot_dir, "test_missing.png"), plot=plt)
plt <- autoplot(fcs_ss, "CD45-KrOr", "SS INT LIN", bins=64)
ggsave(file.path(plot_dir, "test_sample01.png"), plot=plt)
# -------------------------

# create a simple SOM and extract the weights
fsom_ss <- ReadInput(fcs_ss, transform=F, scale=T)
fsom_ss <- BuildSOM(fsom_ss)
weights_ss <- fsom_ss$map$codes

# TODO: check train args, visualize nodes, check the same in python som
