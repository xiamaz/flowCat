library(jsonlite)
library(flowCore)
library(FlowSOM)

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

#' Read an fcs file for a single id from the provided metadata
#'
#' @param id Patient id
#' @param tube Tube number for the file
#' @param metadata Metadata structure from json
#' @param path Base path for fcs files
#' @param ... Forwarded to read.FCS function
get_fcs_data <- function(id, tube, metadata, path, ...) {
  filepaths <- get_id_filepaths(id, metadata)
  fcspath <- file.path(path, filepaths["fcs"][[1]][["path"]][tube])
  # read.FCS(fcspath, dataset=1, ...)
  ReadInput(fcspath, compensate=F, transform=F, scale=T)
}

path <- "output/missing"
metadata <- get_samples(path)

fcs <- get_fcs_data(1, 1, metadata, path)
