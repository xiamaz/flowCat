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

fcs_to_som_weights <- function(fcs, ...) {
  fsom_ss <- ReadInput(fcs, transform=F, scale=T)
  fsom_ss <- BuildSOM(fsom_ss, ...)
  weights_df <- fsom_ss$map$codes
  colnames(weights_df) <- markernames(fcs)
  weights_df
}

som_weights_csv <- function(weights_df, path) {
  write.csv(weights_df, file=path)
}

create_rolling <- function() {
  values <- c()
  function(delta) {
    values <<- c(values, delta)[1:min(length(values) + 1, 11)]
    # values <<- c(values, delta)
    cat(delta, "Mean delta", mean(values), "\n")
  }
}


dataset <- get_dataset("output/4-flowsom-cmp/samples")

plot_dir <- "output/4-flowsom-cmp/flowsom-10-timing"
dir.create(plot_dir, recursive=T)

# -------------------------

num_cases <- nrow(dataset$meta)
start_time <- Sys.time()
rolling_avg <- create_rolling()
for (case_index in 1:num_cases) {
  for (tube in c(1, 2, 3)) {
    label <- get_label(case_index, dataset)
    out_path <- file.path(plot_dir, sprintf("%s_t%d.csv", label, tube))
    fcs <- get_fcs_data(case_index, tube, dataset)
    weights <- fcs_to_som_weights(fcs, xdim=10, ydim=10, codes=NULL, init=F)
    som_weights_csv(weights, out_path)
    end_time <- Sys.time()
    delta <- end_time - start_time
    rolling_avg(delta)
    start_time <- end_time
  }
}
q()

plot_dir <- "output/4-flowsom-cmp/flowsom-32"
dir.create(plot_dir, recursive=T)

# -------------------------

num_cases <- nrow(dataset$meta)
for (case_index in 1:num_cases) {
  for (tube in c(1, 2, 3)) {
    label <- get_label(case_index, dataset)
    out_path <- file.path(plot_dir, sprintf("%s_t%d.csv", label, tube))
    fcs <- get_fcs_data(case_index, tube, dataset)
    weights <- fcs_to_som_weights(fcs, xdim=32, ydim=32, codes=NULL, init=F)
    som_weights_csv(weights, out_path)
  }
}

# create a simple SOM and extract the weights
# weights_ss <- fcs_to_som_weights(fcs_ss)
# som_weights_csv(weights_ss, file.path(plot_dir, "sample_t1.csv"))

# -> further processing and comparison now in python
