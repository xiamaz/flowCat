#' Entry class containing results
#'
#' Most fields are inherited from FlowEntry. Will be used to save upsampling
#' results and facilitate conversion to the csv format.
Upsampled <- setClass("Upsampled", contains = "FlowEntry", representation(histo = "vector"))

#' Create flowsom from flowFrame or flowSeries
#'
#' @param fs Flowframe or flowSeries for fsom creation
#' @param seed Seed for random number generator
#' @param xdim Width of grid map
#' @param ydim Height of grid map
#' @return FSOM object
#' @examples
#' CreateFsom(test.entry@fcs)
CreateFsom <- function(fs, seed = 42, xdim = 10, ydim = 10) {
  set.seed(seed)
  fsom <- FlowSOM::ReadInput(fs,
                             compensate = FALSE,
                             transform = FALSE,
                             scale = TRUE)
  fsom <- FlowSOM::BuildSOM(fsom, xdim = xdim, ydim = ydim)
  return(fsom)
}

#' Create FlowSOM from list of flowframes
FsomFromList <- function(fs) {
  return(CreateFsom(flowCore::flowSet(fs)))
}

FsomFromEntries <- function(entry.list) {
  fcs.list <- lapply(entry.list, function(x) { x@fcs })
  return(FsomFromList(fcs.list))
}

#' Create meta clusters for existing fsom
#'
#' @param fsom FSOM object
#' @param meta.num Number of clusters for metaclustering
CreateMetaclustering <- function(fsom, meta.num = 10) {
  meta <- FlowSOM::metaClustering_consensus(fsom$map$codes, k = meta.num)
  return(meta)
}

#' Upsample a single entry and return upsampled object with histogram slots
#'
#' @param fsom SOM for upsampling of entry.
#' @param entry File object being upsampled.
#' @return Upsampled entry with cell distribution in histogram.
UpsampleCase <- function(fsom, entry) {
  upsampled <- FlowSOM::NewData(fsom, entry@fcs)
  cell.dist <- tabulate(upsampled$map$mapping, nbins = upsampled$map$nNodes)
  cell.dist.rel <- cell.dist / sum(cell.dist)

  entry.upsampled <- Upsampled(entry, histo = cell.dist.rel)
  return(entry.upsampled)
}

#' Create matrix representation from upsampled list
#'
#' @param upsampled.list List of upsampled S4 objects with cell occurrences in
#' histogram
#' @return Matrix with entries as rows with additional columns for label, group
#' for identification
UpsampledListToTable <- function(upsampled.list) {
  up.matrix <- lapply(upsampled.list, function(up) {
           ret <- up@histo
           ret <- c(ret, up@group, up@label)
           names(ret) <- c(1:length(up@histo), "group", "label")
           return(ret)
                             })
  up.matrix <- do.call(rbind, up.matrix)
  return(up.matrix)
}

#' Save matrix to csv file
#'
#' @param folder.path Folder for csv file.
#' @param file.name Filename.
#' @param up.matrix Data to be saved.
SaveMatrix <- function(folder.path, file.name, up.matrix) {
  dir.create(folder.path, showWarnings = F, recursive = T)
  path <- file.path(folder.path, file.name)
  write.table(up.matrix, file = path, sep = ";")
}
