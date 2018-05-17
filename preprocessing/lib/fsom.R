#' Entry class containing results
#'
#' Most fields are inherited from FlowEntry. Will be used to save upsampling
#' results and facilitate conversion to the csv format.
Upsampled <- methods::setClass("Upsampled", contains = "FlowEntry", methods::representation(histo = "vector"))

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
UpsampleCase <- function(fsom, entry, sample.load.func) {
    message("Upsampling ", entry@label)
    loaded <- sample.load.func(entry)
    if (!isS4(loaded)) {
      print("Entry not valid. Skipping...")
      return(NA)
    }
  upsampled <- FlowSOM::NewData(fsom, loaded@fcs)
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
#' @param outpath Output file path.
SaveMatrix <- function(up.matrix, outpath) {
  write.table(up.matrix, file = outpath, sep = ";")
}


SampleGroups <- function(groups.list, sample.size = 20) {
  entry.list <- list()
  for (name in names(groups.list)) {
    cur.group <- groups.list[[name]]
    cur.sel <- sample(cur.group, min(length(cur.group), sample.size))
    entry.list <- c(entry.list, cur.sel)
  }
  return(entry.list)
}


#' Randomly select samples from each group for SOM
#'
#' The consensus fsom is used for upsampling of single cases.
#' Expected input form is a list of lists.
#'
#' @examples
#' som <- CreateConsensusFsom(list(normal = normal.joined, cll = cll.joined))
CreateConsensusFsom <- function(groups.list, sample.size = 20) {
  entry.list <- SampleGroups(groups.list)
  som <- FsomFromEntries(entry.list)
  return(som)
}

CreateLapply <- function(thread.num) {
  if (thread.num > 1) {
    lfunc <- function(x, y) {
      cluster <- parallel::makeCluster(thread.num, type = "FORK")
      result <- parallel::parLapply(cluster, x, y)
      parallel::stopCluster(cluster)
      return(result)
    }
  } else {
    lfunc <- lapply
  }
  return(lfunc)
}

LoadCases <- function(case.list, load.function, thread.num = 1) {
  lfunc <- CreateLapply(thread.num)
  return(lfunc(case.list, load.function))
}

LoadFunction <- function(x, selected) {
  return(flowProc::ProcessSingle(x, selected, trans = "log", remove_margins = F, upper = F, lower = F))
}

LoadFunctionBuilder <- function(selected, load.func) {
  return(function(x) {load.func(x, selected)})
}


#' Create Histogram matrices from fcs case files
#'
#' Directly create histograms from files separated for each tube.
CasesToMatrix <- function(entry.list, thread.num = 1, load.func = LoadFunction,
                          filters = list(tube_set = c(1)),
                          output.dir = "output/preprocess", temp.dir = "output/cache",
                          name = "", sample.size = 40) {
  # filter down to single property, eg the first tube
  entry.list <- entry.list[flowProc::FilterEntries(entry.list, filters)]
  message("Getting marker configuration in file set\n")

  result <- flowProc::FilterChannelMajority(entry.list, threshold = 0.8)
  entry.list <- result$entries
  selected.channels <- result$markers

  sample.load.func <- LoadFunctionBuilder(selected.channels, load.func)

  # create consensus fsom
  selected.samples <- flowProc::GroupBy(entry.list, "group", thread.num = thread.num)
  message("Create consensus FSOM")
  selected.samples <- SampleGroups(selected.samples, sample.size = sample.size)
  message("Loading cases")
  selected.samples <- LoadCases(selected.samples, sample.load.func, thread.num = thread.num)
  message("FSOM creation")
  consensus.fsom <- FsomFromEntries(selected.samples)
  fsom.name <- sprintf("stored_consensus_fsom_%s.rds", name)
  save.fsom <- function(dat, path) {saveRDS(dat, path, compress = F)}
  PutFile(consensus.fsom, fsom.name, output.dir, save.fsom, temp.dir)

  # upsample single cases
  upsampled.list <- lapply(entry.list, function(entry) {
    UpsampleCase(consensus.fsom, entry, sample.load.func)
                             })
  upsampled.list <- upsampled.list[!is.na(upsampled.list)]
  return(UpsampledListToTable(upsampled.list))
}
