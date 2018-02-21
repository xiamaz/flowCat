library(methods)
library(magrittr)
library(parallel)
library(data.table)
library(flowProc)

CreateFsom <- function(fs, meta.num = 10, seed = 42, xdim = 10, ydim = 10) {
  # fsom <- FlowSOM::ReadInput(fs,
  #                            compensate = FALSE,
  #                            transform = FALSE,
  #                            scale = TRUE)
  # fsom <- FlowSOM::BuildSOM(fsom)
  # fsom <- FlowSOM::BuildMST(fsom, tSNE = FALSE)
  fsom <- FlowSOM::FlowSOM(fs,
                           colsToUse = flowCore::colnames(fs),
                           compensate = F,
                           transform = F,
                           scale = T,
                           seed = seed,
                           nClus = meta.num,
                           xdim = xdim,
                           ydim = ydim
                           )
  return(fsom)
}

FillMin <- function(col.names, param.df) {
  min.values <- param.df[param.df[,'name'] %in% col.names, 'minRange']
  return(min.values)
  # return(vector("double", length(col.names)))
}


FsomFromList <- function(fs) {
  return(CreateFsom(flowCore::flowSet(fs)))
}

CreateMetaclustering <- function(fsom, metanum) {
  meta <- FlowSOM::metaClustering_consensus(fsom$map$codes, k = metanum)
  return(meta)
}

# Return NA if filter requirements have not been fulfilled
SelectOnly <- function(entry, filters) {
  # check if all criteria in filters, a named list of values is fulfilled
  for (slot.name in names(filters)) {
    if (!slot(entry, slot.name) %in% filters[[slot.name]]) {
      return(F)
    }
  }
  return(T)
}

#' @examples
#' SelectInList(label.groups[[1]], list(tube_set = c(1,2)))
SelectInList <- function(file.list, filters) {
  return(file.list[sapply(file.list, function(x) {
                     SelectOnly(x, filters)
                             })
  ])
}

#' @examples
#' all.files[FilterOnAttr(all.files, list(group='CLL'))]
FilterOnAttr <- function(entry.data, filter.attr) {
  sapply(entry.data, function(x) {
           for (n in names(filter.attr)) {
             if (!(slot(x, n) == filter.attr[[n]])) {
               return(F)
             }
           }
           return(T)
                             })
}

#' @examples
#' GroupBy(all.files, 'label')
#' GroupBy(all.files, 'label', num.threads=12)
GroupBy <- function(upsampled.list, group.on, num.threads = 1) {
  group.names <- unique(sapply(upsampled.list, function(x) {
                                 slot(x, group.on)
  }))
  if (num.threads > 1) {
    lapply.func <- function(x, fun) {
      cl <- parallel::makeCluster(num.threads, type = "FORK")
      resp <- parallel::parLapply(cl = cl, x, fun)
      parallel::stopCluster(cl)
      return(resp)
    }
  } else {
    lapply.func <- lapply
  }
  # group all files into list of lists
  groups <- lapply.func(group.names,
                   function(x) {
                     f <- list(x)
                     names(f) <- group.on
                     upsampled.list[FilterOnAttr(upsampled.list, f)]
                   })
  return(groups)
}

Upsampled <- setClass("Upsampled",
                            contains = "FlowEntry",
                            representation(histo = "vector",
                                           meta = "vector"))

# identification settings
kRunNumber <- 0
kRunName <- "JoinedTubes"

# number of cpu threads for parallel processes
kThreads <- 12
# directory containing files to be processed
kPath <- "../Moredata"
# specify and create the output path containing plots and output data
kOutputPath <- sprintf("../%s_%d_output", kRunName, kRunNumber)
dir.create(kOutputPath, recursive = T, showWarnings = F)

# group size for inclusion in the flowSOM
kThresholdGroupSize <- 100
kMaterialSelection <- c("1", "2", "3", "4", "5", "PB", "KM")
# number of metaclusters in the flowSOM
kMetaNumber <- 10


cluster <- makeCluster(kThreads, type = "FORK")
all.files <- get_dir(kPath, "LMD", cluster)
all.files <- remove_duplicates(all.files)
stopCluster(cluster)

# restrict the different types of materials we will consider
all.files <- filter_list(all.files, material = kMaterialSelection)

# get all set names
# using the first two tubes at first only

# separate files based on their tube_set
label.groups <- GroupBy(all.files, "label", num.threads = 12)
# only include labels with first two tubes included
label.groups <- lapply(label.groups, function(x) {
                         SelectInList(x, list(tube_set = c(1, 2)))
                            })
# select only items with both tubes available
two.items <- sapply(label.groups, function(x) {
                      if (length(x)) {
                        return(T)
                      } else {
                        return(F)
                      }
                            })
label.groups <- label.groups[two.items]

#' @examples
#' JoinTubesOnSom(test.case)
#' for (i in 10:100) { JoinTubesOnSom(test.case, i) }
#' JoinTubesOnSom(test.case, 24, xdim = 10, ydim = 10)
JoinTubesOnSom <- function(entry.list, seed = 42, xdim = 20, ydim = 20) {
  # get joined tubes
  ret <- flowProc::filter_flowFrame_majority(entry.list, threshold = 1.0)
  entry.list <- ret$entries
  markers <- ret$markers
  # create a joined som from both files
  fcs.list <- lapply(entry.list, function(x) {
                       x <- flowProc::process_single(x, selection = markers,
                                                     remove_margins = F)
                       x@fcs[, markers]
                            })
  res.som <- CreateFsom(flowCore::flowSet(fcs.list), seed = seed, xdim = xdim,
                        ydim = ydim)
  som <- res.som$FlowSOM
  # get list of nodes with cell indices
  som.mappings <- lapply(entry.list, function(x) {
                       x <- flowProc::process_single(x,
                                                     selection = markers,
                                                     remove_margins = F)
                       upsampled <- FlowSOM::NewData(som, x@fcs)
                       mapping <- upsampled$map$mapping[, 1]
                       mapping <- lapply(1:som$map$nNodes, function(x) {
                                           selected <- which(mapping == x)
                                           })
                       return(mapping)
                             })
  # load fcs entries for mapping
  entry.list <- lapply(entry.list, function(x) {
                         flowProc::process_single(x, remove_margins = F)
                             })
  # get parameters from fcs files
  param.meta <- entry.list[[1]]@fcs@parameters@varMetadata
  param.dim <- entry.list[[1]]@fcs@parameters@dimLabels
  param.dfs <- lapply(entry.list, function(x) {
                        p.df <- as(x@fcs@parameters, "data.frame")
                             })
  param.df <- Reduce(function(x, y) {
                       y.diff <- y[!(y[, "name"] %in% x[, "name"]), ]
                       rbind(x, y.diff)
                             },
                             param.dfs)
  # continuous p naming
  rownames(param.df) <- paste("$P", 1:nrow(param.df), sep = "")
  # create annotated dataframe again for the flowframe creation
  param.ad <- Biobase::AnnotatedDataFrame(param.df, param.meta, param.dim)
  # get cell selections for each cluster from both files and merge them
  cell.maps <- lapply(1:som$map$nNodes, function(x) {
                        l <- lapply(som.mappings, function(l) {
                                      l[[x]]
                                           })

                        cells <- mapply(function(cell.sel, entry) {
                                          entry@fcs@exprs[cell.sel, , drop = F]
                                           }
                        , l, entry.list, SIMPLIFY = F)
                        cells.merged <- Reduce(function(x, y) {
                                          if (nrow(x) == 0) {
                                            x <- rbind(x,
                                                       FillMin(colnames(x),
                                                             param.df))
                                          }
                                          if (nrow(y) == 0) {
                                            y <- rbind(y,
                                                       FillMin(colnames(y),
                                                             param.df))
                                          }
                                          res <- merge(x, y, by = NULL)
                                                 return(res)
                        }
                        , cells)

                        # merge duplicate columns by averaging them
                        for (name in markers) {
                          cur.names <- colnames(cells.merged)
                          sel.names <- cur.names[grepl(name, cur.names)]
                          sel.cols <- lapply(sel.names, function(x) {
                                               cells.merged[, x]
                        })
                          # remove columns being merged
                          cells.merged <- cells.merged[,
                                                       !(colnames(cells.merged)
                                                           %in% sel.names)]
                          # set the merged column to the average
                          cells.merged[name] <- do.call(add, sel.cols) /
                            length(sel.cols)
                        }
                          return(cells.merged)
                             })
  cell.matrix <- do.call(rbind, cell.maps)
  # reorder the columns based on the parameter dataframe
  cell.matrix <- cell.matrix[, param.df[, "name"]]
  # get cells numbers in each node
  fcs <- new("flowFrame", exprs = as.matrix(cell.matrix),
             parameters = param.ad)
  entry <- entry.list[[1]]
  entry@tube_set <- c(1, 2)
  entry@fcs <- fcs
  return(entry)
}

# Concerning performance
library(microbenchmark)
test.case <- label.groups[[1]]
test.loaded <- lapply(test.case, function(x) {
                        process_single(x, remove_margins = F)
             })
# Goal: Plot cell number vs execution time
test.100 <- JoinTubesOnSom(test.case,
                              xdim = 10,
                              ydim = 10)
test.400 <- JoinTubesOnSom(test.case,
                              xdim = 20,
                              ydim = 20)

# saving joined fcs file to another file
test.joined <- test.400
joined.path <- '../joined'
group.path <- file.path(joined.path, test.joined@group)
dir.create(group.path, showWarnings = F)
fcs.path <- file.path(group.path, paste(test.joined@label, ".fcs", sep = ""))
flowCore::write.FCS(test.joined@fcs, fcs.path)
# exprs are not saved correct probably because the metadata is missing
reloaded <- flowCore::read.FCS(fcs.path)
