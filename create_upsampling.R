library(methods)
library(parallel)
library(data.table)
library(flowProc)

source("fsom.R")

add <- function(x ,y) { x + y }

FillMin <- function(col.names, param.df) {
  min.values <- param.df[param.df[, "name"] %in% col.names, "minRange"]
  return(min.values)
}

#' Merge parameters dataframe from multiple flowframes
#'
#' @examples
#' GetParameters(test.case)
#' param.df <- GetParameters(test.case)$df
GetParameters <- function(entry.list) {
  # get parameters from fcs files
  param.meta <- entry.list[[1]]@fcs@parameters@varMetadata
  param.dim <- entry.list[[1]]@fcs@parameters@dimLabels
  param.dfs <- lapply(entry.list, function(x) {
                        as(x@fcs@parameters, "data.frame")
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
  return(list(df = param.df, ad = param.ad))
}

CrossJoin <- function(cells, param.df) {
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
  }, cells)
  return(cells.merged)
}

#' @examples
#' AverageJoin(list(matrix(c(1,2,3,4), nrow=2), matrix(c(5,6,7,8), nrow=2)), data.frame())
#' AverageJoin(list(test.case[[1]]@fcs@exprs, test.case[[2]]@fcs@exprs), GetParameters(test.case))
AverageJoin <- function(cells, param.df) {
  cells.merged <- Reduce(function(x, y) {
                    if (nrow(x) == 0) {
                      x.a <- rbind(x,
                                 FillMin(colnames(x),
                                       param.df))
                    } else {
                      x.a <- matrix(apply(x, 2, function(x) {
                                     sum(x) / length(x)
                                 }), nrow = 1)
                      colnames(x.a) <- colnames(x)
                    }
                    if (nrow(y) == 0) {
                      y.a <- rbind(y,
                                 FillMin(colnames(y),
                                       param.df))
                    } else {
                      y.a <- matrix(apply(y, 2, function(x) {
                                     sum(x) / length(x)
                                 }), nrow = 1)
                      colnames(y.a) <- colnames(y)
                    }
                    res.x <- merge(x, y.a, by = NULL)
                    res.y <- merge(x.a, y, by = NULL)
                    res <- rbind(res.x, res.y)
                    return(res)
  }, cells)
  return(cells.merged)
}

CreateCellMatrix <- function(som, som.mappings, join.function, entry.list, param.df, markers) {
    cell.maps <- lapply(1:som$map$nNodes, function(node.num) {
                          # select currently needed node maping from each tube
                          node.mappings <- lapply(som.mappings, function(l) {
                                                    l[[node.num]]
                                             })
                          # select cells in these maps
                          cells <- mapply(function(cell.sel, entry) {
                                            entry@fcs@exprs[cell.sel, , drop = F]
                                             }, node.mappings, entry.list, SIMPLIFY = F)
                          ## different join operations selected in main function
                          cells.merged <- join.function(cells, param.df)
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
    cell.matrix <- cell.matrix[, param.df[, "name"]]
    return(cell.matrix)
}

#' Join Tubes using SOM clustering
#'
#' @param entry.list List of entries
#' @param seed Seed for random number generator
#' @param xdim Map width.
#' @param ydim Map height.
#' @param join.method Name of method for joining of tubes.
#' @return List with joined fcs frames.
#' @examples
#' JoinTubesOnSom(test.case)
#' res <- JoinTubesOnSom(test.case)
#' res <- JoinTubesOnSom(test.case, join.method = "average", xdim = 10, ydim = 10)
#' for (i in 10:100) { JoinTubesOnSom(test.case, i) }
#' JoinTubesOnSom(test.case, 24, xdim = 10, ydim = 10)
JoinTubesOnSom <- function(entry.list, seed = 42, xdim = 20, ydim = 20,
                           join.method = "average") {
  print(entry.list[[1]]@label)
  # get joined tubes
  tm <- system.time(ret <- flowProc::filter_flowFrame_majority(entry.list, threshold = 1.0))
  message("Flow frame maj ", tm[3])
  entry.list <- ret$entries
  markers <- ret$markers

  # configure the loading funciton
  loading.function <- function(entry, selection) {
    flowProc::process_single(entry, selection = selection, remove_margins = F, trans = "log")
  }

  loaded.all.channels <- lapply(entry.list, loading.function)
  tm <- system.time(params <- GetParameters(loaded.all.channels))
  message("Get parameters ", tm[3])
  param.df <- params$df
  param.ad <- params$ad

  # create a joined som from both files
  tm <- system.time(fcs.list <- lapply(entry.list, function(x) {
                       x <- loading.function(x, markers)
                       x@fcs[, markers]
                            }))
  message("Load fcs files ", tm[3])
  tm <- system.time(som <- CreateFsom(flowCore::flowSet(fcs.list), seed = seed, xdim = xdim, ydim = ydim))
  message("Create fsom ", tm[3])

  # get list of nodes with cell indices
  tm <- system.time(som.mappings <- lapply(fcs.list, function(fcs.frame) {
                       upsampled <- FlowSOM::NewData(som, fcs.frame)
                       mapping <- upsampled$map$mapping[, 1]
                       mapping <- lapply(1:som$map$nNodes, function(x) {
                                           which(mapping == x)
                                           })
                       return(mapping)
                             }))
  message("Create SOM mappings ", tm[3])

  # load fcs entries for mapping
  tm <- system.time(entry.list <- lapply(entry.list, function(x) {
                         loading.function(x)
                             }))
  message("Load fcs again for merging. ", tm[3])

  # get cell selections for each cluster from both files and merge them
  if (join.method == "average") {
    join.function <- AverageJoin
  } else if (join.method == "cross") {
    join.function <- CrossJoin
  }
  tm <- system.time(cell.matrix <- CreateCellMatrix(som, som.mappings, join.function, entry.list, param.df, markers))
  message("Create cell matrix ", tm[3])
  # get cells numbers in each node
  fcs <- new("flowFrame", exprs = as.matrix(cell.matrix),
             parameters = param.ad)
  entry <- entry.list[[1]]
  entry@tube_set <- c(1, 2)
  entry@fcs <- fcs
  return(entry)
}

#' Randomly select samples from each group for SOM
#'
#' The consensus fsom is used for upsampling of single cases.
#' Expected input form is a list of lists.
#'
#' @examples
#' som <- CreateConsensusFsom(list(normal = normal.joined, cll = cll.joined))
CreateConsensusFsom <- function(groups.list, sample.size = 20) {
  entry.list <- list()
  for (name in names(groups.list)) {
    cur.group <- groups.list[[name]]
    cur.sel <- sample(cur.group, min(length(cur.group), sample.size))
    entry.list <- c(entry.list, cur.sel)
  }
  som <- FsomFromEntries(entry.list)
  return(som)
}

# identification settings
kRunNumber <- 1
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
all.files <- GetDir(kPath, "LMD", cluster)
all.files <- remove_duplicates(all.files)
stopCluster(cluster)

# restrict the different types of materials we will consider
all.files <- FilterList(all.files, material = kMaterialSelection)

# Get all entries with first two entries
# separate files based on their tube_set
label.groups <- GroupBy(all.files, "label", num.threads = 12)
# only include labels with first two tubes included
label.groups <- lapply(label.groups, function(x) {
                         x[flowProc::FilterEntries(x, list(tube_set = c(1, 2)))]
                            })
# select only items with both tubes available
two.items <- sapply(label.groups, function(x) {
                      if (length(x) == 2) {
                        return(T)
                      } else {
                        return(F)
                      }
                            })
label.groups <- label.groups[two.items]

# select sample subset from each list for the consensus fsom
acc <- function(x, y) { slot(x[[1]], y) }
fsom.sample <- GroupBy(label.groups, group.on = "group", acc = acc, num.threads = 1)
# select a small sample for the reduction
fsom.sample <- lapply(fsom.sample, function(group.list) { sample(group.list, min(20, length(group.list))) })

# joining samples in each group
join.function <- function(entry) {
  JoinTubesOnSom(entry, join.method = "average")
}
fsom.sample.joined <- lapply(fsom.sample, function(samples) {
                               lapply(samples, join.function)
                            })

fsom.samples <- do.call(c, fsom.sample.joined)

result <- flowProc::FilterChannelMajority(fsom.samples, threshold = 0.9)
fsom.samples <- result$entries
selected.channels <- result$markers

consensus.fsom <- FsomFromEntries(fsom.samples)

saveRDS(consensus.fsom, file.path(kOutputPath, "consensus_fsom.rds"))

upsampled.list <- list()
for (label.entry in label.groups) {
  message("Processing ", label.entry[[1]]@label)
  entry.joined <- join.function(label.entry)
  message("Joined fcs")
  if (any(!selected.channels %in% flowCore::colnames(entry.joined@fcs))) {
    message("Skipping because some names are not contained in fsom")
    print(flowCore::colnames(entry.joined@fcs))
    next
  } else {
    entry.joined@fcs <- entry.joined@fcs[, selected.channels]
  }
  upsampled <- UpsampleCase(consensus.fsom, entry.joined)
  upsampled.list <- c(upsampled, upsampled.list)
}

mat <- UpsampledListToTable(upsampled.list)
SaveMatrix("../joined", "joined_all.csv", mat)
