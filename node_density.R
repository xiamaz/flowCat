library(methods)
library(magrittr)
library(parallel)
library(data.table)
library(flowProc)

CreateFsom <- function(fs, meta.num = 10, seed = 42, xdim = 10, ydim = 10) {
  fsom <- FlowSOM::ReadInput(fs,
                             compensate = FALSE,
                             transform = FALSE,
                             scale = TRUE)
  fsom <- FlowSOM::BuildSOM(fsom, xdim = xdim, ydim = ydim)
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
  names(groups) = group.names
  return(groups)
}

#' @examples
#' JoinTubesOnSom(test.case)
#' for (i in 10:100) { JoinTubesOnSom(test.case, i) }
#' JoinTubesOnSom(test.case, 24, xdim = 10, ydim = 10)
JoinTubesOnSom <- function(entry.list, seed = 42, xdim = 20, ydim = 20) {
  print(entry.list[[1]]@label)
  # get joined tubes
  ret <- flowProc::FilterCommonChannels(entry.list, threshold = 1.0)
  entry.list <- ret$entries
  markers <- ret$markers
  # create a joined som from both files
  fcs.list <- lapply(entry.list, function(x) {
                       x <- flowProc::process_single(x, selection = markers,
                                                     remove_margins = F)
                       x@fcs[, markers]
                            })
  som <- CreateFsom(flowCore::flowSet(fcs.list), seed = seed, xdim = xdim,
                        ydim = ydim)
  # som <- res.som$FlowSOM
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

#' Randomly select samples from each group for SOM
#'
#' The consensus fsom is used for upsampling of single cases.
#' Expected input form is a list of lists.
#'
#' @examples
#' CreateConsensusFsom(list(normal = normal.joined, cll = cll.joined))
CreateConsensusFsom <- function(groups.list) {
  entry.list <- list()
  for (name in names(groups.list)) {
    cur.group <- group.list[[name]]
    cur.sel <- sample(cur.group, min(length(cur.group), 10))
    entry.list = c(entry.list, cur.sel)
  }
  print(length(entry.list))
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
label.groups <- GroupBy(all.files, "group", num.threads = 12)

# Analysis of the channel distributions
# steps:
# 1. fetch the signatures for the relevant channels from all nodes
# 2. plot the dsitribution of channel values for all nodoes
# 3. also plot the distribution of cells over these channels in comparison
plot.folder <- "../node_density/plots"

# interesting channels:
# FS, SS, CD45, CD19

#' @examples
#' PlotNodes(label.groups, "CLL", plot.folder)
PlotNodes <- function(files.list, group.name, plot.folder) {
  grouped.list <- GroupBy(label.groups[[group.name]], "label", num.threads = 6)
  sel.channels <- c("FS INT LIN", "SS INT LIN", "CD45-KrOr", "CD19-APCA750")
  grouped.sample <- sample(grouped.list, 10)
  for (name in names(grouped.sample)) {
    label.path <- file.path(plot.folder, group.name,  name)
    dir.create(label.path, showWarnings = F, recursive = T)
    tube.1 <- Reduce(function(x, y) {
             if (y@tube_set == 1) {
               return(y)
             } else {
               return(x)
             }
                              },
                              grouped.sample[[name]], NA)
    if (!isS4(tube.1))
      next
    tube.1 <- flowProc::process_single(tube.1, remove_margins = F, trans = "log")
    som <- CreateFsom(tube.1@fcs, xdim = 10, ydim = 10)
    cluster.values <- som$map$codes
    combn(sel.channels, 2, function(x) {
            desc.x <- strsplit(x[1], "[ -]")[[1]][1]
            desc.y <- strsplit(x[2], "[ -]")[[1]][1]
            jpeg(file.path(label.path, paste(desc.x, desc.y, ".jpg", sep = "")))
            plot(cluster.values[, x[1]], cluster.values[, x[2]],
                 xlab = x[1], ylab = x[2])
            dev.off()
                              })
  }
}

# group by label, thus getting all tubes for a single label
normal.grouped <- GroupBy(label.groups[["normal"]], "label", num.threads = 6)
cll.grouped <- GroupBy(label.groups[["CLL"]], "label", num.threads = 6)

# first tube 1 only
sel.channels <- c("FS INT LIN", "SS INT LIN", "CD45-KrOr", "CD19-APCA750")
normal.sample <- sample(normal.grouped, 10)
for (name in names(normal.sample)) {
  label.path <- file.path(plot.folder, "normal",  name)
  dir.create(label.path, showWarnings = F, recursive = T)
  tube.1 <- Reduce(function(x, y) {
           if (y@tube_set == 1) {
             return(y)
           } else {
             return(x)
           }
                            },
                            normal.sample[[name]], NA)
  if (!isS4(tube.1))
    next
  tube.1 <- flowProc::process_single(tube.1, remove_margins = F, trans = "log")
  som <- CreateFsom(tube.1@fcs, xdim = 10, ydim = 10)
  cluster.values <- som$map$codes
  combn(sel.channels, 2, function(x) {
          desc.x <- strsplit(x[1], "[ -]")[[1]][1]
          desc.y <- strsplit(x[2], "[ -]")[[1]][1]
          jpeg(file.path(label.path, paste(desc.x, desc.y, ".jpg", sep = "")))
          plot(cluster.values[, x[1]], cluster.values[, x[2]],
               xlab = x[1], ylab = x[2])
          dev.off()
                            })
}

test.group <- normal.grouped[[1]]

test.tube1 <- test.group[[1]]
test.tube1 <- flowProc::process_single(test.tube1, remove_margins = F,
                                       trans = "log")

test.som <- CreateFsom(test.tube1@fcs, xdim = 10, ydim = 10)
cluster.values <- test.som$map$codes

sel.values <- cluster.values[, sel.channels]
plot.folder <- "../node_density/plots"
dir.create(plot.folder, showWarnings = F, recursive = T)
jpeg(file.path(plot.folder, "test.jpg"))
plot(sel.values[,1], sel.values[,2], xlab = "FS INT LIN", ylab = "SS INT LIN")
dev.off()

jpeg(file.path(plot.folder, "ss_cd45.jpg"))
plot(sel.values[,3], sel.values[,2], xlab = "CD45-KrOr", ylab = "SS INT LIN")
dev.off()

jpeg(file.path(plot.folder, "ss_cd19.jpg"))
plot(sel.values[,4], sel.values[,2], xlab = "CD19-APCA750", ylab = "SS INT LIN")
dev.off()
