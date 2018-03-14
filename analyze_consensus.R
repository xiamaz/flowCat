library(jsonlite)
library(ggplot2)

Backtickize <- function(x) {return(paste("`", x, "`", sep = ""))}

#' Plot cluster nodes and mark selected nodes with another color
#'
#' Nodes will be plotted two-dimensionally using selected marker channels.
#' Implementation detail: Backticks are used to escape names with illegal
#' characters in R. This is necessary and probably the most elegant solution.
#' @examples
#' PlotNodes(consensus.fsom$map$codes, "SS INT LIN", "CD45-KrOr", c(11, 19, 25))
PlotNodes <- function(node.matrix, col.x, col.y, subset.nums, plotting.folder = "output/visualization") {
  node.df <- as.data.frame(node.matrix)

  subset.df <- node.df[subset.nums, ]
  # backticks because R is unable to deal with spaces
  scol.x <- Backtickize(col.x)
  scol.y <- Backtickize(col.y)
  node.plot <- ggplot(node.df, aes_string(x = scol.x, y = scol.y)) + geom_point() +
    geom_point(data=subset.df, colour = "red")

  plot.filename <- sprintf("%s_%s_clusters_%s.png", col.x, col.y, paste(subset.nums, collapse = "_"))
  plot.filepath <- file.path(plotting.folder, plot.filename)
  ggsave(plot.filepath, plot = node.plot)

}

output.folder <- "output/preprocess"
experiment.name <- "1_NetworkAnalysis"

fsom.filename.1 <- "stored_consensus_fsom_tube1.rds"
fsom.filename.2 <- "stored_consensus_fsom_tube2.rds"

fsom.path.1 <- file.path(output.folder, experiment.name, fsom.filename.1)
fsom.path.2 <- file.path(output.folder, experiment.name, fsom.filename.2)

consensus.fsom <- list(`1` = readRDS(fsom.path.1),
                       `2` = readRDS(fsom.path.2))

marker.names <- lapply(consensus.fsom, function(x) {x$prettyColnames})

plotting.folder <- "output/visualization_2"
if (dir.exists(plotting.folder)) {
  stop(paste(plotting.folder, "already exists. Remove or move it."))
}
output.layers <- fromJSON("output_layers.json")

gatings <- list(CLL = list(`1` = list(c("CD45-KrOr", "SS INT LIN"),
                                      c("CD20-PC7", "CD23-APC"),
                                      c("CD19-APCA750", "CD5-PacBlue")))
)

for (group in names(output.layers)) {
  message("Analyzing ", group)
  clusters.list <- output.layers[[group]]
  sel.clusters <- clusters.list[clusters.list > 5]
  cluster.names <- as.numeric(names(sel.clusters))
  first.cluster.list <- sel.clusters[cluster.names < 100]
  second.cluster.list <- lapply(sel.clusters[cluster.names >= 100], function(x){x - 100})
  sel.clusters <- list(`1` = first.cluster.list,
                       `2` = second.cluster.list)
  plot.destination <- file.path(plotting.folder, group)
  dir.create(plot.destination, recursive = T)
  if (group %in% names(gatings)) {
    for (tube in names(gatings[[group]])) {
      tube.clusters <-sel.clusters[[tube]]
      if (length(tube.clusters) == 0) {
        next
      }
      tube.clusters <- as.numeric(names(tube.clusters))
      print(tube.clusters)
      for (gating in gatings[[group]][[tube]]){
        PlotNodes(consensus.fsom[[tube]]$map$codes, gating[[1]], gating[[2]],
                  tube.clusters, plotting.folder = plot.destination)
      }
    }
  }
}
