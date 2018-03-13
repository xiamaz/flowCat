library(ggplot2)

Backtickize <- function(x) {return(paste("`", x, "`", sep = ""))}

#' Plot cluster nodes and mark selected nodes with another color
#'
#' Nodes will be plotted two-dimensionally using selected marker channels.
#' Implementation detail: Backticks are used to escape names with illegal
#' characters in R. This is necessary and probably the most elegant solution.
#' @examples
#' PlotNodes(consensus.fsom$map$codes, "SS INT LIN", "CD45-KrOr", c(11, 19, 25))
PlotNodes <- function(node.matrix, col.x, col.y, subset.nums) {
  node.df <- as.data.frame(node.matrix)

  subset.df <- node.df[subset.nums,]
  # backticks because R is unable to deal with spaces
  col.x <- Backtickize(col.x)
  col.y <- Backtickize(col.y)
  ggplot(node.df, aes_string(x = col.x, y = col.y)) + geom_point() +
    geom_point(data=subset.df, colour = "red")
}

output.folder <- "output/preprocess"
experiment.name <- "1_NetworkAnalysis"

fsom.filename <- "stored_consensus_fsom_tube1.rds"

fsom.path <- file.path(output.folder, experiment.name, fsom.filename)

consensus.fsom <- readRDS(fsom.path)

marker.names <- consensus.fsom$prettyColnames

PlotNodes(consensus.fsom$map$codes, "SS INT LIN", "CD45-KrOr", c(11, 19, 25))
