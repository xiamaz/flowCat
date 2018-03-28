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
