#' Add backticks to string
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
  node.plot <- ggplot2::ggplot(node.df, aes_string(x = scol.x, y = scol.y)) + ggplot2::geom_point() +
    ggplot2::geom_point(data=subset.df, colour = "red")

  plot.filename <- sprintf("%s_%s_clusters_%s.png", col.x, col.y, paste(subset.nums, collapse = "_"))
  plot.filepath <- file.path(plotting.folder, plot.filename)
  ggplot2::ggsave(plot.filepath, plot = node.plot)

}

#' Create plots to specified path as a png
CreatePngPlot <- function(path, plot.func, ...) {
  png(path, width = 10, height = 10, units = "in", res = 300)
  plot.func(...)
  dev.off()
}

#' Plot flowSOM to different visualizations
PlotClustering <- function(path, fsom, meta) {
  stars.path <- sprintf("%s_stars.png", path)
  CreatePngPlot(stars.path, FlowSOM::PlotStars, backgroundValues = as.factor(meta))
  tsne.path <- sprintf("%s_tsne.png", path)
  CreatePngPlot(tsne.path, FlowSOM::PlotStars, view = "tSNE", backgroundValues = as.factor(meta))
  grid.path <- sprintf("%s_grid.png", path)
  CreatePngPlot(grid.path, FlowSOM::PlotStars, view = "grid", backgroundValues = as.factor(meta))
  # png(stars.path, width = 10, height = 10, units = "in", res = 300)
  # # png(stars.path)
  # FlowSOM::PlotStars(fsom, backgroundValues = as.factor(meta))
  # dev.off()
}
