library(jsonlite)

source("lib/visualization.R")

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
