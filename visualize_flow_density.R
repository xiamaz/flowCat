library(flowProc)
library(flowDensity)

kGatingSequences <- list(`1` = list(list(c("CD45-KrOr", "SS INT LIN"), c(T, F)))
                         )

kInputFiles <- "../Moredata"

kExperimentName <- "flowdensity_0"

kOutputDirectory <- file.path("output/visualization", kExperimentName)

if (dir.exists(kOutputDirectory)) {
  stop(paste(kOutputDirectory, "already exists. Move or delete it."))
}

all.files <- CreateFileInfo(kInputFiles, num.threads = 12)

file.groups <- GroupBy(all.files, "group", num.threads = 12)

# group files by disease cohort and afterwards by tube number
# iterate over entries and apply all gatings that we need
for (group.name in names(file.groups)){
  if (group.name != "CLL") {
    next
  }
  message("Visualizing ", group.name, " with flow density.")
  group.file <- GroupBy(file.groups[[group.name]], "tube_set", num.threads = 12)

  plotting.folder <- file.path(kOutputDirectory, group.name)
  dir.create(plotting.folder, showWarnings = F, recursive = T)

  for (tube.num in names(group.file)) {
    if (!tube.num %in% names(kGatingSequences)) {
      next
    }
    for (entry in group.file[[tube.num]]) {
      for (gating in kGatingSequences[[tube.num]]) {
        plot.name <- sprintf("%s_%s.png", entry@label, paste(gating[[1]], collapse = "_"))
        plot.path <- file.path(plotting.folder, plot.name)

        entry <- ProcessSingle(entry, trans = "log")
        sngl <- flowDensity(entry@fcs, channels = gating[[1]], position = gating[[2]])
        png(plot.path)
        plotDens(entry@fcs, gating[[1]])
        lines(sngl@filter, type = "l")
        dev.off()
      }
    }
  }
}
