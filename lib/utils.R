CreateOutputDirectory <- function(dir.path, text.note) {
  if (dir.exists(dir.path)){
    stop(paste(dir.path, "already exists. Move or delete it."))
  }
  dir.create(dir.path, recursive = T, showWarnings = T)

  info.path <- file.path(dir.path, "00-INFO.txt")
  write(c(as.character(Sys.time()), text.note), info.path, sep = "\n")
}
