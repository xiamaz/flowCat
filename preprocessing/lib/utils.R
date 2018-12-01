CreateOutputDirectory <- function(text.note, dir.path, temp.path) {
  stop.fun <- function() {
    stop(paste(dir.path, "already exists. Move or delete it."))
  }
  con.fun <- function() {
    dat <- c(as.character(Sys.time()), text.note)
    write.fun <- function(dat, filepath) {
      write(dat, filepath, sep = "\n")
    }
    info.path <- PutFile(dat, "00-INFO.txt", dir.path, write.fun, temp.path)
  }
  PathExists(file.path(dir.path, "00-INFO.txt"), stop.fun, con.fun)
}
