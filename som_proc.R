library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
set.seed(42)

# cluster for poor people
cl <- makeCluster(detectCores())

process_csv <- function(cl, infos) {
  # cl - cluster for parallel computing
  # info - file containing metainformation for training
  clusterExport(cl, 'flowFrame')
  clusterExport(cl, 'AnnotatedDataFrame')
  
  csv_to_flowframe <- function(df) {
    # transform csv into a flowframe
    # source: https://gist.github.com/yannabraham/c1f9de9b23fb94105ca5
    meta <- data.frame(name=dimnames(df)[[2]],
                       desc=paste('this is column',dimnames(df)[[2]],'from your CSV')
    )
    meta$range <- apply(apply(df,2,range),2,diff)
    meta$minRange <- apply(df,2,min)
    meta$maxRange <- apply(df,2,max)
    
    ff <- new("flowFrame",
              exprs=data.matrix(df),
              parameters=AnnotatedDataFrame(meta)
    )
    return(ff)
  }
  
  transform_csv <- function(path) {
    c = read.csv(path)
    return(csv_to_flowframe(c))
  }
  
  flows <- parLapply(cl, infos$FilePaths, transform_csv)
  
  return(flows)
}

infofile = 'AMLTraining.csv'
infos <- read.csv(infofile)
infos$FilePaths = lapply(infos$FCSFileName, function(x) { sprintf("CSV/%04d.CSV", x)})

# tube1 <- infos[infos$TubeNumber == 1,]
# tube12 <- infos[infos$TubeNumber == 1 || infos$TubeNumber == 2,]
# pat1 <- infos[infos$SampleNumber == 1,]
# pat_sel <- infos[(infos$SampleNumber %in% 1:30) & (infos$TubeNumber %in% c(1,2)),]

tube_sel <- infos[(infos$TubeNumber %in% c(1,2)),]

fSet <- flowSet(process_csv(cl, tube_sel))
fSOM <- ReadInput(fSet,compensate = FALSE,transform = FALSE, scale = TRUE)
# we need to specify which columns to use, here we use all of them
fSOM <- BuildSOM(fSOM,colsToUse = c(1:7))
fSOM <- BuildMST(fSOM,tSNE=TRUE)
png("foobarp12.png")
PlotStars(fSOM)
dev.off()
stopCluster(cl)