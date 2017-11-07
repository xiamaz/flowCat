library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
library(emdist)
library(foreach)
library(doMC)

# encapsulate the result calculation for every tube

process_csv <- function(infos) {
  # cl - cluster for parallel computing
  # info - file containing metainformation for training
  
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
  
  flows <- mclapply(infos$FilePaths, transform_csv, mc.cores=detectCores())
  
  return(flows)
}

upsample_all <- function(merged, ref_som) {
  upsample <- function(mst, pat){
    # use newData(fsom, ff) to get a new node distribution
    return(NewData(mst, pat))
  }
  csv_to_flowframe <- function(df, ref_som) {
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
    return(upsample(ref_som, ff))
  }
  transform_csv <- function(path, ref_som) {
    c = read.csv(path)
    return(csv_to_flowframe(c, ref_som))
  }
  
  # fsoms <- parLapply(cl, merged$FilePaths, function(x) { transform_csv(x, ref_som) })
  # fsoms <- lapply(merged$FilePaths, function(x) { transform_csv(x, ref_som) })
  fsoms <- mcmapply(transform_csv, merged$FilePaths, ref_soms, mc.cores=detectCores(), SIMPLIFY = FALSE)
  return(fsoms)
}

save_plot <- function(ff, info) {
  if (!dir.exists('upsample'))
    dir.create('upsample')
  info
  path = paste("upsample/", info, '.png', sep = "")
  png(path)
  PlotStars(ff)
  dev.off()
}

calc_emd <- function(ffa, ffb) {
  ma <- cbind(ffa$MST$size, ffa$MST$l)
  mb <- cbind(ffb$MST$size, ffa$MST$l)
  return (emd(ma, mb, dist='euclidean'))
}

create_histogram <- function(fsom) {
  map = fsom$map$mapping
  aggr = aggregate(map, by=list(map[,1]), function(x) { length(x) })
  hist = merge(data.frame(Group.1=1:nrow(fsom$map$codes)), aggr[,c('Group.1', 'V1')], by='Group.1', all.x=TRUE, all.y=TRUE)
  hist[is.na(hist)] <- 0
  mhist <- matrix(unlist(hist[,'V1']), ncol=fsom$map$xdim, byrow = FALSE, nrow=fsom$map$ydim)
  return(mhist)
}

# calculate relief scores for every element in chosen based on nearest
# hit and nearest miss
relief_score  <- function(emd_matrix, labels) {
  rscore  <- function(entry, labels) {
  	  sorted = cbind(entry, labels)
  	  # exclude the first identity entry
  	  sorted = sorted[order(sorted[,1]),][-1,]
  	  # sort by score
  	  # take first normal
  	  dist_normal = sorted[sorted[,2] == 'normal',][1,1]
  	  if (is.na(dist_normal)) dist_normal = 0
  	  # take first aml
  	  dist_other = sorted[sorted[,2] == 'aml',][1,1]
  	  if (is.na(dist_other)) dist_other = 0
  	  result_dist = as.numeric(dist_normal) - as.numeric(dist_other)
  	  return (result_dist)
  }
  result = apply(emd_matrix, 1, rscore, labels=labels)
  return(result)
}

predict_emd <- function(infos) {
  fSet <- flowSet(process_csv(infos))
  fSOM <- ReadInput(fSet,compensate = FALSE,transform = FALSE, scale = TRUE)
  # we need to specify which columns to use, here we use all of them
  fSOM <- BuildSOM(fSOM,colsToUse = c(1:7))
  fSOM <- BuildMST(fSOM,tSNE=TRUE)
  
  ref_soms <- rep(list(fSOM), times=nrow(infos))
  
  fsoms <- upsample_all(infos, ref_soms)
  
  # get histrogram distribution from fsom data
  mhists = mclapply(fsoms, create_histogram)
  
  emd_matrix <- matrix( nrow=length(fsoms), ncol=length(fsoms))
  mapply(emd2d, mhists, mhists)
  foreach(i = 1:length(mhists), j = 1:length(mhists)) %dopar% 
    emd_matrix[i,j] = emd2d(mhists[[i]], mhists[[j]])
  # labels <- matrix(chosen_selection$Label, nrow=length(chosen_selection$Label), ncol=length(chosen_selection$Label), byrow=TRUE)
  scores <- relief_score(emd_matrix, chosen_selection$Label)
  results <- cbind(scores, chosen_selection$Label)
}

parse_info <- function(infopath) {
  infos <- read.csv(infopath)
  # add fixed filepaths to the csv files with a known regular format
  infos$FilePaths = lapply(infos$FCSFileName, function(x) { sprintf("CSV/%04d.CSV", x)})
  return(infos)
}

setwd("~/DREAM")
registerDoMC(cores=detectCores())

infopath = 'AMLTraining.csv'
tubeNum = 2
infos <- parse_info(infopath)
# use the selection to create the fsom cohort and also use these for the prediction
info_selection <- infos[(infos$TubeNumber == tubeNum) & !is.na(infos$Label),]
result <- predict_emd(info_selection)


