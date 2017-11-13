library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
library(emdist)
library(foreach)
library(doMC)

# encapsulate the result calculation for every tube

trans_ffs <- function(csvdf, col) {
  old = csvdf[,col]
  m = sqrt(0.1/var(old))
  b = -(mean(old) * m)
  new = old * m + b
  csvdf[,col] = new
  return(csvdf)
}

csv_to_flowframe <- function(df) {
  # transform csv into a flowframe
  # source: https://gist.github.com/yannabraham/c1f9de9b23fb94105ca5
  df = trans_ffs(df, 'FS.Lin')
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

process_csv <- function(infos) {
  # cl - cluster for parallel computing
  # info - file containing metainformation for training
  flows <- mclapply(infos$FilePaths, transform_csv, mc.cores=detectCores())
  return(flows)
}

upsample <- function(mst, pat){
  # use newData(fsom, ff) to get a new node distribution
  ## only adjust the data from ffs

  return(NewData(mst, pat))
}

upsample_all <- function(merged, ref_soms) {
  # fsoms <- parLapply(cl, merged$FilePaths, function(x) { transform_csv(x, ref_som) })
  # fsoms <- lapply(merged$FilePaths, function(x) { transform_csv(x, ref_som) })
  ffs = process_csv(merged)
  fsoms = mcmapply(upsample, ref_soms, ffs, mc.cores=detectCores(), SIMPLIFY = FALSE)
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

create_histogram <- function(fsom, matrix=TRUE) {
  # map = fsom$map$mapping
  # aggr = aggregate(map, by=list(map[,1]), function(x) { length(x) })
  # hist = merge(data.frame(Group.1=1:nrow(fsom$map$codes)), aggr[,c('Group.1', 'V1')], by='Group.1', all.x=TRUE, all.y=TRUE)
  # hist[is.na(hist)] <- 0
  hist = table(fsom$map$mapping[,1])
  hist = hist / sum(hist)
  if (matrix) {
    # mhist <- matrix(unlist(hist[,'V1']), ncol=fsom$map$xdim, byrow = TRUE, nrow=fsom$map$ydim)
    mhist <- matrix(hist, ncol=fsom$map$xdim, nrow=fsom$map$ydim, byrow=TRUE)
    return(mhist)
  } else {
    # lhist = hist[,c('V1', 'Group.1')]
    lhist = cbind(freq=hist, pos=1:length(hist))
    return(lhist)
  }
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

get_from_distance_matrix <- function(va, vb, dist_matrix) {
  d = dist_matrix[va[1], vb[1]]
  return(d)
}

predict_emd <- function(infos) {
  # ## split info in training and test
  test_set <- infos[is.na(infos$Label),]
  infos <-  infos[!is.na(infos$Label),]
  
  fSet <- flowSet(process_csv(infos))
  fSOM <- ReadInput(fSet,compensate = FALSE,transform = FALSE, toTransform=c(1:7), scale = TRUE)
  # we need to specify which columns to use, here we use all of them
  fSOM <- BuildSOM(fSOM,colsToUse = c(1:7))
  fSOM <- BuildMST(fSOM,tSNE=FALSE)
  # saveRDS(fSOM, 'fsom_tube2.rds')
  # fSOM <- readRDS('fsom_tube2.rds')
  # let weights count as one
  distance_matrix = distances(fSOM$MST$graph, weights=NA)
  # use weights defined on the edges
  ## distance_matrix = distances(fSOM$MST$graph)
  
  ref_soms <- rep(list(fSOM), times=nrow(infos))
  
  fsoms <- upsample_all(infos, ref_soms)
  
  # get histrogram distribution from fsom data
  # mhists = mclapply(fsoms, create_histogram)
  hists = mclapply(fsoms, function(x) { create_histogram(x, matrix=FALSE) })
  
  emd_matrix <- matrix( nrow=length(fsoms), ncol=length(fsoms))
  # for (i in 1:length(hists)) {
  #   foreach (j = 1:length(hists)) %dopar% {
  #  	  emd_matrix[i,j] = emd(hists[[i]], hists[[j]], dist=function(a, b) {get_from_distance_matrix(a,b,distance_matrix) } )
  #   }
  # }
  for (i in 1:length(hists)) {
  	  for (j in 1:length(hists)) {
  	    emd_matrix[i,j] = emd(hists[[i]], hists[[j]], dist=function(a, b) {get_from_distance_matrix(a,b,distance_matrix) } )
  	  }
  }
  # labels <- matrix(chosen_selection$Label, nrow=length(chosen_selection$Label), ncol=length(chosen_selection$Label), byrow=TRUE)
  scores <- relief_score(emd_matrix, infos$Label)
  results <- cbind(scores, infos$Label)
  return(results)
}

parse_info <- function(infopath) {
  infos <- read.csv(infopath, stringsAsFactors=FALSE)
  # add fixed filepaths to the csv files with a known regular format
  infos$FilePaths = lapply(infos$FCSFileName, function(x) { sprintf("CSV/%04d.CSV", x)})
  return(infos)
}

interpret_result <- function(results, info) {
  positive = results[results[,1] > 0,]
  cat("Positive results: \n")
  print(positive)
  tp = positive[positive[,2] == 'aml',]
  fp = positive[positive[,2] == 'normal',]
  cat(paste("These contain ", nrow(tp), " with aml and ", nrow(fp), " without aml."))
  cat("--------\n")
  negative = results[results[,1] <= 0,]
  tn = negative[negative[,2] == 'normal',]
  fn = negative[negative[,2] == 'aml',]
  cat("negative results: \n")
  print(negative)
  cat(paste("These contain ", nrow(fn), " with aml and ", nrow(tn), " without aml."))
  cat("--------\n")
}

setwd("~/DREAM")
registerDoMC(cores=detectCores())

infopath = 'AMLTraining.csv'
tubeNum = 2
infos <- parse_info(infopath)
# use the selection to create the fsom cohort and also use these for the prediction
info_selection <- infos[infos$TubeNumber == tubeNum,]
result <- predict_emd(info_selection)
interpret_result(result, info_selection)
