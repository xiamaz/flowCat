library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
library(emdist)

# load arbitrary amounts of flowSOM MST trees for upsampling

# compare cells of patient against the mst and return distribution of cells across the mst

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
  return (emd(ma, mb, dist='manhattan'))
}

# cl <- makeCluster(detectCores())
# cl <- makeCluster(1)
# clusterExport(cl, 'flowFrame')
# clusterExport(cl, 'AnnotatedDataFrame')
# clusterExport(cl, 'NewData')

setwd("~/DREAM")
ref_som <- readRDS('fsom_mst_tube1.rds')

infofile = 'AMLTraining.csv'
infos <- read.csv(infofile, stringsAsFactors=FALSE)
infos$FilePaths = lapply(infos$FCSFileName, function(x) { sprintf("CSV/%04d.CSV", x)})

# select a specific number of patients
normal_selection <- infos[((((infos$TubeNumber == 1) & (infos$Label == 'normal'))) & !is.na(infos$Label)),][(1:10),]
aml_selection <- infos[((((infos$TubeNumber == 1) & (infos$Label == 'aml'))) & !is.na(infos$Label)),][(1:10),]


all_tube1 <- infos[infos$TubeNumber == 1,]
first_20 <- all_tube1[1:20,]

chosen_selection <- first_20
ref_soms <- rep(list(ref_som), times=nrow(chosen_selection))

fsoms <- upsample_all(chosen_selection, ref_soms)

emd_matrix <- matrix( nrow=length(fsoms), ncol=length(fsoms))
for (i in 1:length(fsoms)) {
  for (j in 1:length(fsoms)) {
    emd_matrix[i,j] = calc_emd(fsoms[[i]], fsoms[[j]])
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
# labels <- matrix(chosen_selection$Label, nrow=length(chosen_selection$Label), ncol=length(chosen_selection$Label), byrow=TRUE)
scores <- relief_score(emd_matrix, chosen_selection$Label)
cbind(scores, chosen_selection$Label)
