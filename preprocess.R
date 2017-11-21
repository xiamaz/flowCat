library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
library(data.table)
library(flowViz)

# encapsulate the result calculation for every tube

linear_transform <- function(df, col, avg, sd) {
  m = sqrt(0.3/sd)
  b = -(avg * m)
  df[,col] = df[,col] * m + b
  return(df)
}

determine_ranges <- function(csvs) {
	bound = rbindlist(csvs)
	m = apply(bound, 2, function(col) {
		c(mean(col), var(col), max(col), min(col))
	});
	rownames(m) = c('mean', 'var', 'max', 'min')
	print(m)
	return(m)
}

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

process_csv <- function(infos) {
  # cl - cluster for parallel computing
  # info - file containing metainformation for training
  csvs = mclapply(infos$FilePaths, read.csv, mc.cores=detectCores())
  ranges = determine_ranges(csvs)
  # linear transform of forward scatter
  # csvs = lapply(csvs, function(x) {
  # 			   linear_transform(x, 'FS.Lin', ranges['mean', 'FS.Lin'], ranges['var', 'FS.Lin'])
  # })
  	  # ranges = determine_ranges(csvs)
  # mc.cores = detectCores()
  flows = lapply(csvs, csv_to_flowframe)
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

create_histogram <- function(fsom, size, matrix=TRUE) {
  hist = table(fsom$map$mapping[,1])
  hist = hist / sum(hist)
  if (matrix) {
    mhist <- matrix(hist, ncol=fsom$map$xdim, nrow=fsom$map$ydim, byrow=TRUE)
    return(mhist)
  } else {
    lhist = cbind(freq=hist, pos=1:size)
    return(hist)
  }
}

create_fsom <- function(infos) {
  # ## split info in training and test
  fSet <- flowSet(process_csv(infos))
  fSOM <- ReadInput(fSet,compensate = FALSE, transform = FALSE, scale = TRUE)
  # we need to specify which columns to use, here we use all of them
  fSOM <- BuildSOM(fSOM,colsToUse = c(1:7))
  fSOM <- BuildMST(fSOM,tSNE=FALSE)
  # saveRDS(fSOM, 'fsom_tube2.rds')
  # fSOM <- readRDS('fsom_tube2.rds')
  # let weights count as one

  # ref_soms <- rep(list(fSOM), times=nrow(infos))
  # fsoms <- upsample_all(infos, ref_soms)
}

parse_info <- function(infopath) {
  infos <- read.csv(infopath, stringsAsFactors=FALSE)
  # add fixed filepaths to the csv files with a known regular format
  infos$FilePaths = lapply(infos$FCSFileName, function(x) { sprintf("CSV/%04d.CSV", x)})
  return(infos)
}

# get folders in named list, returns a matrix with filenames with associated
# labels
get_info <- function(folders, ext) {
	files = matrix(ncol=2, nrow=0)
	for (i in names(folders)) {
		f = list.files(folders[i], pattern=ext, full.names=TRUE)
		files = rbind(files, cbind(filename=f, label=i))
	}
	return(files)
}

read_files <- function(file_matrix) {
	fcs = cbind(file_matrix
				, fcs=mclapply(file_matrix[,'filename'], function(x) {
		f = read.FCS(x, dataset=1)
		log_names = sapply(featureNames(f), function(y) { if (grepl('LOG', y)) { return(y) }})
		log_names = log_names[!sapply(log_names, is.null)]
		lgcl = estimateLogicle(f, channels=unname(log_names))
		transform(f, lgcl)
		}, mc.cores=detectCores(logical=FALSE)))
	return(fcs)
}

setwd("~/DREAM")

f = c(positive='Krawitz/CLL', negative='Krawitz/normal control')

file_matrix = get_info(f, 'CLL 9F 01.*.LMD')
file_matrix <- read_files(file_matrix)

fSOM <- ReadInput(
	# flowSet(file_matrix[file_matrix[,'label'] == 'negative','fcs'])
	flowSet(file_matrix[,'fcs'])
	#flowSet(file_matrix[1:10,'fcs'])
	#file_matrix[[183,3]]
	,compensate = FALSE
	,transform = FALSE
	,scale = TRUE
	)

fSOM <- BuildSOM(fSOM)
fSOM <- BuildMST(fSOM, tSNE=TRUE)

meta <- metaClustering_consensus(fSOM$map$codes, k=5)
positives = file_matrix[file_matrix[,'label'] == 'positive','fcs']

selection <- file_matrix[,]

fsoms <- mclapply(selection[,'fcs'], function(x) { NewData(fSOM, x)}
				  ,mc.cores=detectCores(logical=FALSE))
histos <- lapply(fsoms, function(x) {
	t = tabulate(x$map$mapping, nrow(x$map$codes))
	t / sum(t)
	})
result_data <- do.call(rbind, histos)
# cheap labeling
colnames(result_data) <- c(1:ncol(result_data))
result_data <- cbind(result_data, label=selection[,'label'])
#do.call(function(x){ rbind(x[,'freq'])}, histos)

write.table(result_data, file="matrix_output_logic_trans_no_lin.csv", sep=";")
