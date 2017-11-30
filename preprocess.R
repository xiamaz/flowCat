library(methods)
library(Biobase)
library(parallel)
library(flowCore)
library(FlowSOM)
library(data.table)
library(flowViz)
library(dplyr)

get_dir <- function(path, ext) {
	# infer dataset structure from folder structure and parse filename for id and set information
	#
	# Args:
	# 	path: top directory, child directories determine the classes
	# 	ext: extension of files inside the directory
	#
	# Returns:
	# 	dataframe with columns filename, group, id, set
	l = lapply(list.dirs(path, full.names=FALSE, recursive=FALSE), function(i) {
		filelist = list.files(file.path(path, i), pattern=ext, full.names=FALSE)
		f = sapply(filelist, function(x) {
				   r = regexec('^([KMPB\\d-]+) CLL 9F (\\d+).*.LMD$', x, perl=TRUE)
				   if ('-1' %in% r)
				   	   return(c(NA, NA, NA, NA))
				   m = regmatches(x, r)
				   return(c(file.path(path, i, x), i, m[[1]][[2]], strtoi(m[[1]][[3]])))
  		})
  		f = t(f)
  		f = f[!is.na(f[,1]),]
  		colnames(f) = c('filepath', 'group', 'label', 'set')
  		return(f)
  		})
  	files = do.call(rbind, l)
  	return(files)
}

read_files <- function(file_list) {
	# Load fcs files into file matrix
	#
	# Args:
	# 	file_matrix: matrix containing filename, group, id, and set information
	# 	path: path of top directory used in grouping
	#
	# Returns:
	# 	file_matrix ẃith flowframe column with loaded fcs files
	fcs_list = lapply(file_list[,'filepath'], function(x) {
						  f = read.FCS(as.character(x), dataset=1)
						  m = strsplit(markernames(f), '-')
						  # use simplified markernames that actually describe biological properties
						  newn = sapply(m, function(x) { x[[1]] } )
						  colnames(f) = newn
						  return(f)
		})

	return (cbind(file_list, fcs=fcs_list))
}

create_output_matrix <- function(path, tubenum, metanum){
	f = c(positive='CLL', negative='normal control')
	file_matrix = get_info(path, 'LMD')
	selected_files = file_matrix[file_matrix[,'group'] == 'HZLv',]
	print(selected_files)
	# file_matrix = read_files(selected_files, path)
	# file_matrix <- read_files(file_matrix, 'logicle')
	return()

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
	fSOM <- BuildMST(fSOM, tSNE=FALSE)

	META_NUM = metanum
	meta <- metaClustering_consensus(fSOM$map$codes, k=META_NUM)

	selection <- file_matrix[,]

	fsoms <- mclapply(selection[,'fcs'], function(x) { NewData(fSOM, x)}
					  ,mc.cores=detectCores(logical=FALSE))
	# fsoms <- lapply(selection[,'fcs'], function(x) { NewData(fSOM, x)})

	histos <- lapply(fsoms, function(x) {
		t = tabulate(x$map$mapping, nrow(x$map$codes))
		t / sum(t)
		})

	result_data <- do.call(rbind, histos)
	meta_result <- matrix(0, ncol=META_NUM, nrow=nrow(result_data))
	for (i in 1:ncol(result_data)) {
		meta_result[,meta[i]] = meta_result[,meta[i]] + result_data[,i]
	}


	# cheap labeling
	colnames(result_data) <- c(1:ncol(result_data))
	colnames(meta_result) <- c(1:META_NUM)
	result_data <- cbind(result_data, label=selection[,'label'], ident=selection[,'ident'])
	meta_result <- cbind(meta_result, label=selection[,'label'], ident=selection[,'ident'])
	#do.call(function(x){ rbind(x[,'freq'])}, histos)

	write.table(result_data, file=sprintf("logic_matrix_output_tube%d.csv",tubenum), sep=";")
	write.table(meta_result, file=sprintf("logic_matrix_meta_output_tube%d.csv",tubenum), sep=";")
}
# use environment variabeles to set configuration

column_occurrences <- function(fcs_info) {
	# is this reasonable?
	# get occurence of different terms across all fcs
	colmatrix = sapply(fcs_info[,'fcs'], function(x) { colnames(x) })
	print(colmatrix)
}

# path = "~/DREAM/Krawitz/"
path = Sys.getenv('PREPROCESS_PATH')
file_information = get_dir(path, 'LMD')

tube1 = file_information[file_information[,'set'] == 1, ]
tube2 = file_information[file_information[,'set'] == 2, ]

t1 = read_files(head(tube1, n=10))
column_occurrences(t1)
# column_occurrences(tube2)


# testframe = create_output_matrix(path, 1, 7)
# for (tub in 1:3) {
# 	create_output_matrix(path, tub, 7)
# }
