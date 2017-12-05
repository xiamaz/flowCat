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

read_file <- function(file_row) {
	f = read.FCS(as.character(file_row[['filepath']]), dataset=1)
	# use simplified markernames, this might be an inappropriate simplification
	m = strsplit(markernames(f), '-')
	newn = sapply(m, function(x) { x[[1]] } )
	colnames(f) = newn

	return (c(file_row, fcs=f))
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
	fcs_list = mclapply(file_list[,'filepath'], function(x) {
						  f = read.FCS(as.character(x), dataset=1)
						  m = strsplit(markernames(f), '-')
						  # use simplified markernames that actually describe biological properties
						  newn = sapply(m, function(x) { x[[1]] } )
						  colnames(f) = newn
						  return(f)
		}, mc.cores=detectCores())

	return (cbind(file_list, fcs=fcs_list))
}

create_fsom <- function(fs){
	fSOM <- ReadInput(
		fs
		,compensate = FALSE
		,transform = FALSE
		,scale = TRUE
		)

	fSOM <- BuildSOM(fSOM)
	fSOM <- BuildMST(fSOM, tSNE=FALSE)
	return(fSOM)
}

create_metaclust <- function(fsom, metanum) {
	meta <- metaClustering_consensus(fsom$map$codes, k=metanum)
	return(meta)
}

# use environment variabeles to set configuration

column_occurrences <- function(fcs_info) {
	# is this reasonable?
	# get occurence of different terms across all fcs
	colmatrix = lapply(fcs_info[,'fcs'], colnames)
	colen = max(unlist(lapply(colmatrix, length)))
	colmatrix = sapply(colmatrix, function(x) { length(x) = colen; x})

	tab = table(colmatrix)
	return(tab)
}

majority_markers <- function(fcs_info) {
	threshold = 0.95
	selected = column_occurrences(fcs_info) / nrow(fcs_info)
	return(names(selected)[selected > threshold])
}

modify_selection_row <- function(fcs_row, selection) {
	ff = fcs_row['fcs'][[1]]
	ffn = colnames(ff)
	if (!any(is.na(match(selection, ffn)))) {
	 	fcs_row['fcs'] = list(ff[,selection])
	} else {
		fcs_row['fcs'] = NA
	}
	return(fcs_row)
}

modify_selection <- function(fcs_info, selection) {
	# modify flowframe columns, accepting only flowframes containing the specified selection

	for (i in 1:nrow(fcs_info)) {
		ff = fcs_info[i,'fcs'][[1]]
	 	ffn = colnames(ff)
	 	if (!any(is.na(match(selection, ffn)))) {
	 		fcs_info[i,'fcs'] = list(ff[,selection])
	 	} else {
	 		fcs_info[i,'fcs'] = NA
	 	}
	}
	fcs_info = fcs_info[!is.na(fcs_info[,'fcs']),]
	return(fcs_info)
}

remove_small_cohorts <- function (fcs_info, minsize) {
	## downsampling for flowsom to bite sized chunks
	# set a minimum size to exclude very small cohorts first
	chosen_groups = c()
	file_groups = unique(fcs_info[,'group'])
	for (g in file_groups) {
		if (table(fcs_info[,'group'] == g)['TRUE'] > minsize) {
			chosen_groups = c(chosen_groups, g)
		} else {
			fcs_info = fcs_info[fcs_info[,'group'] != g,]
		}
	}
	return(fcs_info)
}

THRESHOLD_GROUP_SIZE = 30
# path = "~/DREAM/Krawitz/"
path = Sys.getenv('PREPROCESS_PATH')
cached = Sys.getenv('PREPROCESS_CACHED')
file_information = get_dir(path, 'LMD')

# remove duplicates until we have a better idea
file_freq = table(unlist(rownames(file_information)))
file_freq = file_freq[file_freq > 1 ]
if (length(file_freq) > 0) {
	duplicates = names(file_freq)
	for (d in duplicates) {
		g = grep(d, rownames(file_information))
		# remove the file path from both occurences, better not to use this information
		# TODO clarify and remove this potential source of error
		file_information[g[1], 'filepath'] = NA
		file_information[g[2], 'filepath'] = NA
	}
	file_information = file_information[!is.na(file_information[,'filepath']),]
}

file_groups = unique(file_information[,'group'])

tube1 = file_information[file_information[,'set'] == 1, ]
tube2 = file_information[file_information[,'set'] == 2, ]

tf1 = remove_small_cohorts(tube1, THRESHOLD_GROUP_SIZE +10)
tf_subset = matrix(ncol=ncol(tf1), nrow=0)
for (g in unique(tf1[,'group'])) {
	tf_subset = rbind(tf_subset, head(tf1[tf1[,'group'] == g,], n=THRESHOLD_GROUP_SIZE))
}
print(dim(tf_subset))

# loading files first is a really dumb idea
t1 = read_files(tf_subset)
selected = majority_markers(t1)

t1 = modify_selection(t1, selected)

transform_ff <- function(ff, selection) {
	logTrans <- logTransform(transformationId="log10-transformation", logbase=10, r=1, d=1)
	trans_markers = selection[!grepl("LIN", selection)]
	transform_list = transformList(trans_markers, logTrans)
	ff['fcs'] = list(transform(ff['fcs'][[1]], transform_list))
	return(ff)
}
transform_ffs <- function(ffs, selection) {
	# for (i in 1:nrow(ffs)) {
	# 	ff = ffs[i,'fcs'][[1]]
	#  	ffs[i,'fcs'] = list(transform(ff, transform_list))
	# }
	ffs = apply(ffs,1, function(x) { transform_ff(x, selection) })
	ffs = do.call(rbind, ffs)
	return(ffs)
}

t1 = transform_ffs(t1, selected)

fs = flowSet(t1[,'fcs'])

fsom = create_fsom(fs)
saveRDS(fsom, 'tempfsom.rds')

metanum = 7

meta = create_metaclust(fsom, metanum)

# upsample data in a similar fashion to our fsom generation
selection_num = 40

selected_rows = matrix(nrow=0, ncol=ncol(tf1))
for (g in unique(tf1[,'group'])) {
	selected_rows = rbind(selected_rows, head(tf1[tf1[,'group']==g,], n=selection_num))
}

rm(histos)
rm(metas)
for (i in 1:nrow(selected_rows)) {
	print(i)
	cur_info = selected_rows[i,]
	cur_info = read_file(cur_info)
	mf = modify_selection_row(cur_info, selected)
	# skip if we encounter na in the data
	if (any(is.na(mf))) {
		cat(paste("Skipping", cur_info['filepath'], "because NA encountered."))
		next
	}
	upsampled = NewData(fsom, mf['fcs'][[1]])
	histo = tabulate(upsampled$map$mapping, nrow(upsampled$map$codes))
	histo = histo / sum(histo)
	meta_hist = rep(0, times=metanum)
	for (j in 1:length(histo)) {
		meta_hist[meta[j]] = meta_hist[meta[j]] + histo[j]
	}
	names(histo) = c(1:length(histo))
	names(meta_hist) = c(1:length(meta_hist))
	histo = c(histo, cur_info['group'], cur_info['label'])
	meta_hist = c(meta_hist, cur_info['group'], cur_info['label'])
	if (!exists('histos')) {
		histos = histo
	} else {
		histos = rbind(histos, histo)
	}
	if (!exists('metas')) {
		metas = meta_hist
	} else {
		metas = rbind(metas, meta_hist)
	}
}
print(dim(histos))
print(dim(metas))
# save to csv file for further processing
write.table(histos, file="MOREDATA_40ea_histos.csv", sep=";")
write.table(metas, file="MOREDATA_40ea_metas.csv", sep=";")
