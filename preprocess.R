library(methods)
library(Biobase)
library(FlowSOM)
library(data.table)
library(flowProc)

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
THRESHOLD_GROUP_SIZE <- strtoi(Sys.getenv('PREPROCESS_GROUP_THRESHOLD'))
CPUNUM <- strtoi(Sys.getenv('PREPROCESS_THREADS'))
PATH <- Sys.getenv('PREPROCESS_PATH')
METANUM <- strtoi(Sys.getenv('PREPROCESS_METANUM'))
THRESHOLD_GROUP_SIZE = 70
CPUNUM = 8
PATH = '/home/max/DREAM/Moredata'
METANUM = 10

processed = process_dir(PATH, ext='LMD', set=1, threads=CPUNUM, simple_marker_names=FALSE, group_size=70)
file_info = processed$file_info
selection = processed$selection

fs = flowCore::flowSet(file_info[,'fcs'])
fsom = create_fsom(fs)
# saveRDS(fsom, 'tempfsom.rds')
# fsom = readRDS('tempfsom.rds')
metanum = METANUM
meta = create_metaclust(fsom, metanum)

selected_rows = create_file_info(PATH, ext='LMD', set=1)

for (i in 1:nrow(selected_rows)) {
	cur_info = selected_rows[i,]
	cur_info = process_single(cur_info, selection)
	if (is.null(cur_info)) {
		next
	}
	upsampled = NewData(fsom, cur_info['fcs'][[1]])
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
write.table(histos, file="MOREDATA_all_histos.csv", sep=";")
write.table(metas, file="MOREDATA_all_metas.csv", sep=";")
