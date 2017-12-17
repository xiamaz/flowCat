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
RUNUM <- strtoi(Sys.getenv('PREPROCESS_RUNUM'))

for (set in c(1,2)) {
	print(sprintf("Processing set %d", set))
	processed = process_dir(PATH, ext='LMD', set=set, threads=CPUNUM, simple_marker_names=FALSE, group_size=70)
	file_info = processed$file_info
	selection = processed$selection

	fs = flowCore::flowSet(file_info[,'fcs'])
	fsom = create_fsom(fs)
	metanum = METANUM
	meta = create_metaclust(fsom, metanum)
	mkpath = function(fn){sprintf("plots%d_%d/%s.jpg", RUNUM, set, fn)}
	dir.create(sprintf("plots%d_%d", RUNUM, set), showWarnings = FALSE)
	jpeg(mkpath('fsom'), width=10, height=10, units='cm', res=300)
	PlotStars(fsom)
	dev.off()
	selected_rows = create_file_info(PATH, ext='LMD', set=set)
	print("Finished creating som and plot.")
	for (i in 1:nrow(selected_rows)) {
		cur_info = selected_rows[i,]
		cur_info = process_single(cur_info, selection)
		if (is.null(cur_info)) {
			next
		}
		print(sprintf("Upsampling %s %s",cur_info['group'],cur_info['label']))
		upsampled = NewData(fsom, cur_info['fcs'][[1]])
		jpeg(mkpath(sprintf("%s_%s",cur_info['group'],cur_info['label'])), width=10, height=10, units='cm', res=300)
		PlotStars(upsampled)
		dev.off()
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
		if (i == 1) {
			histos = histo
		} else {
			histos = rbind(histos, histo)
		}
		if (i == 1) {
			metas = meta_hist
		} else {
			metas = rbind(metas, meta_hist)
		}
	}
	print(dim(histos))
	print(dim(metas))
	# save to csv file for further processing
	filename = function(run, set, tag) {sprintf("%d_MOREDATA_SET%d_all_%s.csv", run, set, tag)};
	write.table(histos, file=filename(RUNUM, set, 'histos'), sep=";")
	write.table(metas, file=filename(RUNUM,set,'metas'), sep=";")
}
