library(methods)
library(Biobase)
library(FlowSOM)
library(data.table)
library(flowProc)

# use environment variabeles to set configuration
THRESHOLD_GROUP_SIZE <- strtoi(Sys.getenv('PREPROCESS_GROUP_THRESHOLD'))
CPUNUM <- strtoi(Sys.getenv('PREPROCESS_THREADS'))
PATH <- Sys.getenv('PREPROCESS_PATH')
METANUM <- strtoi(Sys.getenv('PREPROCESS_METANUM'))
RUNUM <- strtoi(Sys.getenv('PREPROCESS_RUNUM'))

material <- c('1','2','3','4','5','PB','KM')

load_data <- function(name, runum, set, folder='./') {
	r = toString(runum)
	s = toString(set)
	n = toString(name)
	filename = sprintf("%s_%s_%s.rds", r, n, s)
	filepath = file.path(folder, filename)
	return(readRDS(filepath))
}

create_histo <- function(fsom, cur_info, mkpath, create_upsample_plot=FALSE) {
	print(sprintf("Upsampling %s %s",cur_info['group'],cur_info['label']))
	upsampled = NewData(fsom, cur_info['fcs'][[1]])
	if (create_upsample_plot) {
		jpeg(mkpath(sprintf("%s_%s",cur_info['group'],cur_info['label'])), width=10, height=10, units='cm', res=300)
		PlotStars(upsampled)
		dev.off()
	}
	histo = tabulate(upsampled$map$mapping, nrow(upsampled$map$codes))
	histo = histo / sum(histo)
	meta_hist = rep(0, times=metanum)
	for (j in 1:length(histo)) {
		meta_hist[meta[j]] = meta_hist[meta[j]] + histo[j]
	}
	names(histo) = c(1:length(histo))
	names(meta_hist) = c(1:length(meta_hist))
	histo = c(histo, cur_info['group'], cur_info['label'], cur_info['material'])
	meta_hist = c(meta_hist, cur_info['group'], cur_info['label'], cur_info['material'])
	return(list(histo=histo, meta=meta_hist))
}



runum <- 0
metanum <- 10
RUNUM <- runum
filename = function(run, s, tag) {sprintf("%d_FSOM_FILES_NEWDATA_SET%d_all_%s.csv", run, s, tag)};

create_fsom_plot <- FALSE

for (s in c(1,2)) {
	print(sprintf("Processing set %d", s))
	selection <- load_data('selection', runum=runum, set=s)
	process_with_selection <- function(x) { process_single(x,selection)}
	fsom <- load_data('fsom', runum=runum, set=s)
	meta <- load_data('meta', runum=runum, set=s)
	files <- load_data('files', runum=runum, set=s)
	files <- files[,c('filepath','group','label','material','set')]
	write.table(files, file=filename(RUNUM, s, 'FSOM_FILES'), sep=";")
	write.table(selection, file=filename(RUNUM,s,'FSOM_SELECTION'),sep=";")


	mkpath = function(fn){sprintf("newplots%s_%s/%s.jpg", toString(RUNUM), toString(s), toString(fn))}
	if (create_fsom_plot) {
		dir.create(sprintf("newplots%s_%s", toString(RUNUM), toString(s)), showWarnings = FALSE)
		jpeg(mkpath('fsom'), width=10, height=10, units='cm', res=300)
		fsom <- BuildMST(fsom, tSNE=TRUE)
		PlotStars(fsom, view="grid")
		dev.off()
	}

	selected_rows = create_file_info('../Newdata', ext='LMD', set=s)
	materials = unique(selected_rows[,'material'])
	sample_size = 10
	histos = NULL
	metas = NULL
	for (m in materials) {
		m_rows = selected_rows[selected_rows[,'material'] == m,,drop=FALSE]
		rownum = nrow(m_rows)
		mkpath = function(fn){sprintf("newplots%s_%s/%s/%s.jpg",toString(RUNUM),toString(s),toString(m),toString(fn))}
		if (rownum > sample_size) {
			sel = sample(1:rownum, sample_size)
			m_rows = m_rows[sel,]
		}
		s_size = min(sample_size, rownum)
		create_histo_with_group <- function(x) { create_histo(fsom, x, mkpath)}
		for (i in 1:s_size) {
			r = m_rows[i,]
			r = process_with_selection(r)
			if (is.null(r)) {
				next
			}
			result = create_histo_with_group(r)
			histo = c(result$histo, cur_info['group'], cur_info['label'], cur_info['material'])
			meta_hist = c(result$meta, cur_info['group'], cur_info['label'], cur_info['material'])
			if (is.null(histos)) {
				histos = histo
			} else {
				histos = rbind(histos, histo)
			}
			if (is.null(metas)) {
				metas = meta_hist
			} else {
				metas = rbind(metas, meta_hist)
			}
		}
	}

	filename = function(run, s, tag) {sprintf("%d_NEWDATA_SET%d_all_%s.csv", run, s, tag)};
	write.table(histos, file=filename(RUNUM, s, 'histos'), sep=";")
	write.table(metas, file=filename(RUNUM,s,'metas'), sep=";")

	# selected_rows = filter_file_info(selected_rows, material=material)

	# print("Finished creating som and plot.")
	# for (i in 1:nrow(selected_rows)) {
	# 	cur_info = selected_rows[i,]
	# 	cur_info = process_single(cur_info, selection)
	# 	if (is.null(cur_info)) {
	# 		next
	# 	}
	# 	result = create_histo(fsom, selected_rows, mkpath)
	# 	histo = c(result$histo, cur_info['group'], cur_info['label'], cur_info['material'])
	# 	meta_hist = c(result$meta, cur_info['group'], cur_info['label'], cur_info['material'])
	# 	if (i == 1) {
	# 		histos = histo
	# 	} else {
	# 		histos = rbind(histos, histo)
	# 	}
	# 	if (i == 1) {
	# 		metas = meta_hist
	# 	} else {
	# 		metas = rbind(metas, meta_hist)
	# 	}
	# }
	# print(dim(histos))
	# print(dim(metas))
	# # save to csv file for further processing
	# filename = function(run, s, tag) {sprintf("%d_NEWDATA_SET%d_all_%s.csv", run, s, tag)};
	# write.table(histos, file=filename(RUNUM, s, 'histos'), sep=";")
	# write.table(metas, file=filename(RUNUM,s,'metas'), sep=";")
}
