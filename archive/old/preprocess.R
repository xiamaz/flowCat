library(methods)
library(Biobase)
library(FlowSOM)
library(data.table)
library(flowProc)
library(parallel)
library(flowDensity)

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

create_upsampling <- function(folder, s, selection, mkpath, fsom, meta, metanum, tag, plotting=FALSE, cluster,Upsampled) {
	files = create_file_info(folder,'LMD',s, cluster)
	# files = sample(files, 10)
	# subset for testing
	# files = sample(files, 200)
	# files = files[1500:length(files)]

	if (missing(cluster)) {
		lfunc = lapply
	} else {
		clusterExport(cluster, c('selection','Upsampled','fsom','meta','metanum'),envir=environment())
		lfunc = function(x,y){parLapply(cluster,x,y)}
	}

	histo_list = lfunc(files, function(cur_entry) {
		processed = process_single(cur_entry, selection, trans='log')
		if (!isS4(processed)) {
			return(NA)
		}
		if (!isS4(processed@fcs)){
			return(NA)
		}
		if (nrow(processed@fcs) < 100) {
			print(paste(processed@filepath, "has only", nrow(processed@fcs), 'events'))
			return(NA)
		}
		print(sprintf("Upsampling %s %s",processed@group,processed@label))
		upsampled = NewData(fsom, processed@fcs)
		histo = tabulate(upsampled$map$mapping, nrow(upsampled$map$codes))
		histo = histo / sum(histo)
		meta_hist = rep(0, times=metanum)
		for (j in 1:length(histo)) {
			meta_hist[meta[j]] = meta_hist[meta[j]] + histo[j]
		}
		names(histo) = c(1:length(histo))
		names(meta_hist) = c(1:length(meta_hist))
		output = Upsampled(cur_entry)
		output@histo = histo
		output@meta = meta_hist
		return(output)
	})
	print("Finished upsampling")
	histo_list = histo_list[!is.na(histo_list)]

	histo_matrix = do.call(rbind.data.frame, lapply(histo_list, function(x){t(x@histo)}))
	meta_matrix = do.call(rbind.data.frame, lapply(histo_list, function(x){t(x@meta)}))
	histo_matrix[,'group'] = unlist(lapply(histo_list,function(x){x@group}))
	histo_matrix[,'label'] = unlist(lapply(histo_list,function(x){x@label}))
	histo_matrix[,'material'] = unlist(lapply(histo_list, function(x){x@material}))
	meta_matrix[,'group'] = unlist(lapply(histo_list,function(x){x@group}))
	meta_matrix[,'label'] = unlist(lapply(histo_list,function(x){x@label}))
	meta_matrix[,'material'] = unlist(lapply(histo_list, function(x){x@material}))
	print(dim(histo_matrix))
	print(dim(meta_matrix))
	# save to csv file for further processing
	filename = function(run, s, ident) {sprintf("%d_%s_SET%d_all_%s.csv", run, tag, s, ident)};
	write.table(histo_matrix, file=filename(RUNUM, s, 'histos'), sep=";")
	write.table(meta_matrix, file=filename(RUNUM,s,'metas'), sep=";")
}

create_diagrams <- function(fsom, s, selection, meta, metanum) {
	folder = '../Moredata'
	s = 1
	cluster = makeCluster(12, type='FORK')
	files = create_file_info(folder, 'LMD', s, cluster)
	stopCluster(cluster)
	group_names = unique(lapply(files,function(x){x@group}))
	group_files = lapply(group_names, function(x){
		files[unlist(lapply(files,function(y){y@group==x}))]
	})
	dir.create('t1_2dplots')
	names(group_files) = group_names
	for (g in names(group_files)) {
		if (length(group_files[[g]]) < 20) {
			print(paste("Group",g,"only has size",length(group_files[[g]])))
			next
		}
		normal = group_files[['normal']]
		group_single = lapply(normal,function(x){
								  if (x@label=='16-026038')
								  	  return(x)
								  else
								  	  return(NA)
			})
		group_single = unlist(group_single[!is.na(group_single)])[[1]]
		group_sample = sample(group_files[[g]], 20)
		print(paste("Identifier",lapply(group_sample,function(x){x@label})))
		folderpath = sprintf('t1_2dplots/%s',g)
		if (!dir.exists(folderpath))
			dir.create(folderpath)
		for (g in group_sample) {
			processed = process_single(g,selection,trans='log')
			if (!isS4(processed)) {
				next
			}
			if (!isS4(processed@fcs)){
				next
			}
			if (nrow(processed@fcs) < 100) {
				print(paste(processed@filepath, "has only", nrow(processed@fcs), 'events'))
				next
			}
			upsampled = NewData(fsom, processed@fcs)
			print(paste("Upsampling",processed@label))
			# for (i in 1:length(unique(meta))) {
			for (i in 1:100) {
				mi = i
				# mi = unique(meta)[i]
				jpeg(sprintf("%s/AAA_%s_%d.jpg",folderpath,g@label,mi), width=10, height=10, units='cm', res=300)
				#PlotClusters2D(upsampled, 'Kappa-FITC','CD19-APCA750',(1:100)[meta==mi])
				PlotClusters2D(upsampled, 'CD19-APCA750','CD5-PacBlue',(1:100)[i])
				dev.off()
				jpeg(sprintf("%s/BBB_%s_%d.jpg",folderpath,g@label,mi), width=10, height=10, units='cm', res=300)
				# PlotClusters2D(upsampled, 'Lambda-PE','Kappa-FITC',(1:100)[meta==mi])
				PlotClusters2D(upsampled, 'CD45-KrOr','SS INT LIN',(1:100)[i])
				dev.off()
				jpeg(sprintf("%s/CCC_%s_%d.jpg",folderpath,g@label,mi), width=10, height=10, units='cm', res=300)
				# PlotClusters2D(upsampled, 'Lambda-PE','Kappa-FITC',(1:100)[meta==mi])
				PlotClusters2D(upsampled, 'CD20-PC7','CD5-PacBlue',(1:100)[i])
				dev.off()
			}
		}
	}
}

# use environment variabeles to set configuration
THRESHOLD_GROUP_SIZE <- strtoi(Sys.getenv('PREPROCESS_GROUP_THRESHOLD'))
CPUNUM <- strtoi(Sys.getenv('PREPROCESS_THREADS'))
PATH <- Sys.getenv('PREPROCESS_PATH')
METANUM <- strtoi(Sys.getenv('PREPROCESS_METANUM'))
RUNUM <- strtoi(Sys.getenv('PREPROCESS_RUNUM'))

material <- c('1','2','3','4','5','PB','KM')
Upsampled <- setClass('Upsampled'
					  ,contains='FlowEntry'
					  ,representation(histo='vector',meta='vector')
					  )

CPUNUM = 12
THRESHOLD_GROUP_SIZE = 100
PATH = '/data/ssdraid/Genetik/Moredata'
METANUM = 10
RUNUM = 400


cluster = makeCluster(CPUNUM, type='FORK')
all_files = get_dir(PATH, 'LMD', cluster)
all_files = remove_duplicates(all_files)
stopCluster(cluster)

# some temporary file exploration
filtered = filter_list(all_files,tube_set=1,material=material)
normals = filter_list(filtered,group='normal')
length(filtered)

testnormal = filter_list(normals,label='16-023971')[[1]]
testnormal = process_single(testnormal, selection, trans='log',remove_margins=TRUE,upper=TRUE,lower=FALSE)
sngl = flowDensity(testnormal@fcs,channels=channels,position=logic,ellip.gate=FALSE)
jpeg('flowdensity_normal.jpg',res=300, width=2000, height=2000)
plotDens(testnormal@fcs,channels)
lines(sngl@filter,type='l')
dev.off()

testfile = filtered[[1]]
testfile = read_file(testfile,FALSE)
testfile = process_single(testfile, selection,trans='log')

# infer selection from testfile
selection = colnames(testfile@fcs)
channels = c('CD45-KrOr','SS INT LIN')
logic = c(TRUE,FALSE)
sngl = flowDensity(testfile@fcs,channels=channels,position=logic)
jpeg('flowdensity.jpg',res=300, width=2000, height=2000)
plotDens(testfile@fcs,channels)
lines(sngl@filter,type='l')
dev.off()



for (s in c(1,2)) {
	cluster = makeCluster(CPUNUM, type='FORK')
	print(sprintf("Processing set %d", s))
	# select relevant tube
	set_files = filter_list(all_files, tube_set=s, material=material)

	# choose only files containing all markers we want to use
	r = filter_flowFrame_majority(set_files, 0.8, cluster)
	selection = r$markers
	marker_files = r$entries
	group_names <- sapply(marker_files,function(x){x@group})
	group_sizes <- table(group_names)
	selected_groups = names(group_sizes)[group_sizes >= THRESHOLD_GROUP_SIZE]
	group_lists = unlist(lapply(selected_groups, function(x){
	                     sample(filter_list(marker_files,group=x),THRESHOLD_GROUP_SIZE)}))
	clusterExport(cluster, c('selection'),envir=environment())
	loaded_files = parLapply(cluster,group_lists, function(x){process_single(x,selection,trans='log',lower=FALSE,upper=TRUE,remove_margins=TRUE)})
	loaded_files = loaded_files[!is.na(loaded_files)]

	fs = flowCore::flowSet(lapply(loaded_files,function(x){x@fcs}))
	fsom = create_fsom(fs)
	metanum = METANUM
	meta = create_metaclust(fsom, metanum)
	# mkpath = function(fn){sprintf("plots%d_%d/%s.jpg", RUNUM, s, fn)}
	dir.create(sprintf("plots%d_%d", RUNUM, s), showWarnings = FALSE)
	# jpeg(mkpath('fsom'), width=10, height=10, units='cm', res=300)
	fsom = BuildMST(fsom,tSNE=TRUE)
	# PlotStars(fsom, backgroundValues=as.factor(meta))
	# dev.off()
	# create some 2d plots
	print("Finished creating som and plot.")
	stopCluster(cluster)

	cluster = makeCluster(2, type='FORK')
	create_upsampling('../Moredata', s, selection, mkpath, fsom,meta,metanum, 'Moredata',cluster=cluster,Upsampled=Upsampled)
	stopCluster(cluster)

	cluster = makeCluster(2, type='FORK')
	create_upsampling('../Newdata', s, selection, mkpath, fsom,meta,metanum, 'Newdata',cluster=cluster,Upsampled=Upsampled)
	stopCluster(cluster)
}
