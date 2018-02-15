library(flowProc)
library(flowDensity)
library(flowCore)
library(FlowSOM)
library(parallel)

OUTPUTDIR = 'flowdensity'
SOMDIR = 'flowsoms'
DENSDIR = 'flowdensity'
CSVDIR = 'csvs'
material = c('1','2','3','4','5','6','KM','PB')

fsom_plot <- function(fsom, folder, name) {
	jpeg(file.path(folder, name), res=300, width=10, height=10, units='cm')
	PlotStars(fsom$FlowSOM, backgroundValues = as.factor(fsom$metaclustering))
	dev.off()
}

save_csv <- function(tabledata, folder, name) {
	filename = paste(name, '.csv', sep='')
	write.table(tabledata, file=file.path(folder, filename), sep='\t')
}

create_node_map <- function(fsom) {
	map = fsom$FlowSOM$map
	map_nodes = map$codes
	map_cells = map$mapping
	nodes_occ = table(map_cells[,1])
	# leave empty places in case some nodes do not contain any cells
	v = rep(0, 100)
	names(v) = 1:100
	v[names(nodes_occ)] = nodes_occ
	nodes_occ = v
	sorted_nodes = cbind(nodes_occ, map_nodes)
	sorted_nodes = sorted_nodes[order(sorted_nodes[,'nodes_occ'], decreasing=T),]
	return(sorted_nodes)
}

cluster = makeCluster(4, type='FORK')
path = '/home/max/DREAM/Moredata'
files = get_dir(path, 'LMD', cluster=cluster)
filtered = filter_list(files, tube_set=1,material=material)
r = filter_flowFrame_majority(filtered, 0.8, cluster)
filtered = r$entries
selection = r$markers
stopCluster(cluster)

all_group_names = unique(unlist(lapply(filtered, function(x){x@group})))


mbl = filter_list(filtered, group='MBL')
normals = filter_list(filtered, group='normal')
clls = filter_list(filtered, group='CLL')

gates = c('FS INT LIN','SS INT LIN')
logic = c(T,F)
case = mbl[[1]]





case = normals[[4]]
g2 = c('CD45-KrOr','SS INT LIN')
case = process_single(case, selection, trans='log', remove_margins=FALSE)
nlist = lapply(normals, function(x){process_single(x,selection,trans='log',remove_margins=FALSE)@fcs})
nset = flowSet(nlist)
fnset = as(nset, 'flowFrame')
setden = flowDensity(fnset, channels=g2,position=c(T,F))
gated = lapply(normals, function(x){
	i = process_single(x,selection,trans='log',remove_margins=TRUE, lower=FALSE)
	mi = flowDensity(i@fcs,channels=g2,position=c(T,F),ellip.gate=T,bimodal=c(T,T))
	i@fcs = i@fcs[mi@index,]
	return(i)})
glist = lapply(gated, function(x){x@fcs})
fsom = FlowSOM(flowSet(glist), compensate=F,transform=F,scale=T,nClus=10,colsToUse=colnames(i@fcs))
n=0
jpeg(paste(n,'test.jpg',sep=''),res=300,width=100, height=100, units='mm')
PlotStars(fsom$FlowSOM, backgroundValues = as.factor(fsom$metaclustering))
dev.off()


case = normals[[4]]
case = process_single(case, selection, trans='log', remove_margins=FALSE)
# g2 = c('FS INT LIN', 'SS INT LIN')
more = flowDensity(case@fcs, channels=g2,position=c(T,F),ellip.gate=T,percentile=c(0.5,0.65))
lines(more@filter, type='l')
dev.off()

case2 = clls[[1]]
case2 = process_single(case2, selection, trans='log', remove_margins=FALSE)
more2 = flowDensity(case2@fcs, channels=g2,position=c(T,F), control=c(more,more), use.control=c(T,T))
jpeg('test.jpg',res=300,width=100, height=100, units='mm')
plotDens(case2@fcs, g2)
lines(more2@filter, type='l')
dev.off()

case3 = mbl[[4]]
case3 = process_single(case3, selection, trans='log', remove_margins=FALSE)
more3 = flowDensity(case3@fcs, channels=g2,position=c(T,F), control=c(more,more), use.control=c(T,F),ellip.gate=T)
jpeg('test.jpg',res=300,width=100, height=100, units='mm')
plotDens(case3@fcs, g2)
lines(more3@filter, type='l')
dev.off()

stopCluster(cluster)
