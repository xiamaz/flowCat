library(flowCore)
library(FlowSOM)
testfile <- "output/4-flowsom-cmp/dataset/fcsdata/159e4754f47453ea9f0d177d88592eb62af598d8-2 CLL 9F 01 N09 001.LMD"
data <- read.FCS(testfile, dataset=1, transformation=F, scale=F)
scaled <- ReadInput(data, transform=F, scale=T)
print(scaled$data[1,])
