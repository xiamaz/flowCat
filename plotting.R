library(flowCore)
library(ggplot2)
library(flowViz)

setwd("~/FLOW") 

files <- list.files(path="./Krawitz/CLL", pattern=".LMD", full.names=T, recursive=FALSE) #vector of LMD files
path="./Krawitz/CLL"

createplots<-function (files){#Creates plots for the parameters in all input files
  flowframe<-read.FCS(files[1],transformation = FALSE,alter.names=TRUE,dataset = 1)
  dataframe<-data.frame(flowframe@exprs)
  pdf("Plots.pdf")
  FSplot<-ggplot(dataframe, aes(x=FS.INT.LIN)) + geom_density()
  SSplot<-ggplot(dataframe, aes(x=SS.INT.LIN)) + geom_density()
  FL1plot<-ggplot(dataframe, aes(x=FL1.INT.LOG)) + geom_density()
  FL2plot<-ggplot(dataframe, aes(x=FL2.INT.LOG)) + geom_density()
  FL3plot<-ggplot(dataframe, aes(x=FL3.INT.LOG)) + geom_density()
  FL4plot<-ggplot(dataframe, aes(x=FL4.INT.LOG)) + geom_density()
  FL5plot<-ggplot(dataframe, aes(x=FL5.INT.LOG)) + geom_density()
  FL6plot<-ggplot(dataframe, aes(x=FL6.INT.LOG)) + geom_density()
  FL7plot<-ggplot(dataframe, aes(x=FL7.INT.LOG)) + geom_density()
  FL8plot<-ggplot(dataframe, aes(x=FL8.INT.LOG)) + geom_density()
  FL9plot<-ggplot(dataframe, aes(x=FL9.INT.LOG)) + geom_density()
  FL10plot<-ggplot(dataframe, aes(x=FL10.INT.LOG)) + geom_density()
  
  for (i in 2:length(files)){
    nextflowframe<-read.FCS(files[i],transformation = FALSE,alter.names=TRUE,dataset = 1)
    nextdataframe<-data.frame(nextflowframe@exprs)
    FSplot<-FSplot+geom_density(data=nextdataframe,aes(x=FS.INT.LIN))
    SSplot<-SSplot+geom_density(data=nextdataframe,aes(x=SS.INT.LIN))
    FL1plot<-FL1plot+geom_density(data=nextdataframe,aes(x=FL1.INT.LOG))
    FL2plot<-FL2plot+geom_density(data=nextdataframe,aes(x=FL2.INT.LOG))
    FL3plot<-FL3plot+geom_density(data=nextdataframe,aes(x=FL3.INT.LOG))
    FL4plot<-FL4plot+geom_density(data=nextdataframe,aes(x=FL4.INT.LOG))
    FL5plot<-FL5plot+geom_density(data=nextdataframe,aes(x=FL5.INT.LOG))
    FL6plot<-FL6plot+geom_density(data=nextdataframe,aes(x=FL6.INT.LOG))
    FL7plot<-FL7plot+geom_density(data=nextdataframe,aes(x=FL7.INT.LOG))
    FL8plot<-FL8plot+geom_density(data=nextdataframe,aes(x=FL8.INT.LOG))
    FL9plot<-FL9plot+geom_density(data=nextdataframe,aes(x=FL9.INT.LOG))
    FL10plot<-FL10plot+geom_density(data=nextdataframe,aes(x=FL10.INT.LOG))
  }
  print(FSplot)
  print(SSplot)
  print(FL1plot)
  print(FL2plot)
  print(FL3plot)
  print(FL4plot)
  print(FL5plot)
  print(FL6plot)
  print(FL7plot)
  print(FL8plot)
  print(FL9plot)
  print(FL10plot)
  dev.off()
}  

createplots(files)
