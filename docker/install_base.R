# bioconductor bootstrap based on https://github.com/Bioconductor/bioc_docker
url <- "http://bioconductor.org/packages/3.7/bioc"

if ("BiocInstaller" %in% rownames(installed.packages()))
	remove.packages("BiocInstaller")

install.packages("BiocInstaller", repos=url)

builtins <- c("Matrix", "KernSmooth", "mgcv")

for (builtin in builtins)
    if (!suppressWarnings(require(builtin, character.only=TRUE)))
        BiocInstaller::biocLite(builtin)

suppressWarnings(BiocInstaller::biocValid(fix=TRUE, ask=FALSE))

# install additional packages
install.packages("devtools")
install.packages("ggplot2")
install.packages("optparse")
install.packages("aws.s3")

BiocInstaller::biocLite("flowCore")
BiocInstaller::biocLite("FlowSOM")
BiocInstaller::biocLite("flowDensity")
