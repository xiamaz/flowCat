#!/bin/sh
#PBS -k o
#PBS -l nodes=1:ppn=8,walltime=60:00,mem=20gb
#PBS -M max.zhao@hu-berlin.de
#PBS -m abe -N flowCatter
#PBS -j oe
export PREPROCESS_PATH="$HOME/Moredata"
export PREPROCESS_CACHED_FSOM=FALSE
export PREPROCESS_GROUP_THRESHOLD=40
export PREPROCESS_THREADS=8
export PREPROCESS_METANUM=10
R-3.3.2 CMD BATCH $HOME/flowCat/preprocess.R
