#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=16,walltime=60:00,mem=80gb
#PBS -M max.zhao@hu-berlin.de
#PBS -m abe -N flowCatter
#PBS -j oe
export PREPROCESS_PATH="$HOME/Moredata"
export PREPROCESS_GROUP_THRESHOLD=70
export PREPROCESS_THREADS=16
export PREPROCESS_METANUM=20

/gpfs01/share/R/3.3.2/bin/Rscript $HOME/flowCat/preprocess.R
