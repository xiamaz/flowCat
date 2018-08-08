#!/usr/bin/fish

source ../lib/run_batch.fish

# Create experiments for indiv pregating
set TAG "num_cases"

set RAND 10

set EXPERIMENTS "c1" "c25" "c50"

submit_jobs $TAG $RAND $EXPERIMENTS
