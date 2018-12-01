#!/usr/bin/fish

source ../lib/run_batch.fish

# Create experiments for indiv pregating
set TAG "comp_pregating"

set RAND 5

set EXPERIMENTS "pregated_combined" "somgated" "som" "always_som" "som_combined"

submit_jobs $TAG $RAND $EXPERIMENTS
