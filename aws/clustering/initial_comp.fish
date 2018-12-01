#!/usr/bin/fish

source ../lib/run_batch.fish

# Create experiments for indiv pregating
set TAG "initial_comp"

set RAND 5

set EXPERIMENTS "indiv_pregating" "indiv_pregating_exc" "normal" "normal_exc"

submit_jobs $TAG $RAND $EXPERIMENTS
