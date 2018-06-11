#!/usr/bin/fish

source run_batch.fish

# Create experiments for indiv pregating
set TAG "pregated_revision"

set RAND 5

set EXPERIMENTS "indiv_pregating" "indiv_pregating_exc"

submit_jobs $TAG $RAND $EXPERIMENTS
