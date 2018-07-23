#!/usr/bin/fish

source run_batch.fish

# Create experiments for indiv pregating
set TAG "abstract"

set RAND 10

set EXPERIMENTS "normal" "pregated" "somgated"

submit_jobs $TAG $RAND $EXPERIMENTS
