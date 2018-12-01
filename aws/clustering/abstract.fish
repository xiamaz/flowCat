#!/usr/bin/fish

source ../lib/run_batch.fish

# Create experiments for indiv pregating
set TAG "abstract"

set RAND 10

set EXPERIMENTS "normal" "somgated"

submit_jobs $TAG $RAND $EXPERIMENTS
