#!/usr/bin/fish

source ../lib/run_batch.fish

# Create experiments for indiv pregating
set TAG "exotic"

set RAND 1

set EXPERIMENTS "normal" "somgated"

submit_jobs $TAG $RAND $EXPERIMENTS
