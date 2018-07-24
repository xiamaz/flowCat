#!/usr/bin/fish

source ../lib/run_batch.fish

set QUEUE "CPU-Queue"
set DEFINITION "Classification:3"

# provide
# name
# target_name
batchsub "cllmbl" "experiments/cll_mbl_normal"
