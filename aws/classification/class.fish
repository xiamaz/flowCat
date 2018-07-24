#!/usr/bin/fish

source ../lib/run_batch.fish

set QUEUE "CPU-Queue"
set DEFINITION "Classification:3"

batchsub $argv
