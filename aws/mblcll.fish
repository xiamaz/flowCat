#!/usr/bin/fish

source run_batch.fish

# Create experiments for indiv pregating
set TAG "mblcll"

set RAND 10

set EXPERIMENTS "mblcll"

for i in (seq 1 $RAND)
	set expname $TAG"_run_"$i
	batchsub $expname "run" $TAG.mk $TAG $i
end
