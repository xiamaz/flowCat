#!/usr/bin/fish

set RAND 5

source run_batch.fish

# Create experiments for indiv pregating
set TAG "initial_comp"

set EXPERIMENTS "indiv_pregating" "indiv_pregating_exc" "normal" "normal_exc"

set RUNTYPES "run" "run-selected"

for experiment in $EXPERIMENTS
	set expfile $TAG/$experiment.mk

	for run in $RUNTYPES
		for i in (seq 1 $RAND)
			set expname $TAG"_"$experiment"_"$run"_"$i
			echo "Creating $run $expname with $expfile"
			batchsub $expname $run $expfile $TAG
		end
	end
end
