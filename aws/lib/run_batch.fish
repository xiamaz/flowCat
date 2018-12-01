#!/usr/bin/fish
# Define some convenience features for AWS Batch control

set opt --output text

set STATUS SUBMITTED PENDING RUNNABLE STARTING RUNNING

# List all submitted jobs in any active status
function batchjobs
	if [ (count $argv) -ne 1 ]
		echo "Usage:
		1 - Job queue name"
		exit
	end
	set queue $argv[1]
	for stat in $STATUS
		aws batch list-jobs --job-queue $queue --job-status $stat $opt | awk '{print $4}'
	end
end

# List all available job definitions
function batchls
	aws batch describe-job-definitions --status ACTIVE $opt | awk 'BEGIN { FS = "[ \t]+" };/JOBDEFINITIONS/ {print $3 ":" $4}'
end

# Submit a clustering job
function submit_clustering
	if [ (count $argv) -ne 5 ]
		echo "Usage:
		1 - jobname
		2 - make target name
		3 - experiment name
		4 - tag name
		5 - experiment suffix"
		exit
	end
	set jobname $argv[1]
	if not contains $jobname (batchjobs "GPU-Queue")
		aws batch submit-job --job-definition "Clustering:11" --job-queue "GPU-Queue" --job-name $jobname --parameters target=$argv[2] --container-overrides environment="[{name=EXP_NAME,value=$argv[3]},{name=TAG_NAME,value=$argv[4]},{name=SUFFIX,value=_$argv[5]}]"
	else
		echo "$jobname already submitted"
	end
end

# Submit a classification job
function submit_classification
	if [ (count $argv) -lt 2 ]
		echo "Usage:
		1 - jobname
		2 - runscript target directory or makefile
		"
	end
	set jobname $argv[1]
	set command $argv[2]
	if not contains $jobname (batchjobs "CPU-Queue")
		aws batch submit-job --job-definition "Classification:3" --job-queue "CPU-Queue" --job-name $jobname --container-overrides command=$command
	else
		echo "$jobname already submitted"
	end
end

set RUNTYPES "run"

# set RUNTYPES "run-selected"

function submit_jobs
	set TAG $argv[1]
	set RAND $argv[2]
	set EXPERIMENTS $argv[3..-1]
	for experiment in $EXPERIMENTS
		set expfile $TAG/$experiment.mk
		for run in $RUNTYPES
			for i in (seq 1 $RAND)
				set expname $TAG"_"$experiment"_"$run"_"$i
				echo "Creating $run $expname with $expfile"
				submit_clustering $expname $run $expfile $TAG $i
			end
		end
	end
end
