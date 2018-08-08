#!/usr/bin/fish
# Submit to the classification pipeline

set HELPARGS -h --help

set HELP "Submit to the classification pipeline.
Arg: experiment folder or single experiment file"

source (dirname (status --current-filename))/lib/run_batch.fish

set QUEUE "CPU-Queue"
set DEFINITION "Classification:3"

# check that we have provided arguments
if contains -- $argv[1] $HELPARGS; or [ (count $argv) -eq 0 ]
	echo $HELP
	exit
end

set exp_files $argv[1]
set jobname (basename $argv[1] .mk)

batchsub $jobname $exp_files
