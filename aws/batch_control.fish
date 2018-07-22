#!/usr/bin/fish

function list_queues
	aws batch describe-job-queues --query 'jobQueues[*].[jobQueueName]' --output text
end

set VALID_JOB_STATE SUBMITTED PENDING RUNNABLE STARTING RUNNING

function list_jobs
	set queue_name $argv[1]
	set state $argv[2]
	if [ (count $argv) -ge 3 ]
		set query_fields $argv[3]
	else
		set query_fields 'jobId,jobName'
	end
	aws batch list-jobs --job-queue $queue_name --output text --query 'jobSummaryList[*].['$query_fields']' --job-status $state
end

function list_all_jobs
	set queue_name $argv[1]
	for state in $VALID_JOB_STATE
		echo $state
		list_jobs $queue_name $state 'jobName'
		echo ""
	end
end

# enter a list of jobs to cancel them
function cancel_job
	set job_id $argv[1]
	set reason $argv[2]
	aws batch cancel-job --job-id $job_id --reason $reason
end

# cancel all jobs
function cancel_all_jobs
	set queue $argv[1]
	for state in $VALID_JOB_STATE
		set jobs (list_jobs $queue $state 'jobId')
		for job in $jobs
			echo "Cancelling $job"
			cancel_job $job "Batch cancellation."
		end
	end
end

switch $argv[1]
case ls
	if [ (count $argv) -eq 2 ]
		list_all_jobs $argv[2]
	else
		list_queues
	end
case rmall
	cancel_all_jobs $argv[2]
end
