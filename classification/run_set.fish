#!/usr/bin/fish

function run_folder
	if [ -z "$argv[2]" ]
		set template (echo $argv[1] | sed 's/\/*$//')/(basename $argv[1])".mk"
	else
		set template $argv[2]
	end
	for exp in $argv[1]/*
		echo "Using $exp with $template"
		make run EXP=$exp TEMPLATE=$template
	end
end


run_folder $argv[1] $argv[2]
