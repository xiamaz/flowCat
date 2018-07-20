#!/usr/bin/fish

function run_folder
	if [ (count $argv) -ge 2 ]
		set template $argv[2]
	else
		set template (echo $argv[1] | sed 's/\/*$//')/(basename $argv[1])".mk"
	end
	for exp in $argv[1]/*.mk
		if not [ (basename $exp) = (basename $template) ]
			echo "Using $exp with $template"
			make run EXP=$exp TEMPLATE=$template
			make upload EXP=$exp TEMPLATE=$template
		end
	end
end

if [ (count $argv) = 0 ]
	echo "Usage: experiment_folder <opt: custom master template>"
	exit
end

run_folder $argv
