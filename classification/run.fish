#!/usr/bin/fish
set TEMPLATE_NAME "00-SETTINGS.mk"

function run_folder
	if [ (count $argv) -ge 2 ]
		set template $argv[2]
	else
		set template (echo $argv[1] | sed 's/\/*$//')/$TEMPLATE_NAME
	end
	for exp in $argv[1]/*.mk
		# ignore the template file in the same directory
		if not [ (basename $exp) = $TEMPLATE_NAME ]
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
