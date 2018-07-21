# GENERAL notes
#   Additional information that will be saved in the output directory
NOTE = Process only tube 1

# Tube numbers to be used, if multiple are given, the data in the separate tubes
# will be joined on label
TUBES = 1;2

# pattern to be used to get directories to be processed
PATTERN = miniseq_201806

# Validation method to be used. Both holdout and kfold are possible here.
# valid kfold:num holdout:<a|r>num
# examples: kfold:5;holdout:r0.8
METHOD = kfold:5

# PROCESSING options
# Names of groups to be included
#   different groups can be merged by assigning multiple comma separated groups
#   to a single custom label, such as example:normal,CLL
# GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal"
GROUPS = 

# Set a size for the content
# format: (group?):(min-?)max;...
# SIZE = --size 100
SIZE = 

# specify a number of filters to be used with the data
# valid: smallest - all cohorts are size of the smallest cohort
# MODIFIERS = --modifiers smallest
MODIFIERS = 
