# GENERAL notes
#   Additional information that will be saved in the output directory
NOTE = Process only tube 1 without pregating

# PROCESSING options
# Names of groups to be included
#   different groups can be merged by assigning multiple comma separated groups
#   to a single custom label, such as example:normal,CLL
GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL

# Validation method to be used. Both holdout and kfold are possible here.
METHOD = kfold:5

# specify a number of filters to be used with the data
FILTERS =
# Tube numbers to be used, if multiple are given, the data in the separate tubes
# will be joined on label
TUBES = 1
