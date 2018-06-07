# GENERAL notes
#   Additional information that will be saved in the output directory
NOTE = Process joined tube 1 and tube 2 without pregating

INPUT_TYPE = sans_pregating_selected

# PROCESSING options
# Names of groups to be included
#   different groups can be merged by assigning multiple comma separated groups
#   to a single custom label, such as example:normal,CLL
GROUPS = LPL;Marginal;Mantel;CLLPL;CLL;MBL;FL;normal

# Validation method to be used. Both holdout and kfold are possible here.
METHOD = kfold:5

# specify a number of filters to be used with the data
FILTERS =
# Tube numbers to be used, if multiple are given, the data in the separate tubes
# will be joined on label
TUBES = 1;2
