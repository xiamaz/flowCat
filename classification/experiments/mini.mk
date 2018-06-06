# GENERAL notes
#   Additional information that will be saved in the output directory
NOTE = Very small for testing

# INPUT data source
INPUT_TYPE = avec_pregating_selected

# PROCESSING options
# Names of groups to be included
#   different groups can be merged by assigning multiple comma separated groups
#   to a single custom label, such as example:normal,CLL
GROUPS = CLL;normal;FL

# Validation method to be used. Both holdout and kfold are possible here.
METHOD = holdout:r0.8

# specify a number of filters to be used with the data
FILTERS = 
# Tube numbers to be used, if multiple are given, the data in the separate tubes
# will be joined on label
TUBES = 1
