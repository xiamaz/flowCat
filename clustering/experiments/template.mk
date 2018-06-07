# Reduction method for the data
METHOD = normal
# Run on each case individually before fitting the model
PREMETHOD = normal
# Run on each case individually before transforming using model
TRANSMETHOD = normal
# Used tubes in the data
TUBES = "1;2"
# Number of cases per cohort for consensus SOM
NUM = 5
# Number of cases per cohort to be upsampled (maximum number)
# enable: NUM
# disable: set to 0 or lower
UPSAMPLING_NUM = 800
# Select specific cohorts to be upsampled, otherwise all cohorts will be
# processed
# enable: --groups GROUPS
GROUPS = ""

# additional flags
# possible flags:
# 	--plot = enable plotting data
# 	--refnormal = enable exclusion of normal cohort in reference generation
FLAGS = 
