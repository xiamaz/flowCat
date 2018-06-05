# Reduction method for the data
METHOD = normal
# Run on each case individually before fitting the model
PREMETHOD = pregatesom
# Run on each case individually before transforming using model
TRANSMETHOD = pregate
# Used tubes in the data
TUBES = "1;2"
# Number of cases per cohort for consensus SOM
NUM = 5
# Number of cases per cohort to be upsampled (maximum number)
UPSAMPLING_NUM = 100
# Select specific cohorts to be upsampled, otherwise all cohorts will be
# processed
GROUPS = "CLL;MBL;CLLPL;LPL;Marginal;normal"
# Choose whether data should be plotted
PLOT = --plot
