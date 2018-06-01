# General notes
NOTE = Batch 32 and epoch 100

# processing options
# GROUPS = LPL;Marginal;Mantel;CLLPL;CLL;MBL;FL;normal
# GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;normal
GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL
# GROUPS = CLL;normal
METHOD = kfold:5
# FILTERS = smallest
FILTERS =
TUBES = 1;2
