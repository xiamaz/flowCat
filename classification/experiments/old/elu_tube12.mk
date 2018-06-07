# General notes
NOTE = Process using both tube1 and tube2

# processing options
# GROUPS = LPL;Marginal;Mantel;CLLPL;CLL;MBL;FL;normal
# GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;normal
GROUPS = LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;normal
# GROUPS = CLL;normal
METHOD = kfold:5
# FILTERS = smallest
FILTERS =
TUBES = 1;2
