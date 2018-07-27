SET = abstract
PATTERN = somgated_2

GROUPS = --group "LMg:LPL,Marginal;MtCp:Mantel,CLLPL;CM:CLL,MBL;FL;HZL;normal"

# run only one iteration of holdout with same config as 10-fold
METHOD = holdout:r0.9
TUBES = 1;2

NOTE = SOMGated. Usage for quick classification development.

SIZE = --size 3000