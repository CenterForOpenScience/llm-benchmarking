* Encoding: UTF-8.

DATASET ACTIVATE ohtsubo-final.
RECODE a4 (7=1) (6=2) (5=3) (4=4) (3=5) (2=6) (1=7) INTO a4r.
EXECUTE.


COMPUTE atot=a1 + a2 + a3 + a4r + a5 + a6.
EXECUTE.



T-TEST GROUPS=condition(1 0)
  /MISSING=ANALYSIS
  /VARIABLES=atot
  /CRITERIA=CI(.95).
