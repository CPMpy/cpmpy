* ----------------------------------------------------------------------------------------------------
* File written by CPMpy
*     Format: 'mps'
*     CPMpy Version: 0.10.0
* ----------------------------------------------------------------------------------------------------
* SCIP STATISTICS
*   Problem name     : CPMpy Model
*   Variables        : 2 (0 binary, 2 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 1
NAME          CPMpy Model
OBJSENSE
  MAX
ROWS
 N  Obj 
 E  c1 
COLUMNS
    INTSTART  'MARKER'                            'INTORG'                           
    y         Obj                              2  c1                               1 
    x         Obj                              1  c1                               1 
    INTEND    'MARKER'                            'INTEND'                           
RHS
    RHS       c1                               5 
BOUNDS
 UP Bound     y                               10 
 UP Bound     x                               10 
ENDATA
