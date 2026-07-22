* ----------------------------------------------------------------------------------------------------
* File written by CPMpy
*     Format: 'mps'
*     CPMpy Version: 0.10.0
* ----------------------------------------------------------------------------------------------------
* SCIP STATISTICS
*   Problem name     : CPMpy Model
*   Variables        : 12 (0 binary, 12 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 12
NAME          CPMpy Model
OBJSENSE
  MAX
ROWS
 N  Obj 
 E  c1 
 E  c2 
 E  c3 
 E  c4 
 E  c5 
 E  c6 
 E  c7 
 E  c8 
 E  c9 
 E  c10 
 E  c11 
 E  c12 
COLUMNS
    INTSTART  'MARKER'                            'INTORG'                           
    x5        Obj                              6  c1                               1 
    x5        c4                               1 
    x6        Obj                              7  c5                               1 
    x6        c11                              1  c1                               1 
    x6        c10                              1 
    x0        c2                               1  Obj                              1 
    x0        c8                               1  c1                               1 
    x8        c12                              1  c1                               1 
    x8        Obj                              9  c11                              1 
    x8        c6                               1 
    x2        Obj                              3  c9                               1 
    x2        c1                               1  c3                               1 
    x2        c8                               1 
    x9        c1                               1  c6                               1 
    x9        Obj                             10 
    x3        c1                               1  c3                               1 
    x3        Obj                              4 
    x1        c1                               1  c2                               1 
    x1        Obj                              2 
    x7        c1                               1  c5                               1 
    x7        Obj                              8 
    x4        c1                               1  Obj                              5 
    x4        c10                              1  c9                               1 
    x4        c4                               1 
    x10       Obj                             11  c7                               1 
    x10       c1                               1  c12                              1 
    x11       c1                               1  Obj                             12 
    x11       c7                               1 
    INTEND    'MARKER'                            'INTEND'                           
RHS
    RHS       c1                              36  c2                               3 
    RHS       c3                               5  c4                               7 
    RHS       c5                               4  c6                               6 
    RHS       c7                               3  c8                               4 
    RHS       c9                               6  c10                              4 
    RHS       c11                              6  c12                              4 
BOUNDS
 UP Bound     x5                              10 
 UP Bound     x6                              10 
 UP Bound     x0                              10 
 UP Bound     x8                              10 
 UP Bound     x2                              10 
 UP Bound     x9                              10 
 UP Bound     x3                              10 
 UP Bound     x1                              10 
 UP Bound     x7                              10 
 UP Bound     x4                              10 
 UP Bound     x10                             10 
 UP Bound     x11                             10 
ENDATA