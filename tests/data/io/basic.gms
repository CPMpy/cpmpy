$OFFLISTING
* ----------------------------------------------------------------------------------------------------
* File written by CPMpy
*     Format: 'gms'
*     CPMpy Version: 0.10.0
* ----------------------------------------------------------------------------------------------------
* SCIP STATISTICS
*   Problem name     : CPMpy Model
*   Variables        : 2 (0 binary, 2 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 1

$MAXCOL 255
$OFFDIGIT

Variables
 objvar, y, x;

Integer variables
 y, x;

* Variable bounds
 y.up = 10;
 x.up = 10;

Equations
 objequ, c1;

 objequ .. objvar =e= (2*y +x);

 c1 .. +1*x +1*y =e= 5;

Model m / all /;

option limrow = 0;
option limcol = 0;
$if gamsversion 242 option intvarup = 0;

$if not set MIP $set MIP MIP
Solve m using %MIP% maximizing objvar;
