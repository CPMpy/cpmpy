$OFFLISTING
* ----------------------------------------------------------------------------------------------------
* File written by CPMpy
*     Format: 'gms'
*     CPMpy Version: 0.10.0
* ----------------------------------------------------------------------------------------------------
* SCIP STATISTICS
*   Problem name     : CPMpy Model
*   Variables        : 12 (0 binary, 12 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 12

$MAXCOL 255
$OFFDIGIT

Variables
 objvar, x5, x6, x0, x8, x2, x9, x3, x1, x7, x4, x10, x11;

Integer variables
 x5, x6, x0, x8, x2, x9, x3, x1, x7, x4, x10, x11;

* Variable bounds
 x5.up = 10;
 x6.up = 10;
 x0.up = 10;
 x8.up = 10;
 x2.up = 10;
 x9.up = 10;
 x3.up = 10;
 x1.up = 10;
 x7.up = 10;
 x4.up = 10;
 x10.up = 10;
 x11.up = 10;

Equations
 objequ, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12;

 objequ .. objvar =e= (6*x5 +7*x6 +x0 +9*x8 +3*x2 +10*x9 +4*x3 +2*x1 +8*x7 +5*x4 +11*x10 +12*x11);

 c1 .. +1*x0 +1*x1 +1*x2 +1*x3 +1*x4 +1*x5 +1*x6 +1*x7 +1*x8 +1*x9 +1*x10 +1*x11 =e= 36;

 c2 .. +1*x0 +1*x1 =e= 3;

 c3 .. +1*x2 +1*x3 =e= 5;

 c4 .. +1*x4 +1*x5 =e= 7;

 c5 .. +1*x6 +1*x7 =e= 4;

 c6 .. +1*x8 +1*x9 =e= 6;

 c7 .. +1*x10 +1*x11 =e= 3;

 c8 .. +1*x0 +1*x2 =e= 4;

 c9 .. +1*x2 +1*x4 =e= 6;

 c10 .. +1*x4 +1*x6 =e= 4;

 c11 .. +1*x6 +1*x8 =e= 6;

 c12 .. +1*x8 +1*x10 =e= 4;

Model m / all /;

option limrow = 0;
option limcol = 0;
$if gamsversion 242 option intvarup = 0;

$if not set MIP $set MIP MIP
Solve m using %MIP% maximizing objvar;
