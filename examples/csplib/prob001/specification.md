---
Title:    Car Sequencing
Proposer: Barbara Smith 
Category: Scheduling and related problems
---


A number of cars are to be produced; they are not identical, because different options are available as variants on the basic model. The assembly line has different stations which install the various options (air-conditioning, sun-roof, etc.). These stations have been designed to handle at most a certain percentage of the cars passing along the assembly line. Furthermore, the cars requiring a certain option must not be bunched together, otherwise the station will not be able to cope. Consequently, the cars must be arranged in a sequence so that the capacity of each station is never exceeded. For instance, if a particular station can only cope with at most half of the cars passing along the line, the sequence must be built so that at most 1 car in any 2 requires that option. The problem has been shown to be NP-complete (Gent 1999).

The format of the data files is as follows:

* First line: number of cars; number of options; number of classes.
* Second line: for each option, the maximum number of cars with that option in a block.
* Third line: for each option, the block size to which the maximum number refers.
* Then for each class: index no.; no. of cars in this class; for each option, whether or not this class requires it (1 or 0).

This is the example given in (Dincbas et al., ECAI88):

<pre>
10 5 6
1 2 1 2 1
2 3 3 5 5
0 1 1 0 1 1 0 
1 1 0 0 0 1 0 
2 2 0 1 0 0 1 
3 2 0 1 0 1 0 
4 2 1 0 1 0 0 
5 2 1 1 0 0 0 
</pre>

A valid sequence for this set of cars is:

<pre>
Class	Options req.
0	1 0 1 1 0
1	0 0 0 1 0
5	1 1 0 0 0
2	0 1 0 0 1
4	1 0 1 0 0
3	0 1 0 1 0
3	0 1 0 1 0
4	1 0 1 0 0
2	0 1 0 0 1
5	1 1 0 0 0
</pre>