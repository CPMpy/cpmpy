Constraint Programming
----------------------

.. Basic concepts on Constraint programming
Many real-life decisions involve a large number of options. To decide if a problem is feasible or finding the best one amongst all the options is hard task to do by hand. In other words, to enumerate all the possible combinations of single decisions and evaluate them is infeasible in practice. To avoid this "*brute force*" approach, the paradigm of **constraint programming (CP)** allow us: 1. to model relationships between single decisions smartly; and 2. give an answer efficiently.

A **constraint satisfaction problem (CSP)** consists of a set of variables and constraints stablishing relationships between them. Each variable has a finite of possible values (its domain). The goal is to assign values to the variables in its domains satisfying all the constraints. A more general version, called **constraint optimization programming (C0P)**, finds amongst all the feasible solutions the one that optimizes some measure, called 'objective function'. 

What is necessary to model a CP?
======================

A typical CP is defined by the following elements:

**Variables**: 

**Constraints**:

Moreover, if we want to model an optimization problem we also need an objective function.

Example:

A cryptarithmetic puzzle is a mathematical exercise where the digits of some numbers are represented by letters (or symbols). Each letter represents a unique digit. The goal is to find the digits such that a given mathematical equation is verified. 

For example, we aim to allocate to the letters S,E,N,D,M,O,R,Y a digit between 0 and 9, being all the letters allocated to a different digit and such that the expression: 

SEND + MORE = MONEY

is satisfied. This problem lies into the setting of **constraint satisfaction problem (CSP)**. Here the variables are each letter S,E,N,D,M,O,R,Y and their domain is {0,1,2,...,9}. The constraints represents the fact that




The cpmpy implementation for this CSP looks like:




A possible feasible allocation/solution is 

.. code:: python
  S= 
  E=
  N=
  D=
  M=
  O=
  R=
  Y= 
  
 .. code:: javascript

    function whatever() {
        return "such color"
    }

Note that we can find an slightly different version of this problem by optimizing an objective function, for example, optimizing the number formed by the word MONEY:

max 10000*M + 1000*O + 100*N + 10*E + 1*Y.

The cpmpy implementation for this COP looks like:

But this is just a toy example. In the following we are going to consider more difficult problems and real-world applications.


References
=====

.. Add some references

To learn more about theory and practice of constraint programming you may want to check some references:

1. Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006). Handbook of constraint programming. Elsevier.
2. Apt, K. (2003). Principles of constraint programming. Cambridge university press.
