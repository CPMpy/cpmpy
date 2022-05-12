---
Title:    N-Queens
Proposer: Bilal Syed Hussain
Category: Games and puzzles
---

Overview
========

Can $n$ queens (of the same colour) be placed on a $n\times n$ chessboard so that none of the  queens can attack each other?

In chess a queen attacks other squares on the same row, column, or either diagonal as itself. So the $n$-queens problem is to find a set of $n$ locations on a chessboard, no two of which are on the same row, column or diagonal.  

<center>
<figure>
  <img src="assets/4queens.png" alt="solution to 4-queens">
  <figcaption>A solution to 4-queens</figcaption>
</figure>
</center>

A simple arithmetical observation may be helpful in understanding models. Suppose a queen is represented by an ordered pair (α,β), the value α represents the queen’s column, and β its row on the chessboard. Then two queens do not attack each other iff they have different values of *all* of α, β, α-β, and α+β. It may not be intuitively obvious that chessboard diagonals correspond to sums and differences, but consider moving one square along the two orthogonal diagonals: in one direction the sum of the coordinates does not change, while in the other direction the difference does not change. (We do not suggest that pairs (α,β) is a good representation for solving.) 

The problem has inherent symmetry. That is, for any solution we obtain another solution by any of the 8 symmetries of the chessboard (including the identity) obtained by combinations of rotations by 90 degrees and reflections. 

The problem is extremely well studied in the mathematical literature. An outstanding survey from 2009 is by Bell & Stevens cite{Bell20091}.

See below for discussions of complexity problems with $n$-Queens. For closely related variants without these problems see {prob079}, [prob079], and {prob080}, [prob080].

Complexity
==========

Some care has to be taken when using the $n$-queens problem as a benchmark.  Here are some points to bear in mind:

* The $n$-queens problem is solvable for $n=1$ and $n \geq 4$. So the decision problem is solvable in constant time. 
* A solution to the $n$-queens problem for any $n \not = 2,3$ was given in 1874 by Pauls and can be found in Bell & Stevens' survey  cite{Bell20091}. It can be constructed in time $O(n)$ (assuming arithemetical operations on size $n$ are $O(1)$.) 
* Note that the parameter $n$ for $n$-queens only needs $\log(n)$ bits to specify, so actually $O(n)$ is exponential in the input size. I.e. it's not trivial to provide a witness of poly size in the input. 
* While the decision problem is easy, counting the number of solutions for given $n$ is not. Indeed Bell & Stevens cite{Bell20091} report that there is no closed form expression for it and that it is "beyond #P-Complete", citing cite{Hsiang200487}. (Oddly cite{chaiken-queens} report a closed form solution for the number of solutions to $n$-queens: it's unclear if this contradicts the earlier result, but more importantly it's not clear that this has better complexity than simply enumerating solutions.)
