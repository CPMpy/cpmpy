---
Title:    Steiner triple systems
Proposer: Francisco Azevedo
Category: Combinatorial mathematics
---


The ternary Steiner problem of order n consists of finding a set of $n.(n-1)/6$ triples of distinct integer elements in $\\{1,\dots,n\\}$ such that any two triples have at most one common element. It is a hypergraph problem coming from combinatorial mathematics cite{luneburg1989tools} where n modulo 6 has to be equal to 1 or 3  cite{lindner2011topics}. One possible solution for $n=7$ is {{1, 2, 3}, {1, 4, 5}, {1, 6, 7}, {2, 4, 6}, {2, 5, 7}, {3, 4, 7}, {3, 5, 6}}. The solution contains $7*(7-1)/6 = 7$ triples.

This is a particular case of the more general [Steiner system](http://www.win.tue.nl/~aeb/drg/graphs/S.html).

More generally still, you may refer to Balanced Incomplete Block Designs {prob028}. In fact, a Steiner Triple System with n elements is a BIBD$(n, n.(n-1)/6, (n-1)/2, 3, 1)$