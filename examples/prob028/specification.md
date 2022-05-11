---
Title:    Balanced Incomplete Block Designs
Proposer: Steven Prestwich
Category:
    - Design and configuration
    - Combinatorial mathematics
---

Balanced Incomplete Block Design (BIBD) generation is a standard combinatorial problem from design theory, originally used in the design of statistical experiments but since finding other applications such as cryptography. It is a special case of Block Design, which also includes Latin Square problems.

BIBD generation is described in most standard textbooks on combinatorics. A BIBD is defined as an arrangement of $v$ distinct objects into $b$ blocks such that each block contains exactly $k$ distinct objects, each object occurs in exactly $r$ different blocks, and every two distinct objects occur together in exactly $\lambda$ blocks. Another way of defining a BIBD is in terms of its incidence matrix, which is a $v$ by $b$ binary matrix with exactly $r$ ones per row, $k$ ones per column, and with a scalar product
of $\lambda$ between any pair of distinct rows. A BIBD is therefore specified by its parameters $(v,b,r,k,\lambda)$. An example of a solution for $(7,7,3,3,1)$ is:

    0 1 1 0 0 1 0
    1 0 1 0 1 0 0
    0 0 1 1 0 0 1
    1 1 0 0 0 0 1
    0 0 0 0 1 1 1
    1 0 0 1 0 1 0
    0 1 0 1 1 0 0 

Lam's problem {prob025} is that of finding a BIBD with parameters $(111,111,11,11,1)$. 