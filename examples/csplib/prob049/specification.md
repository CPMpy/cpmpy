---
Title:    Number Partitioning
Proposer: Daniel Diaz
Category: Combinatorial mathematics
---


This problem consists in finding a partition of numbers $1..N$ into two sets A and B such that:

1. A and B have the same cardinality
2. sum of numbers in $A$ = sum of numbers in $B$
3. sum of squares of numbers in $A$ = sum of squares of numbers in $B$

There is no solution for $N < 8$.

Here is an example for$ N = 8$:$ A = (1,4,6,7)$ and $B = (2,3,5,8)$

Then from $N \>= 8$, there is no solution if $N$ is not a multiple of $4$.

### Generalisation

More constraints can thus be added, e.g also impose the equality on the sum of cubes, ...

Let $C_k$ be the constraint about the power $k$ defined as the equality :

$\Sigma_{i=1}^{N/2} A_i^k = \Sigma_{i=1}^{N/2} B_i^k$

Condition (a) corresponds to $k=0$. Condition (b) to $k=1$. Condition (c) to $k=2$.

This generalized problem can be seen as a conjunction of constraints $C_k$ until a power P $(C_0 /\\ C_1 /\\ ... /\\ C_P)$. The above problem corresponds to $P = 2$.

Empirically, I played with $P = 0, 1, 2, 3, 4$:

The sums of powers is known :

-   $\Sigma_{i=1}^{N} i^0 = N$
-   $\Sigma_{i=1}^{N} i^1 = N \* (N+1) / 2$
-   $\Sigma_{i=1}^{N} i^2 = N \* (N+1) \* (2\*N + 1) / 6$
-   $\Sigma_{i=1}^{N} i^3 = N^2 \* (N+1)^2 / 4$
-   $\Sigma_{i=1}^{N} i^4 = N \* (N+1) \* (6\*N^3 + 9\*N^2 + N - 1) / 30$


Recall in our case we need the half sums. The problem has no solution if the above sums are not even numbers. For P = 0 this implies N is a multiple of 2 (groups A and B have the same cardinality). For P = 1 (knowing N is multiple of 2 due to P = 0) then N \* (N + 1) / 2 is even iff N is multiple of 4.

Here are the first solutions computed:

-   $P = 0$: first solutions found for $N = 2, 4, 6, 8, 10, 12, ...$ (obviously for every multiple of 2)
-   $P = 1$: first solutions found for $N = 4, 8, 12, 16, 20, 24, 28, 32$ (then for every multiple of 4 ?)
-   $P = 2$: first solutions found for $N = 8, 12, 16, 20, 24, 28, 32, 36$ (then for every multiple of 4 ?)
-   $P = 3$: first solutions found for$ N = 16, 24, 32, 40 $(then for every multiple of 8 ?)
-   $P = 4$: first solutions found for$ N = 32, 40, 48, 56, 64$ (then forevery multiple of 8 ?)

From these tests, it seems the smallest N for which a solution exists is $2^{P+1}$. Can this be proved ?

After that, it seems there are only solutions for N multiple of 2 (P= 0), 4 (P = 1 or 2), 8 (P = 3 or 4). Is this a constant depending on P ?

Another way to generalize this problem consists in increasing the numbers of groups (for instance consider 3 groups A, B, C).

