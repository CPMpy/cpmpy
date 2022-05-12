---
Title:   2cc Hadamard matrix Legendre pairs
Proposer:
- Ilias S. Kotsireas
Category: Combinatorial mathematics
---


For every odd positive integer $\ell$, (and $m=\displaystyle\frac{\ell - 1}{2}$) 
we define the 2cc Hadamard matrix Legendre pairs CSP
using the \{V,D,C\} format (Variables, Domains, Constraints) as follows:

* $V = \{ a_1, \cdots, a_\ell, b_1, \cdots, b_\ell \}$,  a set of $2 \cdot \ell$ variables 
* $D = \{ D_{a_1}, \ldots, D_{a_\ell}, D_{b_1}, \ldots, D_{b_\ell} \} $, a set of $2 \cdot \ell$ domains, all of them equal to $\{-1,+1\}$
* $C = \{ c_1, \ldots, c_{m}, c_{m+1}, c_{m+2} \}$,  a set of $m+2$ constraints, ($m$ quadratic constraints and 2 linear constraints)


The $m$ quadratic constraints are given by:
$$
        c_s := PAF(A,s)+PAF(B,s)=-2, \forall s=1,\ldots,m
$$
where PAF denotes the periodic autocorrelation function:  ($i+s$ is taken mod $\ell$, when is exceeds $\ell$)
$$
        A = [a_1,\ldots,a_\ell], \,\, PAF(A,s) = \sum_{i=1}^n a_i a_{i+s}
$$
$$
        B = [b_1,\ldots,b_\ell], \,\, PAF(B,s) = \sum_{i=1}^n b_i b_{i+s}
$$
The $2$ linear constraints are given by:
$$
    c_{m+1} := a_1 + \ldots + a_\ell = 1
$$
$$
    c_{m+2} := b_1 + \ldots + b_\ell = 1
$$

The 2cc Hadamard matrix Legendre pairs CSP for all odd $\ell = 3,\ldots,99$ are given in http://www.cargo.wlu.ca/CSP_2cc_Hadamard/ (and in the data section). There are 49 CSPs. All of them are known to have solutions.

It is conjectured that the 2cc Hadamard matrix Legendre pairs CSP has solutions, for every odd $\ell$, and this is linked to the famous Hadamard conjecture cite{Kotsireas}.