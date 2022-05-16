---
Title:    Low Autocorrelation Binary Sequences
Proposer: Toby Walsh
Category: Combinatorial mathematics
---

These problems have many practical applications in communications and electrical engineering. The objective is to construct a binary sequence $S_i$ of length n that minimizes the autocorrelations between bits. Each bit in the sequence takes the value +1 or -1. With non-periodic (or open) boundary conditions, the k-th autocorrelation,  $C_k$ is defined to be $\sum\limits_{i=0}^{n-k-1} S_i * S_{i+k}$. With periodic (or cyclic) boundary conditions, the k-th autocorrelation, $C_k$ is defined to be $\sum\limits_{i=0}^{n-1} s_i * s_{i+k\ mod\ n } $. The aim is to minimize the sum of the squares of these autocorrelations. That is, to minimize $E=\sum\limits_{k=1}^{n-1} C_k^2$.