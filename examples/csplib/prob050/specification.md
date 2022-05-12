---
Title:    Diamond-free Degree Sequences
Proposer: 
    - Alice Miller
    - Patrick Prosser
Category: Combinatorial mathematics
---

Given a simple undirected graph $G = (V,E)$, where $V$ is the set of vertices and $E$ the set of undirected edges, the edge {$u,v$} is in $E$ if and only if vertex u is adjacent to vertex $v  \in G$. The graph is simple in that there are no loop edges, i.e. we have no edges of the form {$v,v$}. Each vertex $v \in V$ has a degree dv i.e. the number of edges incident on that vertex. Consequently a graph has a degree sequence $d1,...,dn$, where $d_i >= d_{i+1}$. A diamond is a set of four vertices in $V$ such that there are at least five edges between those vertices. Conversely, a graph is diamond-free if it has no diamond as an induced subgraph, i.e. for every set of four vertices the number of edges between those vertices is at most four.

In our problem we have additional properties required of the degree sequences of the graphs, in particular that the degree of each vertex is greater than zero (i.e. isolated vertices are disallowed), the degree of each vertex is modulo $3$, and the sum of the degrees is modulo $12$ (i.e. $|E|$ is modulo $6$).

The problem is then for a given value of $n$, produce all unique degree sequences $d1,...,dn$ such that

* $d_i \ge d_{i+1}$
* each degree $d_i > 0$ and $d_i$ is modulo $3$
* the sum of the degrees is modulo $12$
* there exists a simple diamond-free graph with that degree sequence

Below, as an example, is the unique degree sequence for$ n=10$ along with the adjacency matrix of a diamond-free graph with that degree sequence.

      n = 10
      6 6 3 3 3 3 3 3 3 3 

      0 0 0 0 1 1 1 1 1 1 
      0 0 0 0 1 1 1 1 1 1 
      0 0 0 0 0 0 0 1 1 1 
      0 0 0 0 1 1 1 0 0 0 
      1 1 0 1 0 0 0 0 0 0 
      1 1 0 1 0 0 0 0 0 0 
      1 1 0 1 0 0 0 0 0 0 
      1 1 1 0 0 0 0 0 0 0 
      1 1 1 0 0 0 0 0 0 0 
      1 1 1 0 0 0 0 0 0 0 