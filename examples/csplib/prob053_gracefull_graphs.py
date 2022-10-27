"""
K4P2 Graceful Graph in cpmpy.

http://www.csplib.org/Problems/prob053/

A labelling f of the nodes of a graph with q edges is graceful if f assigns each node a unique label
from 0,1,...,q and when each edge xy is labelled with |f(x)-f(y)|, the edge labels are all different.
Gallian surveys graceful graphs, i.e. graphs with a graceful labelling, and lists the graphs whose status
is known.

All-Interval Series is a special case of a graceful graph where the graph is a line.

This cpmpy model was written by Hakan Kjellerstrand (hakank@gmail.com)
See also my cpmpy page: http://hakank.org/cpmpy/

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
from cpmpy import *
import numpy as np

def gracefull_graphs(m,n,graph):
    graph = np.array(graph)

    model = Model()

    # variables
    nodes = intvar(0, m, shape=n, name="nodes")
    edges = intvar(1, m, shape=m, name="edges")

    # constraints
    model += np.abs(nodes[graph[:, 0]] - nodes[graph[:, 1]]) == edges

    model += (AllDifferent(edges))
    model += (AllDifferent(nodes))

    return model, (nodes, edges)


def get_data():
    # data
    m = 16
    n = 8

    graph = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [4, 7], [5, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    return m,n,graph

if __name__ == "__main__":

    data = get_data()
    model, (nodes, edges) = gracefull_graphs(*data)

    if model.solve():
      print(f"edges: {edges.value()}")
      print(f"nodes: {nodes.value()}")
    else:
      print("Model is unsatisfiable")