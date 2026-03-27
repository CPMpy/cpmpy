"""

Internal auxiliary functions to implement the linear decomposition of Table constraints. This happens as follows:
Step 1: The table T in the Table constraint is converted to a Multivalued Decision Diagram (MDD).
    - Step 1.1: The MDD is constructed by adding every row in T iteratively.
    - Step 1.2: The MDD is reduced by merging every equivalent node in the MDD (having the same set of completing suffixes).
Step 2: Using this MDD, the Table constraint is converted to flow constraints, with a total flow of 1, thereby enforcing that the variable corresponds to exactly one row in T.


"""

from collections import defaultdict
import enum
import numpy as np
import cpmpy as cp

class MDD:
    """
    Class identifying the MDD constructed during Step 1 of the decomposition algorithm.
    """
    def __init__(self):
        self.MDD_cache = {}
        self.MDD_cache_reverse = defaultdict(set)
        self.repeated_keys = set()

class TerminatingState(enum.Enum):
    """
        Class identifying a terminal state in the MDD.
    """
    SNK = 'snk'

class Prefix:
    """
        Class for prefixes (of rows in the table), which are used to uniquely identify nodes in the MDD.
    """
    def __init__(self, prefix):
        self.prefix = prefix

    def __eq__(self, other):
        if not isinstance(other, Prefix):
            return False
        else:
            return self.prefix == other.prefix

    def __hash__(self):
        return hash(self.prefix)

    def __deepcopy__(self):
        return Prefix(self.prefix)

class MDD_node:
    """
        Class for nodes in the MDD, which are characterized by a prefix identifying them, a level (depth from the source node),
        and a transition function that points to other prefixes / nodes depending on the next value.

    """
    def __init__(self, prefix, level, transition):
        self.prefix = prefix
        self.level = level
        self.transition = transition

    def __eq__(self, other):
        if not isinstance(other, MDD_node):
            return False

        if self.level != other.level:
            return False

        if self.transition.keys() != other.transition.keys():
            return False

        for key in self.transition:
            if self.transition[key] != other.transition[key]:
                return False

        return True

    __hash__ = object.__hash__

class MDD_node_key:
    """
        Wrapper class supporting a hashable canonical representation of an MDD node.
    """

    def __init__(self, node: "MDD_node"):
        self.level = node.level

        def key_value_repr(v):
            if isinstance(v, Prefix):
                return (v.prefix)
            elif isinstance(v, TerminatingState):
                return ("TerminatingState")
            else:
                raise TypeError(f"Unsupported transition value type: {type(v)}")

        self.transition = tuple(
            sorted((k, key_value_repr(v)) for k, v in node.transition.items())
        )

        self._hash = hash((self.level, self.transition))

    def __eq__(self, other):
        if not isinstance(other, MDD_node_key):
            return False

        return (self.level == other.level) and (self.transition == other.transition)

    def __hash__(self):
        return self._hash

def lookup_mdd(mdd_node, cache):
    if isinstance(mdd_node, Prefix):
        mdd_node = cache[mdd_node]
    return mdd_node

def add_row_to_mdd(row, mdd_node, mdd, level=0):
    """
    Auxiliary function performing the construction step of the MDD (step 1.1)
    :param row: the current row being added
    :param mdd_node: the current MDD node the algorithm is operating on
    :param mdd: the current MDD
    :param level: at which level of the MDD construction is happening
    :return: A prefix identifying the new MDD node after construction
    """
    mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)

    if len(row) == level:
        return TerminatingState.SNK

    prefix = Prefix(tuple(row[:level]))
    value = row[level]

    if value not in mdd_node.transition:
        mdd_node.transition[value] = add_row_to_mdd(row, MDD_node(tuple(row[:(level + 1)]), level + 1, {}), mdd, level + 1)
    else:
        mdd_node.transition[value] = add_row_to_mdd(row, mdd_node.transition[value], mdd, level + 1)

    mdd.MDD_cache[prefix] = mdd_node

    return prefix

def reduce_mdd(mdd_node, mdd, level):
    """
        Auxiliary function performing the reduction step of the MDD (step 1.1)
        :param mdd_node: the current MDD node the algorithm is operating on
        :param mdd: the current MDD
        :param level: at which level of the MDD construction is happening
        :return: A prefix identifying the new MDD node after reduction
    """
    if isinstance(mdd_node, TerminatingState):
        return mdd_node

    mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)

    B = {}
    for key in mdd_node.transition.keys():
        reduced_mdd = reduce_mdd(mdd_node.transition[key], mdd, level + 1)
        if reduced_mdd != False:
            B[key] = reduced_mdd

    if len(B.keys()) == 0:
        return False

    G = MDD_node(mdd_node.prefix, level, B)

    temp = None

    key = MDD_node_key(G)
    lookups = mdd.MDD_cache_reverse.get(key, set())

    for lookup in lookups:
        if lookup in mdd.repeated_keys:
            return lookup.__deepcopy__()
        else:
            temp = lookup.__deepcopy__()

    if temp is not None:
        mdd.repeated_keys.add(temp)
        return temp

    mdd.MDD_cache[Prefix(mdd_node.prefix)] = G

    return Prefix(mdd_node.prefix)

def construct_mdd(table):
    """
        Coordinator function for constructing and reducing the MDD (Step 1)
        :param table: The table to convert to an MDD
        :return: The resulting MDD (represented as a lookup hash table)
    """
    mdd = MDD()

    mdd_node = MDD_node(tuple(), 0, {})

    for i in range(0, table.shape[0]):
        row = table[i]
        mdd_node = add_row_to_mdd(row, mdd_node, mdd, 0)

    for lookup, node in mdd.MDD_cache.items():
        key = MDD_node_key(node)
        mdd.MDD_cache_reverse[key].add(lookup)
    mdd_node = lookup_mdd(mdd_node, mdd.MDD_cache)
    for key in mdd_node.transition.keys():
        reduced_mdd = reduce_mdd(mdd_node.transition[key], mdd, 1)
        mdd_node.transition[key] = reduced_mdd

    return mdd.MDD_cache

def dom_size(x):
    return x.ub - x.lb + 1

class Flow:
    """
        Class representing the ingoing and outgoing flow associated with a MDD node.
    """
    def __init__(self):
        self.flow_in = []
        self.flow_out = []

    def add_flow_in(self, value):
        self.flow_in.append(value)

    def add_flow_out(self, value):
        self.flow_out.append(value)

def get_corresponding_bv(column_index, X):
    """
        Retrieves the correct direct encoding variable given the column index of the Boolean table.
    :param column_index: The column index of the Boolean table
    :param X: The variable of the Table constraint
    :return: The corresponding Boolean variable
    """
    cumulative = 0

    for i, x in enumerate(X):
        d_size = dom_size(x)

        if column_index < cumulative + d_size:
            offset = column_index - cumulative

            return x == (offset + x.lb)

        cumulative += d_size

    return None

def mdd_to_flow(mdd_cache, X):
    """
    Performs Step 2 of the decomposition algorithm: converting the obtained MDD to flow constraints.
    """
    domains = np.array([dom_size(x) for x in X])
    lb = np.array([x.lb for x in X])

    no_columns = sum(domains)

    column_counter = {k: 0 for k in range(no_columns)}
    flow = {k: Flow() for k in mdd_cache.keys()}
    flow['snk'] = Flow()

    for key in mdd_cache.keys():
        val = lookup_mdd(mdd_cache[key], mdd_cache)
        for (k, v) in val.transition.items():
            column = sum(domains[:val.level]) + k - lb[val.level]
            if column < 0 or column >= len(column_counter):
                continue
            column_counter[column] += 1
            flow[key].add_flow_out((column, column_counter[column]))
            if isinstance(v, Prefix):
                flow[v].add_flow_in((column, column_counter[column]))
            if isinstance(v, TerminatingState):
                flow['snk'].add_flow_in((column, column_counter[column]))

    cons = []
    substitution = {}
    for key in column_counter.keys():
        if column_counter[key] == 0:
            continue
        elif column_counter[key] == 1:
            substitution[(key, 1)] = get_corresponding_bv(key, X)
        else:
            bvs = cp.boolvar(shape=column_counter[key], name=f"e_{key}")

            for n in range(1, column_counter[key] + 1):
                substitution[(key, n)] = bvs[n - 1]

    excluded = {Prefix(tuple()), "snk"}
    for key in sorted(
            (k for k in flow if k not in excluded),
            key=lambda k: (len(k.prefix), k.prefix)
    ):
        if (len(flow[key].flow_in) == 0) or (len(flow[key].flow_out) == 0):
            continue
        elif (len(flow[key].flow_in) == 1 and len(flow[key].flow_out) == 1
              and substitution[flow[key].flow_out[0]] == substitution[flow[key].flow_in[0]]):
            continue
        elif (len(flow[key].flow_in) == 1 and len(flow[key].flow_out) == 1
              and column_counter[flow[key].flow_in[0][0]] > 1 and column_counter[flow[key].flow_out[0][0]] > 1):
            substitution[flow[key].flow_out[0]] = substitution[flow[key].flow_in[0]]
        else:
            cons += [cp.sum([substitution[(c, m)] for (c, m) in flow[key].flow_in]) == cp.sum([substitution[(c, m)] for (c, m) in flow[key].flow_out])]

    cons += [cp.sum([substitution[(c, m)] for (c, m) in flow['snk'].flow_in]) == 1]
    cons += [cp.sum([substitution[(c, m)] for (c, m) in flow[Prefix(tuple())].flow_out]) == 1]

    for key in column_counter.keys():
        if column_counter[key] > 1:
            cons += [cp.sum([substitution[(key, n)] for n in range(1, column_counter[key] + 1)]) == get_corresponding_bv(key, X)]
    return cons


def filter_table(X, table):
    """
    Filters rows with values that are invalid for the domain out of the table.
    """
    lb = np.array([x.lb for x in X])
    ub = np.array([x.ub for x in X])

    mask = np.all((table >= lb) & (table <= ub), axis=1)
    new_table = table[mask]

    return new_table