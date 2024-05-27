import copy
from functools import reduce

from pycsp3.parser.callbacks import Callbacks
from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.enums import (TypeConditionOperator, TypeArithmeticOperator, TypeUnaryArithmeticOperator, TypeLogicalOperator,
                                            TypeOrderedOperator, TypeRank, TypeObj)
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import (Node)

from pycsp3.parser.xentries import XVar
from pycsp3.tools.utilities import _Star

import cpmpy as cp
from cpmpy.expressions.utils import is_any_list, get_bounds, is_boolexpr


class CallbacksCPMPy(Callbacks):

    def __init__(self):
        super().__init__()
        self.cpm_model = cp.Model()
        self.cpm_variables = dict()
        self.print_general_methods = False
        self.print_specific_methods = False

    def get_condition(self, condition):
        map = {"LT": "<", "LE": "<=", "EQ": "=", "GE": ">=", "GT": ">"}
        if condition.operator.name not in map:
            raise ValueError("Unknown condition operator", condition.operator.name, "expected any of", set(map.keys()))

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        if min_value == 0 and max_value == 1:
            #boolvar
            newvar = cp.boolvar(name=x.id)
        else:
            newvar = cp.intvar(min_value, max_value, name=x.id)
        self.cpm_variables[x] = newvar

    def var_integer(self, x: Variable, values: list[int]):
        mini = min(values)
        maxi = max(values)
        if mini == 0 and maxi == 1:
            #boolvar
            newvar = cp.boolvar(name=x.id)
        else:
            newvar = cp.intvar(mini, maxi, name=x.id)
        self.cpm_variables[x] = newvar
        nbvals = maxi - mini + 1
        if nbvals < len(values):
            # only do this if there are holes in the domain
            self.cpm_model += cp.InDomain(newvar, values)

    def load_instance(self, discarded_classes=None):
        return self.cpm_model, self.cpm_variables

    def ctr_true(self, scope: list[Variable]):
        return cp.BoolVal(True)

    def ctr_false(self, scope: list[Variable]):
        assert False, "Problem as a constraint with 0 supports: " + str(scope)

    def ctr_intension(self, scope: list[Variable], tree: Node):
        cons = self.intentionfromtree(tree)
        self.cpm_model += cons

    funcmap = {
        # Arithmetic
        "neg": (1, lambda x: -x),
        "abs": (1, lambda x: abs(x)),
        "add": (0, lambda x: cp.sum(x)),
        "sub": (2, lambda x, y: x - y),
        "mul": (0, lambda x: reduce((lambda x, y: x*y),x)),
        "div": (2, lambda x, y: x // y),
        "mod": (2, lambda x, y: x % y),
        "sqr": (1, lambda x: x ** 2),
        "pow": (2, lambda x, y: x ** y),
        "min": (0, lambda x: cp.min(x)),
        "max": (0, lambda x: cp.max(x)),
        "dist": (2, lambda x, y: abs(x - y)),
        # Relational
        "lt": (2, lambda x, y: x < y),
        "le": (2, lambda x, y: x <= y),
        "ge": (2, lambda x, y: x >= y),
        "gt": (2, lambda x, y: x > y),
        "ne": (2, lambda x, y: x != y),
        "eq": (0, lambda x: x[0] == x[1] if len(x) == 2 else cp.AllEqual(x)),
        # Set
        'in': (2, lambda x, y: cp.InDomain(x, y)),
        # Logic
        "not": (1, lambda x: ~x),
        "and": (0, lambda x: cp.all(x)),
        "or": (0, lambda x: cp.any(x)),
        "xor": (0, lambda x: cp.Xor(x)),
        "iff": (0, lambda x: cp.all(a == b for a, b in zip(x[:-1], x[1:]))),
        "imp": (2, lambda x, y: x.implies(y)),
        # control
        "if": (3, lambda b, x, y: cp.IfThenElse(b, x, y) if is_boolexpr(x) and is_boolexpr(y)
                                                         else cp.IfThenElseNum(b,x,y))
    }

    def intentionfromtree(self, node):
        if isinstance(node, Node):
            if node.type.lowercase_name == 'var':
                return self.cpm_variables[node.cnt]
            if node.type.lowercase_name == 'int':
                return node.cnt
            arity, cpm_op = self.funcmap[node.type.lowercase_name]
            cpm_args = []
            for arg in node.cnt:
                cpm_args.append(self.intentionfromtree(arg))
            if arity != 0:
                return cpm_op(*cpm_args)
            return cpm_op(cpm_args)
        else:
            return node

    def ctr_primitive1a(self, x: Variable, op: TypeConditionOperator, k: int):
        assert op.is_rel()
        x = self.get_cpm_var(x)
        arity, cpm_op = self.funcmap[op.name.lower()]
        if arity == 2:
            self.cpm_model += cpm_op(x, k)
        elif arity == 0:
            self.cpm_model += cpm_op([x, k])
        else:
            self._unimplemented(x, op, k)

    def ctr_primitive1b(self, x: Variable, op: TypeConditionOperator, term: list[int] | range):
        assert op.is_set()
        x = self.get_cpm_var(x)
        arity, cpm_op = self.funcmap[op.name.lower()]
        if isinstance(term, range):
            term = [x for x in term] #list from range
        if arity == 2:
            self.cpm_model += cpm_op(x, term)
        elif arity == 0:
            self.cpm_model += cpm_op([x, term])
        else:
            self._unimplemented(x, op, term)


    def ctr_primitive1c(self, x: Variable, aop: TypeArithmeticOperator, p: int, op: TypeConditionOperator, k: int):
        assert op.is_rel()
        self.ctr_primitive3(x, aop, p, op, k) #for cpmpy ints and vars are just interchangeable..

    def ctr_primitive2a(self, x: Variable, aop: TypeUnaryArithmeticOperator, y: Variable):
        self._unimplemented(x, aop, y)

    def ctr_primitive2b(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator, k: int):
        #(x aop y) rel k
        self.ctr_primitive3(x, aop, y, op, k) #for cpmpy ints and vars are just interchangeable..

    def ctr_primitive2c(self, x: Variable, aop: TypeArithmeticOperator, p: int, op: TypeConditionOperator, y: Variable):
        #(x aop p) op y
        assert op.is_rel()
        self.ctr_primitive3(x, aop, p, op, y) #for cpmpy ints and vars are just interchangeable..

    def ctr_primitive3(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator, z: Variable):
        # (x aop y) op z
        assert op.is_rel()
        arity_op, cpm_op = self.funcmap[(op.name).lower()]
        arity, cpm_aop = self.funcmap[aop.name.lower()]
        x = self.get_cpm_var(x)
        y = self.get_cpm_var(y)
        z = self.get_cpm_var(z)
        if arity == 2:
            if arity_op == 2:
                self.cpm_model += cpm_op(cpm_aop(x, y), z)
            else:  # eq is arity 0, because of allequal global
                self.cpm_model += cpm_op([cpm_aop(x, y), z])
        elif arity == 0:
            if arity_op == 2:
                self.cpm_model += cpm_op(cpm_aop([x, y]), z)
            else:  # eq is arity 0, because of allequal global
                self.cpm_model += cpm_op([cpm_aop([x, y]), z])

    def ctr_logic(self, lop: TypeLogicalOperator, scope: list[Variable]):  # lop(scope)
        if lop == TypeLogicalOperator.AND:
            self.cpm_model += self.get_cpm_vars(scope)
        elif lop == TypeLogicalOperator.OR:
            self.cpm_model += cp.any(self.get_cpm_vars(scope))
        elif lop ==TypeLogicalOperator.IFF:
            assert len(scope) == 2
            a,b = scope
            self.cpm_model += self.get_cpm_var(a) == self.get_cpm_var(b)
        elif lop == TypeLogicalOperator.IMP:
            assert len(scope) == 2
            a,b = scope
            self.cpm_model += self.get_cpm_var(a).implies(self.get_cpm_var(b))
        elif lop == TypeLogicalOperator.XOR:
            self.cpm_model += cp.Xor(self.get_cpm_vars(scope))
        else:
            self._unimplemented(lop, scope)

    def ctr_logic_reif(self, x: Variable, y: Variable, op: TypeConditionOperator, k: int | Variable):  # x = y <op> k
        from cpmpy.expressions.utils import eval_comparison
        assert op.is_rel()
        self.cpm_model += eval_comparison(op.to_str(),
                                          self.get_cpm_var(x) == self.get_cpm_var(y),
                                          self.get_cpm_var(k))

    def ctr_logic_eqne(self, x: Variable, op: TypeConditionOperator, lop: TypeLogicalOperator, scope: list[Variable]):  # x = lop(scope) or x != lop(scope)
        assert op in (TypeConditionOperator.EQ, TypeConditionOperator.NE) # TODO: what is this???
        self._unimplemented(x, op, lop, scope)

    def ctr_extension_unary(self, x: Variable, values: list[int], positive: bool, flags: set[str]):
        if positive:
            #unary table constraint is just an inDomain
            if len(values) == 1:
                self.cpm_model += self.get_cpm_var(x) == values[0]
            else:
                self.cpm_model += cp.InDomain(self.get_cpm_var(x), values)
        else:
            # negative, so not in domain
            if len(values) == 1:
                self.cpm_model += self.get_cpm_var(x) != values[0]
            else:
                self.cpm_model += ~cp.InDomain(self.get_cpm_var(x), values)

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        def strwildcard(x):
            if isinstance(x,_Star):
                return '*'
            return x
        if 'starred' in flags:
            cpm_vars = self.vars_from_node(scope)
            exttuples = [tuple([strwildcard(x) for x in tup]) for tup in tuples]
            if positive:
                self.cpm_model += cp.ShortTable(cpm_vars, exttuples)
            else:
                self.cpm_model += ~cp.ShortTable(cpm_vars, exttuples)
        else:
            cpm_vars = self.vars_from_node(scope)
            if positive:
                self.cpm_model += cp.Table(cpm_vars, tuples)
            else:
                self.cpm_model += ~cp.Table(cpm_vars, tuples)


    def ctr_regular(self, scope: list[Variable], transitions: list, start_state: str, final_states: list[str]):
        self._unimplemented(scope, transitions, start_state, final_states) # TODO: add after Helene PR

    def ctr_mdd(self, scope: list[Variable], transitions: list):
        self._unimplemented(scope, transitions) # TODO: add after Helen PR

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        if excepting is None:
            cpm_exprs = self.exprs_from_node(scope)
            return cp.AllDifferent(cpm_exprs)
        elif excepting == [0]:
            return cp.AllDifferentExcept0(self.exprs_from_node(scope))
        else:
            self._unimplemented(scope, excepting) # TODO: parse to AllDifferentExceptN

    def ctr_all_different_lists(self, lists: list[list[Variable]], excepting: None | list[list[int]]):
        self.cpm_model += cp.AllDifferentLists([self.get_cpm_vars(lst) for lst in lists]) # TODO: what about the excepting arg??

    def ctr_all_different_matrix(self, matrix: list[list[Variable]], excepting: None | list[int]):
        if excepting is None:
            cpm_exprs = self.exprs_from_node(matrix)
            return cp.AllDifferent(cpm_exprs)
        elif excepting == [0]:
            return cp.AllDifferentExcept0(self.exprs_from_node(matrix))
        else:
            self._unimplemented(excepting)

    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        self.cpm_model += cp.AllEqual(self.get_cpm_exprs(scope))

    def ctr_ordered(self, lst: list[Variable], operator: TypeOrderedOperator, lengths: None | list[int] | list[Variable]):
        cpm_vars = self.get_cpm_vars(lst)
        if operator == TypeOrderedOperator.INCREASING:
            self.cpm_model += cp.Increasing(cpm_vars)
        elif operator == TypeOrderedOperator.STRICTLY_INCREASING:
            self.cpm_model += cp.IncreasingStrict(cpm_vars)
        elif operator == TypeOrderedOperator.DECREASING:
            self.cpm_model += cp.Decreasing(cpm_vars)
        elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
            self.cpm_model += cp.DecreasingStrict(cpm_vars)
        else:
            self._unimplemented(lst, operator, lengths)

    def ctr_lex_limit(self, lst: list[Variable], limit: list[int], operator: TypeOrderedOperator):  # should soon enter XCSP3-core
        self._unimplemented(lst, limit, operator)

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        self._unimplemented(lists, operator) # TODO: after merge of Dimos PR

    def ctr_lex_matrix(self, matrix: list[list[Variable]], operator: TypeOrderedOperator):
        self._unimplemented(matrix, operator) # TODO: after merge of Dimos PR

    def ctr_precedence(self, lst: list[Variable], values: None | list[int], covered: bool):
        self._unimplemented(lst, values, covered)

    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        cpm_vars = []
        if isinstance(lst[0], XVar):
            for xvar in lst:
                cpm_vars.append(self.cpm_variables[xvar])
        else:
            cpm_vars = self.exprs_from_node(lst)
        arity, op = self.funcmap[condition.operator.name.lower()]
        if hasattr(condition, "variable"):
            rhs = condition.variable
        elif hasattr(condition, 'value'):
            rhs = condition.value
        elif hasattr(condition, 'max'):
            #operator = in
            rhs = [x for x in range(condition.min, condition.max + 1)]
        else:
            pass
        cpm_rhs = self.get_cpm_var(rhs)
        if coefficients is None:
            cpsum = cp.sum(cpm_vars)
        else:
            cp_coeffs = self.get_cpm_vars(coefficients)
            cpsum = cp.sum(cp.cpm_array(cpm_vars) * cp_coeffs)
        if arity == 0:
            self.cpm_model += op([cpsum, cpm_rhs])
        else:
            self.cpm_model += op(cpsum, cpm_rhs)

    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        cpm_vars = self.get_cpm_vars(lst)
        cpm_vals = self.get_cpm_vars(values)
        self._unimplemented(lst, values, condition)

    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        cpm_vars = self.get_cpm_vars(lst)
        self.cpm_model += (cp.Count(cpm_vars, value) >= k)

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        cpm_vars = self.get_cpm_vars(lst)
        self.cpm_model += (cp.Count(cpm_vars, value) <= k)

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        cpm_vars = self.get_cpm_exprs(lst)
        self.cpm_model += (cp.Count(cpm_vars, value) == k)

    def ctr_among(self, lst: list[Variable], values: list[int], k: int | Variable):
        self._unimplemented(lst, values, k) # TODO: add after Ignace PR

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        arity, op = self.funcmap[condition.operator.name.lower()]
        cpm_rhs = self.cpm_variables[condition.right_operand()]
        if excepting is None:
            if arity == 2: #should always be a comparison
                self.cpm_model += op(cp.NValue(self.get_cpm_exprs(lst)), cpm_rhs)
            else:
                assert False, "condition should be a comparision"
        else:
            self._unimplemented()
    def ctr_not_all_qual(self, lst: list[Variable]):
        cpm_vars = self.get_cpm_vars(lst)
        self.cpm_model += ~cp.AllEqual(cpm_vars)

    def ctr_cardinality(self, lst: list[Variable], values: list[int] | list[Variable], occurs: list[int] | list[Variable] | list[range], closed: bool):
        if closed == False:
            self.cpm_model += cp.GlobalCardinalityCount(self.get_cpm_exprs(lst), self.get_cpm_exprs(values), self.get_cpm_exprs(occurs))
        else:
            self._unimplemented()

    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        cpm_vars = self.get_cpm_vars(lst)
        arity, op = self.funcmap[condition.operator.name.lower()]
        cpm_rhs = self.cpm_variables[condition.variable]
        if arity == 0:
            self.cpm_model += op([cp.Minimum(cpm_vars), cpm_rhs])
        else:
            self.cpm_model += op(cp.Minimum(cpm_vars), cpm_rhs)

    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        cpm_vars = self.get_cpm_vars(lst)
        arity, op = self.funcmap[condition.operator.name.lower()]
        cpm_rhs = self.cpm_variables[condition.variable]
        if arity == 0:
            self.cpm_model += op([cp.Maximum(cpm_vars), cpm_rhs])
        else:
            self.cpm_model += op(cp.Maximum(cpm_vars), cpm_rhs)

    def ctr_minimum_arg(self, lst: list[Variable] | list[Node], condition: Condition, rank: TypeRank):  # should enter XCSP3-core
        self._unimplemented(lst, condition, rank)

    def ctr_maximum_arg(self, lst: list[Variable] | list[Node], condition: Condition, rank: TypeRank):  # should enter XCSP3-core
        self._unimplemented(lst, condition, rank)

    def ctr_element(self, lst: list[Variable] | list[int], i: Variable, condition: Condition):
        cpm_list = self.get_cpm_vars(lst)
        arity, op = self.funcmap[condition.operator.name.lower()]
        if hasattr(condition, 'variable'):
            cpm_rhs = self.cpm_variables[condition.variable]
        elif hasattr(condition, 'value'):
            cpm_rhs = condition.value
        else:
            self._unimplemented(lst, condition)
        cpm_index = self.cpm_variables[i]
        if arity == 0:
            self.cpm_model += op([cp.Element(cpm_list,cpm_index), cpm_rhs])
        else:
            self.cpm_model += op(cp.Element(cpm_list, cpm_index), cpm_rhs)

    def ctr_element_matrix(self, matrix: list[list[Variable]] | list[list[int]], i: Variable, j: Variable, condition: Condition):
        self._unimplemented(matrix, i, j, condition) # TOOD: implement, already in CPMpy

    def ctr_channel(self, lst1: list[Variable], lst2: None | list[Variable]):
        if lst2 is None:
            raise NotImplementedError()
        cpm_vars1 = self.get_cpm_vars(lst1)
        cpm_vars2 = self.get_cpm_vars(lst2)
        # make lists same length, last part is irrelevant if not same length
        if len(cpm_vars2) > len(cpm_vars1):
            cpm_vars2 = cpm_vars2[0:len(cpm_vars1)]
        elif len(cpm_vars1) > len(cpm_vars2):
            cpm_vars1 = cpm_vars1[0:len(cpm_vars2)]
        self.cpm_model += cp.Inverse(cpm_vars1, cpm_vars2)

    def ctr_channel_value(self, lst: list[Variable], value: Variable):
        self._unimplemented(lst, value)

    def ctr_nooverlap(self, origins: list[Variable], lengths: list[int] | list[Variable],
                      zero_ignored: bool):  # in XCSP3 competitions, no 0 permitted in lengths
        self._unimplemented(origins, lengths, zero_ignored)

    def ctr_nooverlap_multi(self, origins: list[list[Variable]], lengths: list[list[int]] | list[list[Variable]], zero_ignored: bool):
        self._unimplemented(origins, lengths, zero_ignored)

    def ctr_nooverlap_mixed(self, xs: list[Variable], ys: list[Variable], lx: list[Variable], ly: list[int], zero_ignored: bool):
        self._unimplemented(xs, ys, lx, ly, zero_ignored) # TODO: add after merge Ignace PR

    def ctr_cumulative(self, origins: list[Variable], lengths: list[int] | list[Variable], heights: list[int] | list[Variable], condition: Condition):
        #self._unimplemented(origins, lengths, heights, condition)
        cpm_start = self.get_cpm_exprs(origins)
        cpm_durations = self.get_cpm_exprs(lengths)
        cpm_demands = self.get_cpm_exprs(heights)
        cpm_cap = self.get_cpm_var(condition.right_operand())
        cpm_ends = []
        for s,d in zip(cpm_start, cpm_durations):
            expr = s + d
            cpm_ends.append(cp.intvar(*get_bounds(expr)))

        if condition.operator.name == 'LE':
            self.cpm_model += cp.Cumulative(cpm_start, cpm_durations, cpm_ends, cpm_demands, cpm_cap)
        else:
            # post decomposition directly
            # be smart and chose task or time decomposition
            if max(get_bounds(cpm_ends)) >= 100:
                self._cumulative_task_decomp(cpm_start, cpm_durations, cpm_ends, heights, cpm_cap, condition.operator.to_str())
            else:
                self._cumulative_task_decomp(cpm_start, cpm_durations, cpm_ends, heights, cpm_cap, condition.operator.to_str())

    def _cumulative_task_decomp(self, cpm_start, cpm_duration, cpm_ends, cpm_demands, capacity, condition):
        from cpmpy.expressions.utils import eval_comparison
        cpm_demands = cp.cpm_array(cpm_demands)
        # ensure durations are satisfied
        for s,d,e in zip(cpm_start, cpm_duration, cpm_ends):
            self.cpm_model += s + d == e

        # task decomposition
        for s,d,e in zip(cpm_start, cpm_duration, cpm_ends):
            # find overlapping tasks
            total_running = cp.sum(cpm_demands * ((cpm_start <= s) & (cpm_ends > s)))
            self.cpm_model += eval_comparison(condition, total_running, capacity)

    def _cumulative_time_decomp(self, cpm_start, cpm_duration, cpm_ends, cpm_demands, capacity, condition):
        from cpmpy.expressions.utils import eval_comparison
        cpm_demands = cp.cpm_array(cpm_demands)

        # ensure durations are satisfied
        for s, d, e in zip(cpm_start, cpm_duration, cpm_ends):
            self.cpm_model += s + d == e

        lb = min(get_bounds(cpm_start)[0])
        ub = max(get_bounds(cpm_ends)[1])
        # time decomposition
        for t in range(lb,ub+1):
            total_running = cp.sum(cpm_demands * ((cpm_start <= t) & (cpm_ends > t)))
            self.cpm_model += eval_comparison(condition, total_running, capacity)

    def ctr_binpacking(self, lst: list[Variable], sizes: list[int], condition: Condition):
        from cpmpy.expressions.utils import eval_comparison

        cpm_vars = self.get_cpm_vars(lst)
        rhs = self.get_cpm_var(condition.right_operand())

        for bin in range(1, len(cpm_vars)+1):
            self.cpm_model += eval_comparison(condition.operator.to_str(),
                                              cp.sum((cpm_vars == bin) * sizes),
                                              rhs)
    def ctr_binpacking_limits(self, lst: list[Variable], sizes: list[int], limits: list[int] | list[Variable]):
        from cpmpy.expressions.utils import eval_comparison

        cpm_vars = self.get_cpm_vars(lst)

        for bin, lim in enumerate(limits):
            self.cpm_model += eval_comparison("<=",
                                              cp.sum((cpm_vars == (bin+1)) * sizes),
                                              lim)

    def ctr_binpacking_loads(self, lst: list[Variable], sizes: list[int], loads: list[int] | list[Variable]):
        from cpmpy.expressions.utils import eval_comparison

        cpm_vars = self.get_cpm_vars(lst)
        cpm_loads = self.get_cpm_vars(loads)

        for bin, load in enumerate(cpm_loads):
            self.cpm_model += eval_comparison("==",
                                              cp.sum((cpm_vars == (bin + 1)) * sizes),
                                              load)

    def ctr_binpacking_conditions(self, lst: list[Variable], sizes: list[int], conditions: list[Condition]):  # not in XCSP3-core
        self._unimplemented(lst, sizes, conditions)

    def ctr_knapsack(self, lst: list[Variable], weights: list[int], wcondition: Condition, profits: list[int], pcondition: Condition):
        from cpmpy.expressions.utils import eval_comparison

        vars = self.get_cpm_vars(lst)
        cpm_weight = self.get_cpm_var(wcondition.right_operand())
        cpm_profit = self.get_cpm_var(pcondition.right_operand())

        total_weight = cp.sum(vars * weights)
        total_profit = cp.sum(vars * profits)
        self.cpm_model += eval_comparison(wcondition.operator.to_str(), total_weight, cpm_weight)
        self.cpm_model += eval_comparison(pcondition.operator.to_str(), total_profit, cpm_profit)


    def ctr_flow(self, lst: list[Variable], balance: list[int] | list[Variable], arcs: list, capacities: None | list[range]):  # not in XCSP3-core
        self._unimplemented(lst, balance, arcs, capacities)

    def ctr_flow_weighted(self, lst: list[Variable], balance: list[int] | list[Variable], arcs: list, capacities: None | list[range],
                          weights: list[int] | list[Variable],
                          condition: Condition):  # not in XCSP3-core
        self._unimplemented(lst, balance, arcs, capacities, weights, condition)

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        self.cpm_model += cp.Table(self.get_cpm_vars(lst), [values])

    def ctr_clause(self, pos: list[Variable], neg: list[Variable]):  # not in XCSP3-core
        self._unimplemented(pos, neg)

    def ctr_circuit(self, lst: list[Variable], size: None | int | Variable):  # size is None in XCSP3 competitions
        return cp.SubCircuitWithStart(lst, start_index=0)

    # # # # # # # # # #
    # All methods about objectives to be implemented
    # # # # # # # # # #

    def obj_minimize(self, term: Variable | Node):
        if isinstance(term, Node):
            term = term.cnt
        self.cpm_model.minimize(self.get_cpm_var(term))

    def obj_maximize(self, term: Variable | Node):
        self.cpm_model.maximize(self.get_cpm_exprs(term)[0])

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        if obj_type == TypeObj.SUM:
            if coefficients is None:
                self.cpm_model.minimize(cp.sum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.minimize(cp.sum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.PRODUCT:
            if coefficients is None:
                self.cpm_model.minimize(reduce((lambda x, y: x*y),self.get_cpm_exprs(terms)))
            else:
                self._unimplemented(obj_type, terms, coefficients)
        elif obj_type == TypeObj.EXPRESSION:
            self._unimplemented(obj_type, terms, coefficients)
        elif obj_type == TypeObj.MAXIMUM:
            if coefficients is None:
                self.cpm_model.minimize(cp.Maximum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.minimize(cp.Maximum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.MINIMUM:
            if coefficients is None:
                self.cpm_model.minimize(cp.Minimum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.minimize(cp.Minimum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.NVALUES:
            if coefficients is None:
                self.cpm_model.minimize(cp.NValue(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.minimize(cp.NValue(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        else:
            self._unimplemented(obj_type, terms, coefficients)

    def obj_maximize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        if obj_type == TypeObj.SUM:
            if coefficients is None:
                self.cpm_model.maximize(cp.sum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.maximize(cp.sum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.PRODUCT:
            if coefficients is None:
                self.cpm_model.maximize(reduce((lambda x, y: x * y), self.get_cpm_exprs(terms)))
            else:
                self._unimplemented(obj_type, terms, coefficients)
        elif obj_type == TypeObj.EXPRESSION:
            self._unimplemented(obj_type, terms, coefficients)
        elif obj_type == TypeObj.MAXIMUM:
            if coefficients is None:
                self.cpm_model.maximize(cp.Maximum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.maximize(cp.Maximum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.MINIMUM:
            if coefficients is None:
                self.cpm_model.maximize(cp.Minimum(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.maximize(cp.Minimum(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        elif obj_type == TypeObj.NVALUES:
            if coefficients is None:
                self.cpm_model.maximize(cp.NValue(self.get_cpm_exprs(terms)))
            else:
                self.cpm_model.maximize(cp.NValue(cp.cpm_array(self.get_cpm_exprs(terms)) * coefficients))
        else:
            self._unimplemented(obj_type, terms, coefficients)

    def vars_from_node(self, scope):
        cpm_vars = []
        for var in scope:
            cpm_var = self.cpm_variables[var]
            cpm_vars.append(cpm_var)
        return cpm_vars

    def exprs_from_node(self, node):
        cpm_exprs = []
        for expr in node:
            cpm_expr = self.intentionfromtree(expr)
            cpm_exprs.append(cpm_expr)
        return cpm_exprs

    def get_cpm_var(self, x):
        if isinstance(x, XVar):
            return self.cpm_variables[x]
        else:
            return x #constants

    def get_cpm_vars(self, list):
        if isinstance(list[0], (XVar, int)):
            return [self.get_cpm_var(x) for x in list]
        else:
            return self.vars_from_node(list)

    def get_cpm_exprs(self, list):
        if isinstance(list[0], XVar):
            return [self.get_cpm_var(x) for x in list]
        else:
            return self.exprs_from_node(list)

    def end_instance(self):
        pass

    def load_annotation(self, annotation):
        pass
    def load_annotations(self, annotations):
        pass

    def load_objectives(self, objectives):
        pass
