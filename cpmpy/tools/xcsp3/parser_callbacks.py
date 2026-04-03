"""
Collection of callbacks for the PyCSP3 parser
"""

from functools import reduce

from pycsp3.parser.callbacks import Callbacks
from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.enums import (TypeConditionOperator, TypeArithmeticOperator, TypeUnaryArithmeticOperator,
                                            TypeLogicalOperator,
                                            TypeOrderedOperator, TypeRank, TypeObj)
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import (Node)

from pycsp3.parser.xentries import XVar
from pycsp3.tools.utilities import _Star

import cpmpy as cp
from cpmpy.tools.xcsp3 import globals as xglobals
from cpmpy import cpm_array
from cpmpy.expressions.utils import flatlist, get_bounds, is_boolexpr


class CallbacksCPMPy(Callbacks):  
    """
        A pycsp3-compatible callback for parsing XCSP3 instances into a CPMPy model.
    """

    def __init__(self):
        super().__init__()
        self.cpm_model = cp.Model()
        self.cpm_variables = dict()
        self.print_general_methods = False
        self.print_specific_methods = False

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        if min_value == 0 and max_value == 1:
            # boolvar
            newvar = cp.boolvar(name=x.id)
        else:
            newvar = cp.intvar(min_value, max_value, name=x.id)
        self.cpm_variables[x] = newvar

    def var_integer(self, x: Variable, values: list[int]):
        mini = min(values)
        maxi = max(values)
        if mini == 0 and maxi == 1:
            # boolvar
            newvar = cp.boolvar(name=x.id)
        else:
            newvar = cp.intvar(mini, maxi, name=x.id)
        self.cpm_variables[x] = newvar
        nbvals = maxi - mini + 1
        if len(values) < nbvals:
            # only do this if there are holes in the domain
            self.cpm_model += cp.InDomain(newvar, values)  # faster decomp, only works in positive context

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
        "mul": (0, lambda x: reduce((lambda x, y: x * y), x)),
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
        'in': (2, lambda x, y: cp.InDomain(x, y)),  # could be mixed context here!
        'notin': (2, lambda x, y: xglobals.NotInDomain(x, y)),  # could be mixed context here!
        'set': (0, lambda x: list(set(x))),
        # TODO 'notin' is the only other set operator (negative indomain)
        # Logic
        "not": (1, lambda x: ~x),
        "and": (0, lambda x: cp.all(x)),
        "or": (0, lambda x: cp.any(x)),
        "xor": (0, lambda x: cp.Xor(x)),
        "iff": (0, lambda x: cp.all(a == b for a, b in zip(x[:-1], x[1:]))),
        "imp": (2, lambda x, y: x.implies(y)),
        # control
        "if": (3, lambda b, x, y: cp.IfThenElse(b, x, y) if is_boolexpr(x) and is_boolexpr(y)
        else xglobals.IfThenElseNum(b, x, y))
    }

    def eval_cpm_comp(self, lhs, op: TypeConditionOperator, rhs):
        if (op == TypeConditionOperator.IN) or (op == TypeConditionOperator.NOTIN):
            assert isinstance(rhs, list), f"Expected list as rhs but got {rhs}"

        arity, cpm_op = self.funcmap[op.name.lower()]
        if arity == 2:
            return cpm_op(lhs, rhs)
        elif arity == 0:
            return cpm_op([lhs, rhs])
        else:
            raise ValueError(f"Expected operator of arity 0 or 2 but got {cpm_op} which is of arity {arity}")

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
        cpm_x = self.get_cpm_var(x)
        self.cpm_model += self.eval_cpm_comp(cpm_x, op, k)

    def ctr_primitive1b(self, x: Variable, op: TypeConditionOperator, term: list[int] | range):
        assert op.is_set()
        x = self.get_cpm_var(x)
        arity, cpm_op = self.funcmap[op.name.lower()]
        if isinstance(term, range):
            term = [x for x in term]  # list from range
        if arity == 2:
            self.cpm_model += cpm_op(x, term)
        elif arity == 0:
            self.cpm_model += cpm_op([x, term])
        else:
            self._unimplemented(x, op, term)

    def ctr_primitive1c(self, x: Variable, aop: TypeArithmeticOperator, p: int, op: TypeConditionOperator, k: int):
        assert op.is_rel()
        self.ctr_primitive3(x, aop, p, op, k)  # for cpmpy ints and vars are just interchangeable..

    def ctr_primitive2a(self, x: Variable, aop: TypeUnaryArithmeticOperator, y: Variable):
        # TODO this was unimplemented not sure if it should be aop(x) == y or x == aop(y)..
        arity, cpm_op = self.funcmap[aop.name.lower()]
        assert arity == 1, "unary operator expected"
        x = self.get_cpm_var(x)
        y = self.get_cpm_var(y)
        self.cpm_model += cpm_op(x) == y

    def ctr_primitive2b(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator, k: int):
        # (x aop y) rel k
        self.ctr_primitive3(x, aop, y, op, k)  # for cpmpy ints and vars are just interchangeable..

    def ctr_primitive2c(self, x: Variable, aop: TypeArithmeticOperator, p: int, op: TypeConditionOperator, y: Variable):
        # (x aop p) op y
        assert op.is_rel()
        self.ctr_primitive3(x, aop, p, op, y)  # for cpmpy ints and vars are just interchangeable..

    def ctr_primitive3(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator,
                       z: Variable):
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
        elif lop == TypeLogicalOperator.IFF:
            assert len(scope) == 2
            a, b = scope
            self.cpm_model += self.get_cpm_var(a) == self.get_cpm_var(b)
        elif lop == TypeLogicalOperator.IMP:
            assert len(scope) == 2
            a, b = scope
            self.cpm_model += self.get_cpm_var(a).implies(self.get_cpm_var(b))
        elif lop == TypeLogicalOperator.XOR:
            self.cpm_model += cp.Xor(self.get_cpm_vars(scope))
        else:
            self._unimplemented(lop, scope)

    def ctr_logic_reif(self, x: Variable, y: Variable, op: TypeConditionOperator, k: int | Variable):  # x = y <op> k
        assert op.is_rel()
        cpm_x, cpm_y, cpm_k = self.get_cpm_vars([x, y, k])
        self.cpm_model += self.eval_cpm_comp(cpm_x == cpm_y, op, cpm_k)

    def ctr_logic_eqne(self, x: Variable, op: TypeConditionOperator, lop: TypeLogicalOperator,
                       scope: list[Variable]):  # x = lop(scope) or x != lop(scope)
        cpm_x = self.get_cpm_var(x)
        arity, cpm_op = self.funcmap[(lop.name).lower()]
        if arity == 0:
            rhs = cpm_op(self.get_cpm_vars(scope))
        elif arity == 2:
            rhs = cpm_op(*self.get_cpm_vars(scope))
        else:
            self._unimplemented(lop, scope)

        self.cpm_model += self.eval_cpm_comp(cpm_x, op, rhs)

    def unroll(self, values):
        res = []
        for v in values:
            if isinstance(v, range):
                res.extend(v)
            else:
                res.append(v)
        return res

    def ctr_extension_unary(self, x: Variable, values: list[int], positive: bool, flags: set[str]):
        if len(values) == 1 and isinstance(values[0], range):
            values = list(eval(str(values[0])))

        if positive:
            # unary table constraint is just an inDomain
            if len(values) == 1:
                self.cpm_model += self.get_cpm_var(x) == values[0]
            else:
                self.cpm_model += cp.InDomain(self.get_cpm_var(x), self.unroll(values))  # faster decomp, only works in positive context
        else:
            # negative, so not in domain
            if len(values) == 1:
                self.cpm_model += self.get_cpm_var(x) != values[0]
            else:
                self.cpm_model += xglobals.NotInDomain(self.get_cpm_var(x), self.unroll(values))

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        def strwildcard(x):
            if isinstance(x, _Star):
                return '*'
            return x

        if 'starred' in flags:
            cpm_vars = self.vars_from_node(scope)
            exttuples = [tuple([strwildcard(x) for x in tup]) for tup in tuples]
            if positive:
                self.cpm_model += xglobals.RowSelectingShortTable(cpm_vars, exttuples)
            else:
                self.cpm_model += xglobals.NegativeShortTable(cpm_vars, exttuples)
        else:
            cpm_vars = self.vars_from_node(scope)
            if positive:
                self.cpm_model += xglobals.NonReifiedTable(cpm_vars, tuples)
            else:
                self.cpm_model += cp.NegativeTable(cpm_vars, tuples)

    def ctr_regular(self, scope: list[Variable], transitions: list, start_state: str, final_states: list[str]):
        self.cpm_model += xglobals.Regular(self.get_cpm_vars(scope), transitions, start_state, final_states)

    def ctr_mdd(self, scope: list[Variable], transitions: list):
        self.cpm_model += xglobals.MDD(self.get_cpm_vars(scope), transitions)

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        cpm_exprs = self.get_cpm_exprs(scope)
        if excepting is None:
            self.cpm_model += cp.AllDifferent(cpm_exprs)
        elif len(excepting) > 0:
            self.cpm_model += cp.AllDifferentExceptN(cpm_exprs, excepting)
        elif len(excepting) == 0:
            # just in case they get tricky
            self.cpm_model += cp.AllDifferent(cpm_exprs)
        else:  # unsupported for competition
            self._unimplemented(scope, excepting)

    def ctr_all_different_lists(self, lists: list[list[Variable]], excepting: None | list[list[int]]):
        if excepting is None:
            self.cpm_model += xglobals.AllDifferentLists([self.get_cpm_vars(lst) for lst in lists])
        else:
            self.cpm_model += xglobals.AllDifferentListsExceptN([self.get_cpm_vars(lst) for lst in lists], excepting)

    def ctr_all_different_matrix(self, matrix: list[list[Variable]], excepting: None | list[int]):
        import numpy as np
        for row in matrix:
            self.ctr_all_different(row, excepting)
        for col in np.array(matrix).T:
            self.ctr_all_different(col, excepting)

    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        if excepting is None:
            self.cpm_model += cp.AllEqual(self.get_cpm_exprs(scope))
        else:
            self.cpm_model += cp.AllEqualExceptN(self.get_cpm_exprs(scope), excepting)

    def ctr_ordered(self, lst: list[Variable], operator: TypeOrderedOperator,
                    lengths: None | list[int] | list[Variable]):

        cpm_vars = self.get_cpm_vars(lst)

        if lengths is None:
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

        # also handle the lengths parameter
        if lengths is not None:
            lengths = self.get_cpm_vars(lengths)
            if operator == TypeOrderedOperator.INCREASING:
                for x, l, y in zip(cpm_vars[:-1], lengths, cpm_vars[1:]):
                    self.cpm_model += x + l <= y
            elif operator == TypeOrderedOperator.STRICTLY_INCREASING:
                for x, l, y in zip(cpm_vars[:-1], lengths, cpm_vars[1:]):
                    self.cpm_model += x + l < y
            elif operator == TypeOrderedOperator.DECREASING:
                for x, l, y in zip(cpm_vars[:-1], lengths, cpm_vars[1:]):
                    self.cpm_model += x + l >= y
            elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
                for x, l, y in zip(cpm_vars[:-1], lengths, cpm_vars[1:]):
                    self.cpm_model += x + l > y
            else:
                self._unimplemented(lst, operator, lengths)

    def ctr_lex_limit(self, lst: list[Variable], limit: list[int],
                      operator: TypeOrderedOperator):  # should soon enter XCSP3-core
        self.ctr_lex([lst, limit], operator)

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        cpm_lists = [self.get_cpm_vars(lst) for lst in lists]
        if operator == TypeOrderedOperator.STRICTLY_INCREASING:
            self.cpm_model += cp.LexChainLess(cpm_lists)
        elif operator == TypeOrderedOperator.INCREASING:
            self.cpm_model += cp.LexChainLessEq(cpm_lists)
        elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
            rev_lsts = list(reversed(cpm_lists))
            self.cpm_model += cp.LexChainLess(rev_lsts)
        elif operator == TypeOrderedOperator.DECREASING:
            rev_lsts = list(reversed(cpm_lists))
            self.cpm_model += cp.LexChainLessEq(rev_lsts)
        else:
            self._unimplemented(lists, operator)

    def ctr_lex_matrix(self, matrix: list[list[Variable]], operator: TypeOrderedOperator):
        import numpy as np
        # lex_chain on rows
        self.ctr_lex(matrix, operator)
        # lex chain on columns
        self.ctr_lex(np.array(matrix).T.tolist(), operator)

    def ctr_precedence(self, lst: list[Variable], values: None | list[int], covered: bool):
        if covered is True:  # not supported for competition
            self._unimplemented(lst, values, covered)

        cpm_vars = self.get_cpm_vars(lst)
        if values is None:  # assumed to be ordered set of all values collected from domains in lst
            lbs, ubs = get_bounds(cpm_vars)
            values = set()
            for lb, ub in zip(lbs, ubs):
                values.update(list(range(lb, ub + 1)))
            values = sorted(values)

        self.cpm_model += cp.Precedence(cpm_vars, values)

    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable],
                condition: Condition):
        # cpm_vars = []
        # if isinstance(lst[0], XVar):
        #     for xvar in lst:
        #         cpm_vars.append(self.cpm_variables[xvar])
        # else:
        #     cpm_vars = self.exprs_from_node(lst)
        # arity, op = self.funcmap[condition.operator.name.lower()]
        # if hasattr(condition, "variable"):
        #     rhs = condition.variable
        # elif hasattr(condition, 'value'):
        #     rhs = condition.value
        # elif hasattr(condition, 'max'):
        #     #operator = in
        #     rhs = [x for x in range(condition.min, condition.max + 1)]
        # else:
        #     pass
        # cpm_rhs = self.get_cpm_var(rhs)
        # if coefficients is None:
        #     cpsum = cp.sum(cpm_vars)
        # else:
        #     cp_coeffs = self.get_cpm_vars(coefficients)
        #     cpsum = cp.sum(cp.cpm_array(cpm_vars) * cp_coeffs)
        # if arity == 0:
        #     self.cpm_model += op([cpsum, cpm_rhs])
        # else:
        #     self.cpm_model += op(cpsum, cpm_rhs)

        import numpy as np
        if coefficients is None or len(coefficients) == 0:
            coefficients = np.ones(len(lst), dtype=int)  # TODO I guess, if wsums are preferred over sums
        elif isinstance(coefficients[0], Variable):  # convert to cpmpy var
            coefficients = cp.cpm_array(self.get_cpm_vars(coefficients))
        else:
            coefficients = np.array(coefficients)

        lhs = cp.sum(coefficients * self.get_cpm_exprs(lst))

        if (condition.operator == TypeConditionOperator.IN) or (condition.operator == TypeConditionOperator.NOTIN):
            from pycsp3.classes.auxiliary.conditions import ConditionInterval, ConditionSet
            assert isinstance(condition,
                              (ConditionInterval, ConditionSet)), "Competition only supports intervals when operator is `in` or `notin`"  # TODO and not in? (notin)
            if isinstance(condition, ConditionInterval):
                rhs = list(range(condition.min, condition.max + 1))
            else: # ConditionSet
                rhs = list(condition.t)
        else:
            rhs = self.get_cpm_var(condition.right_operand())

        self.cpm_model += self.eval_cpm_comp(lhs, condition.operator, rhs)

    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        # General case of count, can accept list of variables for any arg and any operator
        cpm_vars = self.get_cpm_exprs(lst)
        cpm_vals = self.get_cpm_vars(values)
        if condition.operator == TypeConditionOperator.IN or (condition.operator == TypeConditionOperator.NOTIN):
            from pycsp3.classes.auxiliary.conditions import ConditionInterval, ConditionSet
            assert isinstance(condition,
                              (ConditionInterval, ConditionSet)), "Competition only supports intervals when operator is `in` or `notin`"  # TODO and not in? (notin)
            if isinstance(condition, ConditionInterval):
                rhs = list(range(condition.min, condition.max + 1))
            else: # ConditionSet
                rhs = list(condition.t)
        else:
            rhs = self.get_cpm_var(condition.right_operand())

        count_for_each_val = [cp.Count(cpm_vars, val) for val in cpm_vals]
        self.cpm_model += self.eval_cpm_comp(cp.sum(count_for_each_val), condition.operator, rhs)

    def ctr_atleast(self, lst: list[Variable] | list[Node], value: int, k: int):
        cpm_exprs = self.get_cpm_exprs(lst)
        self.cpm_model += (cp.Count(cpm_exprs, value) >= k)

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        cpm_vars = self.get_cpm_vars(lst)
        self.cpm_model += (cp.Count(cpm_vars, value) <= k)

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        cpm_vars = self.get_cpm_exprs(lst)
        self.cpm_model += (cp.Count(cpm_vars, value) == self.get_cpm_var(k))

    def ctr_among(self, lst: list[Variable], values: list[int], k: int | Variable):
        self.cpm_model += cp.Among(self.get_cpm_vars(lst), values) == self.get_cpm_var(k)

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        if excepting is None:
            lhs = cp.NValue(self.get_cpm_exprs(lst))
        else:
            assert len(excepting) == 1, "Competition only allows 1 integer value in excepting list"
            lhs = cp.NValueExcept(self.get_cpm_exprs(lst), excepting[0])
        
        if condition.operator == TypeConditionOperator.IN or (condition.operator == TypeConditionOperator.NOTIN):
            from pycsp3.classes.auxiliary.conditions import ConditionInterval, ConditionSet
            assert isinstance(condition,
                              (ConditionInterval, ConditionSet)), "Competition only supports intervals when operator is `in` or `notin`"  # TODO and not in? (notin)
            if isinstance(condition, ConditionInterval):
                rhs = list(range(condition.min, condition.max + 1))
            else: # ConditionSet
                rhs = list(condition.t)
        else:
            rhs = self.get_cpm_var(condition.right_operand())

        self.cpm_model += self.eval_cpm_comp(lhs, condition.operator, rhs)

    def ctr_not_all_qual(self, lst: list[Variable]):
        cpm_vars = self.get_cpm_vars(lst)
        self.cpm_model += cp.NValue(cpm_vars) > 1

    def ctr_cardinality(self, lst: list[Variable], values: list[int] | list[Variable],
                        occurs: list[int] | list[Variable] | list[range], closed: bool):
        self.cpm_model += cp.GlobalCardinalityCount(self.get_cpm_exprs(lst),
                                                    self.get_cpm_exprs(values),
                                                    self.get_cpm_exprs(occurs),
                                                    closed=closed)

    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        cpm_vars = self.get_cpm_exprs(lst)
        self.cpm_model += self.eval_cpm_comp(cp.Minimum(cpm_vars),
                                             condition.operator,
                                             self.get_cpm_var(condition.right_operand()))

    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        cpm_vars = self.get_cpm_exprs(lst)
        self.cpm_model += self.eval_cpm_comp(cp.Maximum(cpm_vars),
                                             condition.operator,
                                             self.get_cpm_var(condition.right_operand()))

    def ctr_minimum_arg(self, lst: list[Variable] | list[Node], condition: Condition,
                        rank: TypeRank):  # should enter XCSP3-core
        self._unimplemented(lst, condition, rank)

    def ctr_maximum_arg(self, lst: list[Variable] | list[Node], condition: Condition,
                        rank: TypeRank):  # should enter XCSP3-core
        self._unimplemented(lst, condition, rank)

    def ctr_element(self, lst: list[Variable] | list[int], i: Variable, condition: Condition):
        cpm_lst = self.get_cpm_vars(lst)
        cpm_index = self.get_cpm_var(i)
        cpm_rhs = self.get_cpm_var(condition.right_operand())
        self.cpm_model += self.eval_cpm_comp(cp.Element(cpm_lst, cpm_index), condition.operator, cpm_rhs)

    def ctr_element_matrix(self, matrix: list[list[Variable]] | list[list[int]], i: Variable, j: Variable,
                           condition: Condition):
        
        # this can be optimized by indexing into the matrix directly
        mtrx = cp.cpm_array([self.get_cpm_vars(lst) for lst in matrix])
        dim1, dim2 = mtrx.shape

        cpm_i, cpm_j = self.get_cpm_vars([i, j])
        cpm_rhs = self.get_cpm_var(condition.right_operand())
        # ensure i,j are within bounds, we can do this as it is a toplevel constraint
        self.cpm_model += [cpm_i >= 0, cpm_i < dim1, cpm_j >= 0, cpm_j < dim2]

        # flatten matrix and lookup with weighed sum
        self.cpm_model += self.eval_cpm_comp(cp.Element(flatlist(mtrx), dim1 * cpm_i + cpm_j), condition.operator, cpm_rhs)


    def ctr_channel(self, lst1: list[Variable], lst2: None | list[Variable]):

        if lst2 is None:
            self.cpm_model += xglobals.InverseOne(self.get_cpm_vars(lst1))
        else:
            cpm_vars1 = self.get_cpm_vars(lst1)
            cpm_vars2 = self.get_cpm_vars(lst2)
            # Ignace: deprecated, we have InverseAsym now which gets constructed automatically
            # # make lists same length, last part is irrelevant if not same length
            # if len(cpm_vars2) > len(cpm_vars1):
            #     cpm_vars2 = cpm_vars2[0:len(cpm_vars1)]
            # elif len(cpm_vars1) > len(cpm_vars2):
            #     cpm_vars1 = cpm_vars1[0:len(cpm_vars2)]
            self.cpm_model += xglobals.SafeOnlyInverse(cpm_vars1, cpm_vars2)

    def ctr_channel_value(self, lst: list[Variable], value: Variable):
        self.cpm_model += xglobals.Channel(self.get_cpm_vars(lst), self.get_cpm_var(value))

    def ctr_nooverlap(self, origins: list[Variable], lengths: list[int] | list[Variable],
                      zero_ignored: bool):  # in XCSP3 competitions, no 0 permitted in lengths
        cpm_start = self.get_cpm_vars(origins)
        cpm_dur = self.get_cpm_vars(lengths)
        cpm_end = [cp.intvar(*get_bounds(s + d)) for s, d in zip(cpm_start, cpm_dur)]
        self.cpm_model += cp.NoOverlap(cpm_start, cpm_dur, cpm_end)

    def ctr_nooverlap_multi(self, origins: list[list[Variable]], lengths: list[list[int]] | list[list[Variable]],
                            zero_ignored: bool):
        dim = len(origins[0])
        if dim == 2:

            start_x, start_y = self.get_cpm_vars([o[0] for o in origins]), self.get_cpm_vars([o[1] for o in origins])
            dur_x, dur_y = self.get_cpm_vars([l[0] for l in lengths]), self.get_cpm_vars([l[1] for l in lengths])

            end_x = [cp.intvar(*get_bounds(s + d)) for s, d in zip(start_x, dur_x)]
            end_y = [cp.intvar(*get_bounds(s + d)) for s, d in zip(start_y, dur_y)]

            self.cpm_model += xglobals.NoOverlap2d(start_x, dur_x, end_x,
                                             start_y, dur_y, end_y)

        else:  # n-dimensional, post decomposition directly
            from cpmpy.expressions.utils import all_pairs
            from cpmpy import any as cpm_any
            starts = cp.cpm_array([self.get_cpm_vars(lst) for lst in origins])
            durs = cp.cpm_array([self.get_cpm_vars(lst) for lst in lengths])

            for i, j in all_pairs(list(range(len(origins)))):
                self.cpm_model += cpm_any([(starts[i, d] + durs[i, d] <= starts[j, d]) | \
                                           (starts[j, d] + durs[j, d] <= starts[i, d]) for d in range(dim)])

    def ctr_nooverlap_mixed(self, xs: list[Variable], ys: list[Variable], lx: list[Variable], ly: list[int],
                            zero_ignored: bool):
        start_x = self.get_cpm_vars(xs)
        start_y = self.get_cpm_vars(ys)
        dur_x = self.get_cpm_vars(lx)
        dur_y = ly

        end_x = [cp.intvar(*get_bounds(s + d)) for s, d in zip(start_x, dur_x)]
        end_y = [cp.intvar(*get_bounds(s + d)) for s, d in zip(start_y, dur_y)]

        self.cpm_model += xglobals.NoOverlap2d(start_x, dur_x, end_x,
                                         start_y, dur_y, end_y)

    def ctr_cumulative(self, origins: list[Variable], lengths: list[int] | list[Variable],
                       heights: list[int] | list[Variable], condition: Condition):
        cpm_start = self.get_cpm_exprs(origins)
        cpm_durations = self.get_cpm_exprs(lengths)
        cpm_demands = self.get_cpm_exprs(heights)
        cpm_ends = []
        for s, d in zip(cpm_start, cpm_durations):
            expr = s + d
            cpm_ends.append(cp.intvar(*get_bounds(expr)))

        if condition.operator == TypeConditionOperator.LE:
            self.cpm_model += xglobals.DynamicCumulative(cpm_start, cpm_durations, cpm_ends, cpm_demands,
                                            self.get_cpm_var(condition.right_operand()))
        else:
            # post decomposition directly
            # be smart and chose task or time decomposition #TODO you did task decomp in both cases
            if max(get_bounds(cpm_ends)[1]) >= 100:
                self._cumulative_task_decomp(cpm_start, cpm_durations, cpm_ends, heights, condition)
            else:
                self._cumulative_time_decomp(cpm_start, cpm_durations, cpm_ends, heights, condition)

    def _cumulative_task_decomp(self, cpm_start, cpm_duration, cpm_ends, cpm_demands, condition: Condition):
        cpm_demands = cp.cpm_array(cpm_demands)
        cpm_cap = self.get_cpm_var(condition.right_operand())
        # ensure durations are satisfied
        for s, d, e in zip(cpm_start, cpm_duration, cpm_ends):
            self.cpm_model += s + d == e

        # task decomposition
        for s, d, e in zip(cpm_start, cpm_duration, cpm_ends):
            # find overlapping tasks
            total_running = cp.sum(cpm_demands * ((cpm_start <= s) & (cpm_ends > s)))
            self.cpm_model += self.eval_cpm_comp(total_running, condition.operator, cpm_cap)

    def _cumulative_time_decomp(self, cpm_start, cpm_duration, cpm_ends, cpm_demands, condition: Condition):
        cpm_demands = cp.cpm_array(cpm_demands)
        cpm_cap = self.get_cpm_var(condition.right_operand)
        # ensure durations are satisfied
        for s, d, e in zip(cpm_start, cpm_duration, cpm_ends):
            self.cpm_model += s + d == e

        lb = min(get_bounds(cpm_start)[0])
        ub = max(get_bounds(cpm_ends)[1])
        # time decomposition
        for t in range(lb, ub + 1):
            total_running = cp.sum(cpm_demands * ((cpm_start <= t) & (cpm_ends > t)))
            self.cpm_model += self.eval_cpm_comp(total_running, condition.operator, cpm_cap)

    def ctr_binpacking(self, lst: list[Variable], sizes: list[int], condition: Condition):
        cpm_vars = self.get_cpm_vars(lst)
        cpm_rhs = self.get_cpm_var(condition.right_operand())

        for bin in range(0, len(cpm_vars)):  # bin labeling starts at 0, contradicting the xcsp3 specification document?
            self.cpm_model += self.eval_cpm_comp(cp.sum((cpm_array(cpm_vars) == bin) * sizes),
                                                 condition.operator,
                                                 cpm_rhs)

    def ctr_binpacking_limits(self, lst: list[Variable], sizes: list[int], limits: list[int] | list[Variable]):
        from cpmpy.expressions.utils import eval_comparison

        cpm_vars = self.get_cpm_vars(lst)

        for bin, lim in enumerate(limits):
            self.cpm_model += eval_comparison("<=",
                                              cp.sum((cpm_array(cpm_vars) == (bin)) * sizes),
                                              lim)

    def ctr_binpacking_loads(self, lst: list[Variable], sizes: list[int], loads: list[int] | list[Variable]):
        from cpmpy.expressions.utils import eval_comparison

        cpm_vars = self.get_cpm_vars(lst)
        cpm_loads = self.get_cpm_vars(loads)

        for bin, load in enumerate(cpm_loads):
            self.cpm_model += eval_comparison("==",
                                              cp.sum((cpm_array(cpm_vars) == (bin)) * sizes),
                                              load)

    def ctr_binpacking_conditions(self, lst: list[Variable], sizes: list[int],
                                  conditions: list[Condition]):  # not in XCSP3-core
        self._unimplemented(lst, sizes, conditions)

    def ctr_knapsack(self, lst: list[Variable], weights: list[int], wcondition: Condition, profits: list[int],
                     pcondition: Condition):

        vars = cpm_array(self.get_cpm_vars(lst))
        cpm_weight = self.get_cpm_var(wcondition.right_operand())
        cpm_profit = self.get_cpm_var(pcondition.right_operand())

        total_weight = cp.sum(vars * weights)
        total_profit = cp.sum(vars * profits)
        self.cpm_model += self.eval_cpm_comp(total_weight, wcondition.operator, cpm_weight)
        self.cpm_model += self.eval_cpm_comp(total_profit, pcondition.operator, cpm_profit)

    def ctr_flow(self, lst: list[Variable], balance: list[int] | list[Variable], arcs: list,
                 capacities: None | list[range]):  # not in XCSP3-core
        self._unimplemented(lst, balance, arcs, capacities)

    def ctr_flow_weighted(self, lst: list[Variable], balance: list[int] | list[Variable], arcs: list,
                          capacities: None | list[range],
                          weights: list[int] | list[Variable],
                          condition: Condition):  # not in XCSP3-core
        self._unimplemented(lst, balance, arcs, capacities, weights, condition)

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        self.cpm_model += xglobals.NonReifiedTable(self.get_cpm_vars(lst), [values])

    def ctr_clause(self, pos: list[Variable], neg: list[Variable]):  # not in XCSP3-core
        self._unimplemented(pos, neg)

    def ctr_circuit(self, lst: list[Variable], size: None | int | Variable):  # size is None in XCSP3 competitions
        self.cpm_model += xglobals.SubCircuitWithStart(self.get_cpm_vars(lst), start_index=0)

    # # # # # # # # # #
    # All methods about objectives to be implemented
    # # # # # # # # # #

    def obj_minimize(self, term: Variable | Node):
        if isinstance(term, Node):
            cpm_expr = self.exprs_from_node([term])
            assert len(cpm_expr) == 1
            cpm_expr = cpm_expr[0]
        else:
            cpm_expr = self.get_cpm_var(term)
        self.cpm_model.minimize(cpm_expr)

    def obj_maximize(self, term: Variable | Node):
        if isinstance(term, Node):
            cpm_expr = self.exprs_from_node([term])
            assert len(cpm_expr) == 1
            cpm_expr = cpm_expr[0]
        else:
            cpm_expr = self.get_cpm_var(term)
        self.cpm_model.maximize(cpm_expr)

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node],
                             coefficients: None | list[int]):
        import numpy as np
        if coefficients is None:
            coefficients = np.ones(len(terms))
        else:
            coefficients = np.array(coefficients)

        if obj_type == TypeObj.SUM:
            self.cpm_model.minimize(cp.sum(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.MAXIMUM:
            self.cpm_model.minimize(cp.max(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.MINIMUM:
            self.cpm_model.minimize(cp.min(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.NVALUES:
            self.cpm_model.minimize(cp.NValue(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.EXPRESSION:
            assert all(coeff == 1 for coeff in coefficients)
            assert len(terms) == 1
            cpm_expr = self.get_cpm_exprs(terms)[0]
            self.cpm_model.minimize(cpm_expr)
        else:
            self._unimplemented(obj_type, coefficients, terms)

    # TODO objectives are a bit confusing but i think it's correct
    def obj_maximize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node],
                             coefficients: None | list[int]):
        import numpy as np
        if coefficients is None:
            coefficients = np.ones(len(terms))
        else:
            coefficients = np.array(coefficients)

        if obj_type == TypeObj.SUM:
            self.cpm_model.maximize(cp.sum(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.MAXIMUM:
            self.cpm_model.maximize(cp.max(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.MINIMUM:
            self.cpm_model.maximize(cp.min(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.NVALUES:
            self.cpm_model.maximize(cp.NValue(coefficients * self.get_cpm_exprs(terms)))
        elif obj_type == TypeObj.EXPRESSION:
            assert all(coeff == 1 for coeff in coefficients)
            assert len(terms) == 1
            cpm_expr = self.get_cpm_exprs(terms)[0]
            self.cpm_model.maximize(cpm_expr)
        else:
            self._unimplemented(obj_type, coefficients, terms)

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
            return x  # constants

    def get_cpm_vars(self, lst):
        if isinstance(lst[0], (XVar, int)):
            return [self.get_cpm_var(x) for x in lst]
        if isinstance(lst[0], range):
            assert len(lst) == 1, "Expected range here, but got list with multiple elements, what's the semantics???"
            return list(lst[0])  # this should work without converting to str first
            # return list(eval(str(lst[0])))
        else:
            return self.vars_from_node(lst)

    def get_cpm_exprs(self, lst):
        if isinstance(lst[0], XVar):
            return [self.get_cpm_var(x) for x in lst]
        if isinstance(lst[0], range):
            # assert len(lst) == 1, f"Expected range here, but got list with multiple elements, what's the semantics???{lst}"

            if len(lst) == 1:
                return list(lst[0])  # this should work without converting to str first
            else:
                return [cp.intvar(l.start, l.stop - 1) for l in lst]

            # return list(eval(str(lst[0])))
        else:
            return self.exprs_from_node(lst)

    def end_instance(self):
        pass

    def load_annotation(self, annotation):
        pass

    def load_annotations(self, annotations):
        pass

    def load_objectives(self, objectives):
        pass

    def ann_decision(self, lst: list[Variable]):
        pass

    def ann_val_heuristic_static(self, lst: list[Variable], order: list[int]):
        pass
