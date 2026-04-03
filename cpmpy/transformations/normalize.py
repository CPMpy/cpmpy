from ..expressions.core import Expression, Operator, Comparison, BoolVal
from ..expressions.variables import NDVarArray, _BoolVarImpl, _IntVarImpl, _FloatVarImpl
from ..expressions.utils import is_num, is_any_list
from .visitor import Visitor


def def toplevel_list(cpm_expr, merge_and=True):
    """
    unravels nested lists and top-level 'and' constraints
    """
    # normalise all expressions to lists
    if not is_any_list(cpm_expr):
        cpm_expr = [cpm_expr]

    new_expr = []
    for expr in cpm_expr:
        if is_any_list(expr):
            new_expr.extend(toplevel_list(expr))
        elif merge_and and isinstance(expr, Operator) and expr.name == 'and':
            new_expr.extend(toplevel_list(expr.args))
        else:
            new_expr.append(expr)
    return new_expr


def simplify_boolean(constraints):
    """
    Specialized function for simplifying boolean expressions.
    - Replaces boolval with 0/1 in comparisons.
    - Simplifies boolean operators with constants.
    - Simplifies comparisons with constants.
    - Chains of comparisons are not handled here.
    """
    return SimplifyVisitor().visit(constraints)

class SimplifyVisitor(Visitor):
    """
    Visitor that simplifies boolean expressions.
    - Replaces boolval with 0/1 in comparisons.
    - Simplifies boolean operators with constants.
    - Simplifies comparisons with constants.
    - Chains of comparisons are not handled here.
    """
    def visit_boolval(self, expr, *args, **kwargs):
        return expr

    def visit_param(self, expr, *args, **kwargs):
        return expr

    def visit_expression(self, expr, *args, **kwargs):
        # recurse
        expr.args = [self.visit(arg) for arg in expr.args]
        return self.visit_operator(expr) if isinstance(expr, Operator) \
            else self.visit_comparison(expr) if isinstance(expr, Comparison) \
            else expr

    def visit_operator(self, expr):
        if expr.name == 'not':
            # ~(~a) -> a
            if isinstance(expr.args[0], Operator) and expr.args[0].name == 'not':
                return expr.args[0].args[0]
            # not(BoolVal)
            if isinstance(expr.args[0], BoolVal):
                return BoolVal(not expr.args[0].value)
            return expr

        # and, or, xor, ->
        if expr.name == 'and':
            new_args = []
            for arg in expr.args:
                if isinstance(arg, BoolVal):
                    if arg.value is False:
                        return BoolVal(False)
                elif isinstance(arg, Operator) and arg.name == 'and':
                    new_args.extend(arg.args)
                else:
                    new_args.append(arg)
            if len(new_args) == 0:
                return BoolVal(True)
            if len(new_args) == 1:
                return new_args[0]
            expr.args = new_args
            return expr
        elif expr.name == 'or':
            new_args = []
            for arg in expr.args:
                if isinstance(arg, BoolVal):
                    if arg.value is True:
                        return BoolVal(True)
                elif isinstance(arg, Operator) and arg.name == 'or':
                    new_args.extend(arg.args)
                else:
                    new_args.append(arg)
            if len(new_args) == 0:
                return BoolVal(False)
            if len(new_args) == 1:
                return new_args[0]
            expr.args = new_args
            return expr
        elif expr.name == '->':
            # a -> b is ~a | b
            # if a is true, -> is b
            if isinstance(expr.args[0], BoolVal) and expr.args[0].value is True:
                return expr.args[1]
            # if a is false, -> is true
            if isinstance(expr.args[0], BoolVal) and expr.args[0].value is False:
                return BoolVal(True)
            # if b is true, -> is true
            if isinstance(expr.args[1], BoolVal) and expr.args[1].value is True:
                return BoolVal(True)
            # if b is false, -> is ~a
            if isinstance(expr.args[1], BoolVal) and expr.args[1].value is False:
                return ~expr.args[0]
        elif expr.name == 'xor':
            true_count = 0
            new_args = []
            for arg in expr.args:
                if isinstance(arg, BoolVal):
                    if arg.value is True:
                        true_count += 1
                else:
                    new_args.append(arg)

            if len(new_args) == 0:
                return BoolVal(true_count % 2 == 1)

            if len(new_args) == 1:
                res = new_args[0]
            else:
                # rebuild expression
                expr.args = new_args
                res = expr

            if true_count % 2 == 1:
                return ~res
            return res

        return expr

    def visit_comparison(self, expr):
        # replace any boolval with their integer equivalent
        if isinstance(expr.args[0], BoolVal):
            expr.args[0] = int(expr.args[0].value)
        if isinstance(expr.args[1], BoolVal):
            expr.args[1] = int(expr.args[1].value)

        lhs, rhs = expr.args
        # if both are numbers, we can evaluate it
        if is_num(lhs) and is_num(rhs):
            return BoolVal(expr.eval())

        # if one is a boolvar...
        if isinstance(lhs, _BoolVarImpl):
            return self._simplify_boolvar_comp(expr.name, lhs, rhs)
        if isinstance(rhs, _BoolVarImpl):
            # swap arguments and comparison operator
            rev_op = {'<':' >', '>':' <', '<=':' >=', '>=':' <=', '==':'==', '!=':'!='}
            return self._simplify_boolvar_comp(rev_op[expr.name], rhs, lhs)

        # if one is a (boolean) expression...
        if isinstance(lhs, Expression) and lhs.is_bool():
            return self._simplify_boolexpr_comp(expr.name, lhs, rhs)
        if isinstance(rhs, Expression) and rhs.is_bool():
            rev_op = {'<':' >', '>':' <', '<=':' >=', '>=':' <=', '==':'==', '!=':'!='}
            return self._simplify_boolexpr_comp(rev_op[expr.name], rhs, lhs)

        return expr

    def _simplify_boolvar_comp(self, op, bv, num):
        # bv op num
        if op == '==':
            if num < 0 or num > 1: return BoolVal(False)
            if num == 0: return ~bv
            if num == 1: return bv
        if op == '!=':
            if num < 0 or num > 1: return BoolVal(True)
            if num == 0: return bv
            if num == 1: return ~bv
        if op == '>':
            if num >= 1: return BoolVal(False)
            if num < 0: return BoolVal(True)
            return bv # >0, >0.5
        if op == '<':
            if num <= 0: return BoolVal(False)
            if num > 1: return BoolVal(True)
            return ~bv # <1, <0.5
        if op == '>=':
            if num > 1: return BoolVal(False)
            if num <= 0: return BoolVal(True)
            return bv # >=1, >=0.5
        if op == '<=':
            if num < 0: return BoolVal(False)
            if num >= 1: return BoolVal(True)
            return ~bv # <=0, <=0.5
        # unknown operator
        return Comparison(op, bv, num)

    def _simplify_boolexpr_comp(self, op, expr, num):
        # expr op num
        if op == '==':
            if num == 0: return ~expr
            if num == 1: return expr
        if op == '!=':
            if num == 0: return expr
            if num == 1: return ~expr
        if op == '>':
            if num >= 1: return BoolVal(False)
            if num < 0: return BoolVal(True)
            return expr # >0, >0.5
        if op == '<':
            if num <= 0: return BoolVal(False)
            if num > 1: return BoolVal(True)
            return ~expr # <1, <0.5
        if op == '>=':
            if num > 1: return BoolVal(False)
            if num <= 0: return BoolVal(True)
            return expr # >=1, >=0.5
        if op == '<=':
            if num < 0: return BoolVal(False)
            if num >= 1: return BoolVal(True)
            return ~expr # <=0, <=0.5
        # unhandled, return as is
        return Comparison(op, expr, num)
