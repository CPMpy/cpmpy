import cpmpy as cp
from cpmpy.transformations.safening import no_partial_functions
from cpmpy.expressions.utils import argval


class TestTransSafen:

    def test_division_by_zero(self):
        a = cp.intvar(1, 10, name="a")
        b = cp.intvar(0, 10, name="b")
        expr = (a // b) == 3

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert cp.Model(safe_expr).solve()
        assert argval(safe_expr)

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        assert solcount == 110

        # check with reification
        bv = cp.boolvar(name="bv")
        reif_expr = bv == expr
        def check():
            assert reif_expr.value()
        solcount = cp.Model(no_partial_functions([reif_expr])).solveAll(display=check)
        assert solcount == 110

    def test_division_by_zero_proper_hole(self):
        a = cp.intvar(1, 10, name="a")
        b = cp.intvar(-1, 10, name="b")
        expr = (a // b) <= 3

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert cp.Model(safe_expr).solve()
        assert argval(safe_expr)

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        assert solcount == 120

        bv = cp.boolvar(name="bv")
        reif_expr = bv == expr
        def check():
            assert reif_expr.value()
        solcount = cp.Model(no_partial_functions([reif_expr])).solveAll(display=check)
        assert solcount == 120

    def test_division_by_constant_zero(self):
        a = cp.intvar(1, 10, name="a")
        b = cp.intvar(0, 0, name="b")
        expr = (a // b) <= 3

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert not cp.Model(safe_expr).solve()

        safened = no_partial_functions([~expr])
        assert str(safened[0]) == "not(boolval(False))"

    def test_element_out_of_bounds(self):
        arr = cp.intvar(1,3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")
        expr = arr[idx] == 2

        safe_expr = no_partial_functions([expr])
        assert cp.Model(safe_expr).solve()
        assert argval(safe_expr)

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        assert solcount == 162

        bv = cp.boolvar(name="bv")
        reif_expr = bv == expr
        def check():
            assert reif_expr.value()
        solcount = cp.Model(no_partial_functions([reif_expr])).solveAll(display=check)
        assert solcount == 162
    
    def test_multiple_partial_functions(self):
        a = cp.intvar(1, 5)
        b = cp.intvar(0, 2)
        arr = cp.intvar(1, 3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")

        expr = (a // b + arr[idx]) == 2

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert cp.Model(safe_expr).solve()
        assert argval(safe_expr)

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        assert solcount == 15*162

        bv = cp.boolvar(name="bv")
        reif_expr = bv == expr
        def check():
            assert reif_expr.value()
        solcount = cp.Model(no_partial_functions([reif_expr])).solveAll(display=check)
        assert solcount == 15*162

    def test_nested_partial_functions(self):
        a = cp.intvar(1, 10)
        arr = cp.intvar(0,3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")

        expr = (a // arr[idx]) == 2

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert cp.Model(safe_expr).solve()
        assert argval(safe_expr)

        safened = no_partial_functions([expr | ~expr])
        solcount = cp.Model(safened).solveAll()
        assert solcount == 10*(4**3)*6

        bv = cp.boolvar(name="bv")
        reif_expr = bv == expr
        def check():
            assert reif_expr.value()
        solcount = cp.Model(no_partial_functions([reif_expr])).solveAll(display=check)
        assert solcount == 10*(4**3)*6

    def test_nested_partial_functions2(self):
        a = cp.intvar(1, 10)
        arr = cp.intvar(0,3, shape=3, name="x")
        idx = cp.intvar(-1, 4, name="i")

        expr = ~((a * arr[idx]) == 0)

        safe_expr = no_partial_functions([expr], safen_toplevel={"div"})
        assert cp.Model([safe_expr, idx == 4]).solve()

    def test_partial_under_numerical_is_not_nested(self):
        """Partial functions under sum/min/max still have the constraint root as nearest bool parent."""
        a = cp.intvar(1, 5)
        b = cp.intvar(0, 2)
        c = cp.intvar(1, 3)
        arr = cp.intvar(1, 3, shape=3)
        idx = cp.intvar(-1, 4)

        for expr in [
            cp.sum([a // b, c]) == 4,
            cp.min([a // b, c]) == 2,
            cp.sum([arr[idx], c]) == 4,
            (a // b) == 2,
            arr[idx] == 2,
        ]:
            assert str(no_partial_functions([expr])) == str([expr])

    def test_partial_under_nested_bool_is_safened(self):
        a = cp.intvar(1, 5)
        b = cp.intvar(0, 2)
        bv = cp.boolvar()
        orig = bv == ((a // b) == 2)

        safe = no_partial_functions([orig])
        assert safe != [orig]
        assert "and(" in str(safe[0])
        assert cp.Model(safe).solve()
