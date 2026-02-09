import os
import tempfile

import pytest
import cpmpy as cp
from cpmpy.tools.dimacs import read_dimacs, write_dimacs, write_gdimacs, read_gdimacs
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.solvers.pindakaas import CPM_pindakaas
from test_tocnf import get_cnf_cases, get_gcnf_cases


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class TestCNFTool:
    def setup_method(self) -> None:
        self.tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)

    def teardown_method(self) -> None:
        self.tmpfile.close()
        os.remove(self.tmpfile.name)

    def dimacs_to_model(self, cnf_str):
        typ = "gcnf" if "gcnf" in cnf_str else "cnf"
        with open(self.tmpfile.name, "w") as f:
            f.write(cnf_str)
        return read_dimacs(self.tmpfile.name) if typ == "cnf" else read_gdimacs(self.tmpfile.name)

    def test_read_cnf(self):
        model = self.dimacs_to_model("p cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n")
        bvs = sorted(get_variables_model(model), key=str)
        assert str(model) == str(
            cp.Model(cp.any([~bvs[1], ~bvs[2]]), cp.any([bvs[2], bvs[1], bvs[0]]), ~bvs[0])
        )

    def test_empty_formula(self):
        model = self.dimacs_to_model("p cnf 0 0")
        assert model.solve()
        assert model.status().exitstatus == ExitStatus.FEASIBLE

    def test_empty_clauses(self):
        model = self.dimacs_to_model("p cnf 0 2\n0\n0")
        assert not model.solve()
        assert model.status().exitstatus == ExitStatus.UNSATISFIABLE

    def test_with_comments(self):
        model = self.dimacs_to_model(
            "c this file starts with some comments\nc\np cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n"
        )
        vars = sorted(get_variables_model(model), key=str)

        sols = set()
        addsol = lambda: sols.add(tuple([v.value() for v in vars]))

        assert model.solveAll(display=addsol) == 2
        assert sols == {(False, False, True), (False, True, False)}

    def test_write_cnf(self):

        a, b, c = [cp.boolvar(name=n) for n in "abc"]

        m = cp.Model()
        m += cp.any([a, b, c])
        m += b.implies(~c)
        m += a <= 0

        assert write_dimacs(model=m, canonical=True) == "p cnf 3 3\n1 2 3 0\n-2 -3 0\n-1 0\n"

    def test_missing_p_line(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("1 -2 0\np cnf 2 2")

    def test_incorrect_p_line(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0")

    def test_too_many_clauses(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0\n1 0\n2 0")

    def test_too_few_clauses(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 0")

    def test_too_many_variables(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("p cnf 2 1\n1 2 3 0")

    def test_non_int_literal(self):
        with pytest.raises(ValueError):
            self.dimacs_to_model("p cnf 2 1\n1 b 2 0")

    def test_non_terminated_final_clause(self):
        with pytest.raises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0\n-1 -2 0\n2")

    def test_too_few_variables(self):
        """ "Fewer variables is still technically correct DIMACS"""
        self.dimacs_to_model("p cnf 2 1\n1 0")

    def test_gdimacs_roundtrip(self):
        # example from https://satisfiability.org/competition/2011/rules.pdf, but fixed p header vars to 3
        example = """p gcnf 3 7 4
{0} 1 2 3 0
{1} -1 2 0
{1} -2 3 0
{2} -3 0
{3} 2 -3 0
{3} -2 -3 0
{4} -2 3 0
"""
        m, soft, hard, assumptions = self.dimacs_to_model(example)
        out = write_gdimacs(soft, hard=hard, disjoint=False, canonical=True)
        assert example == out

    def test_gcnf(self):
        x = cp.boolvar(shape=3)

        def x_(i):
            return x[i - 1]

        # example from https://satisfiability.org/competition/2011/rules.pdf
        # c
        # c Example of group oriented CNF
        # c
        # c Represents the following formula
        # c D
        # = {x1 or x2 or x3}
        # c G1 = {x1 -> x2, x2 -> x3}
        # c G2 = {x3}
        # c G3 = {x3 -> x2, -x2 or -x3}
        # c G4 = {x2 -> x3}
        # c
        hard = [cp.any(x)]
        soft = [
            x_(1).implies(x_(2)) & x_(2).implies(x_(3)),
            x_(3),
            x_(3).implies(x_(2)) & (~x_(2)) | (~x_(3)),
            x_(2).implies(x_(3)),
        ]

        # compare with (canonicalized) example
        assert (
            write_gdimacs(soft, hard=hard, encoding="direct", canonical=True, disjoint=False)
            == """p gcnf 5 12 4
{0} 1 2 3 0
{0} -4 5 0
{0} -2 -4 0
{0} 2 4 -5 0
{0} 2 -3 -5 0
{0} 3 5 0
{0} -2 5 0
{1} -1 2 0
{1} -2 3 0
{2} 3 0
{3} -3 4 0
{4} -2 3 0
"""
        )

        # note: 2nd clause of group 4 is merged with 2nd clause of group 1
        assert """p gcnf 6 13 4
{0} 1 2 3 0
{0} -4 5 0
{0} -2 -4 0
{0} 2 4 -5 0
{0} 2 -3 -5 0
{0} 3 5 0
{0} -2 5 0
{0} -2 3 -6 0
{1} -1 2 0
{1} -2 3 0
{2} 3 0
{3} -3 4 0
{4} 6 0
""" == write_gdimacs(soft, hard=hard, name="a", encoding="direct", disjoint=True, canonical=True)

    @pytest.mark.parametrize(
        "gcnf_str,expected_soft,expected_hard,expected_assumptions,should_solve,extra_check",
        [
            # Basic GCNF with hard and soft constraints
            (
                """p gcnf 3 5 2
{0} 1 2 0
{0} -2 3 0
{1} 1 -3 0
{2} 2 3 0
{2} -1 0
""",
                2,
                1,
                2,
                True,
                None,
            ),
            # Non-consecutive group numbers (missing group 2)
            (
                """p gcnf 3 5 3
{0} 1 2 0
{1} 1 -2 0
{1} -3 0
{3} 2 3 0
{3} -1 -2 0
""",
                2,
                1,
                2,
                True,
                None,
            ),
            # Only hard constraints (group 0)
            (
                """p gcnf 2 3 0
{0} 1 2 0
{0} -1 0
{0} -2 0
""",
                0,
                1,
                0,
                False,
                None,
            ),
            # Single soft constraint group
            (
                """p gcnf 2 2 1
{0} 1 0
{1} -1 -2 0
""",
                1,
                1,
                1,
                True,
                None,
            ),
            # Empty clause in soft group - model solvable but assumption False
            (
                """p gcnf 1 2 1
{0} 1 0
{1} 0
""",
                1,
                1,
                1,
                True,
                lambda model, soft, hard, assumptions: assumptions[0].value() == False,
            ),
            # With comments
            (
                """c This is a comment
c Another comment line
p gcnf 2 3 1
c Comment in the middle
{0} 1 2 0
{1} -1 0
{1} -2 0
""",
                1,
                1,
                1,
                True,
                None,
            ),
        ],
        ids=[
            "basic",
            "missing_group",
            "only_hard",
            "single_group",
            "empty_clause",
            "with_comments",
        ],
    )
    def test_read_gcnf_parametrized(
        self, gcnf_str, expected_soft, expected_hard, expected_assumptions, should_solve, extra_check
    ):
        """Parametrized test for reading various GCNF files"""
        model, soft, hard, assumptions = self.dimacs_to_model(gcnf_str)

        assert len(soft) == expected_soft
        assert len(hard) == expected_hard
        assert len(assumptions) == expected_assumptions

        if should_solve:
            assert model.solve()
        else:
            assert not model.solve()

        if extra_check is not None:
            assert extra_check(model, soft, hard, assumptions)

        print(model)
        back = write_gdimacs(soft, hard=hard, canonical=True, disjoint=False, name="a")
        print(back)
        gcnf_str = "\n".join(l for l in gcnf_str.split("\n") if not l.startswith("c"))
        assert back == gcnf_str

    def test_read_gcnf_negative_group_number(self):
        """Test that negative group numbers raise an error"""
        gcnf_str = """p gcnf 2 3 2
{0} 1 2 0
{1} 1 -2 0
{-1} -1 -2 0
"""
        with pytest.raises(AssertionError, match="Group number must be non-negative"):
            self.dimacs_to_model(gcnf_str)
