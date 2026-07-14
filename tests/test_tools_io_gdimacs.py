"""
GDIMACS (Grouped DIMACS / GCNF) specific tests for ``cpmpy.tools.io.gdimacs``.

Covers the GCNF parser, the ``write_gdimacs`` writer and round-trips between them.
"""

import pytest
import cpmpy as cp
from cpmpy.tools.io.gdimacs import load_gdimacs, write_gdimacs
from cpmpy.solvers.pindakaas import CPM_pindakaas


@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class TestGDimacsTool:

    def test_gcnf_mus2011_example(self):
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
        m, soft, hard, assumptions = load_gdimacs(example, var_name="x")
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
{0} 2 -3 -5 0
{0} 3 5 0
{0} -2 5 0
{0} -4 5 0
{0} -2 -4 0
{0} 2 4 -5 0
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
{0} 2 -3 -5 0
{0} 3 5 0
{0} -2 5 0
{0} -4 5 0
{0} -2 -4 0
{0} 2 4 -5 0
{0} -2 3 -6 0
{1} -1 2 0
{1} -2 3 0
{2} 3 0
{3} -3 4 0
{4} 6 0
""" == write_gdimacs(soft, hard=hard, encoding="direct", disjoint=True, canonical=True)

    @pytest.mark.parametrize(
        "gcnf_str",
        [
            # Basic GCNF with hard and soft constraints
            """p gcnf 3 5 2
{0} 1 2 0
{0} -2 3 0
{1} 1 -3 0
{2} 2 3 0
{2} -1 0
""",
            # Non-consecutive group numbers (missing group 2)
            """p gcnf 3 5 3
{0} 1 2 0
{1} 1 -2 0
{1} -3 0
{3} 2 3 0
{3} -1 -2 0
""",
            # Only hard constraints (group 0)
            """p gcnf 2 3 0
{0} 1 2 0
{0} -1 0
{0} -2 0
""",
            # Single soft constraint group
            """p gcnf 2 2 1
{0} 1 0
{1} -1 -2 0
""",
            # Empty clause in soft group - model solvable but assumption False
            (
                """p gcnf 1 2 1
{0} 1 0
{1} 0
"""
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
"""
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
    def test_gdimacs_roundtrip(self, gcnf_str, request):
        """Parametrized test for reading various GCNF files"""
        model, soft, hard, assumptions = load_gdimacs(gcnf_str, var_name="x", assumption_name="a")

        back = write_gdimacs(
            soft,
            hard=hard,
            canonical=True,
            disjoint=False,
        )
        gcnf_str = "\n".join(l for l in gcnf_str.split("\n") if not l.startswith("c"))
        if request.node.callspec.id != "missing_group":
            assert back == gcnf_str, f"Roundtrip failed from:\n\n{gcnf_str}\nto\n\n{back}"

    def test_read_gcnf_negative_group_number(self):
        """Test that negative group numbers raise an error"""
        gcnf_str = """p gcnf 2 3 2
{0} 1 2 0
{1} 1 -2 0
{-1} -1 -2 0
"""
        with pytest.raises(AssertionError, match="Group number must be non-negative"):
            load_gdimacs(gcnf_str)
