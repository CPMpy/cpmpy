"""
GDIMACS (Grouped DIMACS / GCNF) specific tests for ``cpmpy.tools.io.gdimacs``.

Generic load/write/round-trip coverage for other formats lives in ``test_tools_io.py``.
These tests focus on GCNF parser edge cases, p-line validation, writer output shape,
and loader/writer round-trips.
"""

import pytest
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.transformations.get_variables import get_variables_model

from cpmpy.tools.io.gdimacs import load_gdimacs, write_gdimacs


BASIC_GCNF = """p gcnf 3 4 2
{0} 1 2 3 0
{1} -1 0
{1} -2 0
{2} -3 0
"""


def _assert_sat(model):
    assert model.solve()
    assert model.status().exitstatus == ExitStatus.FEASIBLE


def _assert_unsat(model):
    assert not model.solve()
    assert model.status().exitstatus == ExitStatus.UNSATISFIABLE


# --------------------------------------------------------------------------- #
#                             GCNF parsing (load)                             #
# --------------------------------------------------------------------------- #

class TestLoadGCNF:

    def test_basic(self):
        model, soft, hard, assumptions = load_gdimacs(BASIC_GCNF)

        assert len(soft) == 2
        assert len(hard) == 1
        assert len(assumptions) == 2
        _assert_sat(model)

    def test_group_semantics(self):
        """Hard clause (x1|x2|x3), group 1 (~x1 & ~x2), group 2 (~x3): together UNSAT, each alone SAT"""
        model, soft, hard, assumptions = load_gdimacs(BASIC_GCNF)

        _assert_unsat(cp.Model(model.constraints + list(assumptions)))
        _assert_sat(cp.Model(model.constraints + [assumptions[0]]))
        _assert_sat(cp.Model(model.constraints + [assumptions[1]]))

    def test_var_and_assumption_names(self):
        model, soft, hard, assumptions = load_gdimacs(BASIC_GCNF, var_name="x", assumption_name="a")

        names = {str(v) for v in get_variables_model(model)}
        assert {"x[0]", "x[1]", "x[2]"} <= names
        assert [str(a) for a in assumptions] == ["a[0]", "a[1]"]

    def test_with_comments(self):
        model, soft, hard, assumptions = load_gdimacs(
            "c starting comment\nc\n\n"
            "p gcnf 2 3 1\n"
            "{0} 1 2 0\n"
            "c mid comment\n"
            "{1} -1 0\n"
            "{1} -2 0\n"
        )

        assert len(soft) == 1
        assert len(hard) == 1

    def test_only_hard(self):
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 2 3 0\n{0} 1 2 0\n{0} -1 0\n{0} -2 0\n"
        )

        assert soft == []
        assert len(assumptions) == 0
        _assert_unsat(model)

    def test_only_soft(self):
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 2 2 2\n{1} 1 0\n{2} -1 -2 0\n"
        )

        assert hard == []
        assert len(soft) == 2
        _assert_sat(cp.Model(model.constraints + list(assumptions)))

    def test_multiple_clauses_per_group(self):
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 3 3 1\n{0} 1 2 3 0\n{1} -1 0\n{1} -2 0\n"
        )

        assert len(soft) == 1  # both clauses of group 1 form one soft constraint
        assert len(assumptions) == 1

    def test_multiple_clauses_one_line(self):
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 2 2 1\n{1} 1 0 -2 0\n"
        )

        assert len(soft) == 1
        assert hard == []

    def test_interleaved_hard_group(self):
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 2 3 1\n{0} 1 0\n{1} -1 -2 0\n{0} 2 0\n"
        )

        assert len(soft) == 1
        # both group-0 runs are enforced as hard constraints
        assert len(hard) == 2
        _assert_sat(model)

    def test_non_consecutive_group_ids(self):
        """Group ids need not be consecutive; each id becomes one soft constraint"""
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 3 5 3\n{0} 1 2 0\n{1} 1 -2 0\n{1} -3 0\n{3} 2 3 0\n{3} -1 -2 0\n"
        )

        assert len(soft) == 2  # groups 1 and 3 (group 2 unused)
        assert len(hard) == 1

    def test_empty_clause_in_soft_group(self):
        """An empty soft clause keeps the model satisfiable, but its assumption cannot hold"""
        model, soft, hard, assumptions = load_gdimacs("p gcnf 1 2 1\n{0} 1 0\n{1} 0\n")

        _assert_sat(model)
        _assert_unsat(cp.Model(model.constraints + [assumptions[0]]))

    def test_load_from_path_and_textio(self, tmp_path):
        path = tmp_path / "instance.gcnf"
        path.write_text(BASIC_GCNF)

        for source in (str(path), path, open(path)):
            model, soft, hard, assumptions = load_gdimacs(source)
            assert len(soft) == 2
            assert len(hard) == 1


class TestLoadGCNFErrors:

    def test_missing_p_line(self):
        with pytest.raises(AssertionError, match="Expected p-line before first clause"):
            load_gdimacs("{0} 1 2 0\n")

    def test_cnf_p_line(self):
        with pytest.raises(AssertionError):
            load_gdimacs("p cnf 2 1\n{0} 1 2 0\n")

    def test_unsupported_format_in_p_line(self):
        with pytest.raises(ValueError):
            load_gdimacs("p wcnf 2 1 3\n{0} 1 2 0\n")

    def test_missing_group_prefix(self):
        with pytest.raises(AssertionError, match="Expected clause to be prefixed with its group"):
            load_gdimacs("p gcnf 2 1 0\n1 2 0\n")

    def test_negative_group_number(self):
        with pytest.raises(AssertionError, match="Group number must be non-negative"):
            load_gdimacs("p gcnf 2 3 2\n{0} 1 2 0\n{1} 1 -2 0\n{-1} -1 -2 0\n")

    def test_too_many_variables(self):
        with pytest.raises(AssertionError):
            load_gdimacs("p gcnf 2 1 0\n{0} 1 2 3 0\n")

    def test_too_many_clauses(self):
        with pytest.raises(AssertionError, match="Too many clauses"):
            load_gdimacs("p gcnf 2 1 1\n{0} 1 0\n{1} 2 0\n")

    def test_too_few_clauses(self):
        with pytest.raises(AssertionError, match="Number of clauses did not match"):
            load_gdimacs("p gcnf 2 2 1\n{0} 1 0\n")

    def test_non_int_literal(self):
        with pytest.raises(ValueError):
            load_gdimacs("p gcnf 2 1 0\n{0} 1 b 2 0\n")

    def test_non_terminated_final_clause(self):
        with pytest.raises(AssertionError, match="terminated"):
            load_gdimacs("p gcnf 2 2 0\n{0} 1 2 0\n{0} -1\n")


# --------------------------------------------------------------------------- #
#                                Writing (GCNF)                               #
# --------------------------------------------------------------------------- #

@pytest.mark.requires_dependency("pindakaas")
class TestWriteGCNF:

    def test_p_header_counts(self):
        a, b = cp.boolvar(name="wa"), cp.boolvar(name="wb")
        text = write_gdimacs([~a, ~b], hard=[a | b])

        p, typ, n_vars, n_clauses, n_groups = text.splitlines()[0].split()
        assert (p, typ) == ("p", "gcnf")
        assert int(n_groups) == 2
        assert len(text.splitlines()) == 1 + int(n_clauses)

    def test_group_prefixes(self):
        a, b = cp.boolvar(name="wa"), cp.boolvar(name="wb")
        text = write_gdimacs([~a, ~b], hard=[a | b])

        groups = [line.split()[0] for line in text.splitlines()[1:]]
        assert set(groups) == {"{0}", "{1}", "{2}"}

    def test_only_hard(self):
        a, b = cp.boolvar(name="wa"), cp.boolvar(name="wb")
        text = write_gdimacs([], hard=[a | b], canonical=True)

        assert text == "p gcnf 2 1 0\n{0} 1 2 0\n"

    def test_write_to_file(self, tmp_path):
        a, b = cp.boolvar(name="wa"), cp.boolvar(name="wb")
        path = tmp_path / "model.gcnf"
        text = write_gdimacs([~a, ~b], hard=[a | b], path=path)

        assert path.read_text() == text

    def test_mus2011_example(self):
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

    def test_spec_example(self):
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


# --------------------------------------------------------------------------- #
#                                 Round-trips                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.requires_dependency("pindakaas")
class TestGCNFRoundtrip:

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
            """p gcnf 1 2 1
{0} 1 0
{1} 0
""",
            # With comments
            """c This is a comment
c Another comment line
p gcnf 2 3 1
c Comment in the middle
{0} 1 2 0
{1} -1 0
{1} -2 0
""",
        ],
        ids=[
            "basic",
            "only_hard",
            "single_group",
            "empty_clause",
            "with_comments",
        ],
    )
    def test_roundtrip(self, gcnf_str):
        model, soft, hard, assumptions = load_gdimacs(gcnf_str, var_name="x", assumption_name="a")

        back = write_gdimacs(soft, hard=hard, canonical=True, disjoint=False)

        gcnf_str = "\n".join(l for l in gcnf_str.split("\n") if not l.startswith("c"))
        assert back == gcnf_str, f"Roundtrip failed from:\n\n{gcnf_str}\nto\n\n{back}"

    def test_roundtrip_renumbers_groups(self):
        """Non-consecutive group ids are renumbered consecutively when writing back"""
        model, soft, hard, assumptions = load_gdimacs(
            "p gcnf 3 5 3\n{0} 1 2 0\n{1} 1 -2 0\n{1} -3 0\n{3} 2 3 0\n{3} -1 -2 0\n",
            var_name="x",
        )

        back = write_gdimacs(soft, hard=hard, canonical=True, disjoint=False)
        assert back.splitlines()[0] == "p gcnf 3 5 2"

    def test_roundtrip_preserves_semantics(self):
        model, soft, hard, assumptions = load_gdimacs(BASIC_GCNF, var_name="x")

        model2, soft2, hard2, assumptions2 = load_gdimacs(
            write_gdimacs(soft, hard=hard), var_name="y"
        )

        assert len(soft2) == len(soft)
        _assert_unsat(cp.Model(model2.constraints + list(assumptions2)))
        _assert_sat(cp.Model(model2.constraints + [assumptions2[0]]))
        _assert_sat(cp.Model(model2.constraints + [assumptions2[1]]))
