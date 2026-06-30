"""
Tests for the ``cpmpy.tools.io`` load/write tools.

Both the writer and loader tests are driven by a single flat list of :class:`IOCase`
entries. Each case bundles, for one small problem:

- ``model``        : the canonical CPMpy model (writer input),
- ``filename``     : a small instance file under ``tests/data/io/loader`` (loader input),
- ``expected_repr``: the exact ``str(model)`` a correct loader must produce (ground truth).

The loader is tested in all three supported input shapes:
- a file path, 
- a raw content string,
- an open file handle
Every one must reproduce ``expected_repr`` exactly.

The auto-counter variable names (``BV0``/``IV0``...) depend on global counter state, 
so the counters are reset immediately before each load to keep the
ground-truth strings stable.
"""

import os
import re
import runpy
import tempfile
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple
from unittest.mock import patch

import pytest

import cpmpy as cp
from cpmpy.tools.io import load, write
from cpmpy.tools.io.loader import _get_loader
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl, _BV_PREFIX, _IV_PREFIX
from cpmpy.transformations.get_variables import get_variables_model


# --------------------------------------------------------------------------- #
#                          Optional-dependency gating                         #
# --------------------------------------------------------------------------- #

def has_pindakaas() -> bool:
    """Pindakaas backs the to_cnf/to_opb based writers (OPB/DIMACS/WCNF)."""
    try:
        from cpmpy.solvers.pindakaas import CPM_pindakaas
        return CPM_pindakaas.supported()
    except Exception:
        return False


def has_pyscipopt() -> bool:
    """pyscipopt backs the SCIP read/write formats (mps/lp/cip/fzn/pip/gms)."""
    try:
        from cpmpy.solvers.scip import CPM_scip
        return CPM_scip.supported()
    except Exception:
        return False


def has_pycsp3() -> bool:
    """pycsp3 backs the XCSP3 reader (``read_xcsp3``)."""
    # find_spec locates the package without importing it (importing pycsp3 runs its
    # compiler against sys.argv, which we must avoid at collection time).
    try:
        return importlib.util.find_spec("pycsp3") is not None
    except Exception:
        return False


_CAPABILITIES = {
    "pindakaas": has_pindakaas,
    "pyscipopt": has_pyscipopt,
    "pycsp3": has_pycsp3,
}


def _missing(needs) -> list:
    return [cap for cap in needs if not _CAPABILITIES[cap]()]


def _capability_from_requirement(req: str) -> str:
    return re.split(r"[<>=!~;\s\[]", req, 1)[0].lower()


@lru_cache(maxsize=1)
def _format_dependencies() -> dict[str, Tuple[str, ...]]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_path = os.path.join(repo_root, "setup.py")
    old_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        with patch("setuptools.setup"):
            setup_vars = runpy.run_path(setup_path)
    finally:
        os.chdir(old_cwd)

    dependencies = {}
    for extra, reqs in setup_vars["format_dependencies"].items():
        if not extra.startswith("io.") or extra == "io.all":
            continue
        fmt = extra.removeprefix("io.")
        dependencies[fmt] = tuple(
            cap for req in reqs
            if (cap := _capability_from_requirement(req)) in _CAPABILITIES
        )
    return dependencies


def _format_needs(format: Optional[str]) -> Tuple[str, ...]:
    if format is None:
        return ()
    return _format_dependencies().get(format, ())


# --------------------------------------------------------------------------- #
#                                  Fixtures                                   #
# --------------------------------------------------------------------------- #

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "io")
ATTRIBUTION_FILE = "ATTRIBUTION.md"

PUBLIC_FIXTURES = [
    "jsplib_ft06.txt",
    "rcpsp_j3013_1.sm",
    "nurserostering_instance1.txt",
    "large_tseitin_n18.cnf",
    "large_tseitin_n18.opb",
    "large_ramsey_k4_n5.wcnf",
    "large_same_queens_knights_05.xml",
]


def _fixture(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def _reset_var_counters() -> None:
    """Reset the global variable-name counters so loaded auto-var names are stable."""
    _BoolVarImpl.counter = 0
    _IntVarImpl.counter = 0


def _assert_same_solutions(ref: cp.Model, other: cp.Model) -> None:
    """Assert two models are solve-equivalent.

    - optimization models: both solve and reach the same optimal objective value;
    - satisfaction models: both have the same number of solutions.
    Variable names/order and auxiliary encoding variables may differ between the two,
    so we compare observable behaviour rather than structure.
    """
    if ref.has_objective():
        assert ref.solve(), "reference model should solve"
        assert other.solve(), "round-tripped model should solve"
        assert ref.objective_value() == other.objective_value(), \
            f"objective mismatch: {ref.objective_value()} != {other.objective_value()}"
    else:
        n_ref = ref.solveAll(solution_limit=1000)
        n_other = other.solveAll(solution_limit=1000)
        assert n_ref == n_other, f"solution count mismatch: {n_ref} != {n_other}"


# --------------------------------------------------------------------------- #
#                               Canonical models                              #
# --------------------------------------------------------------------------- #
# Each model is small enough to write its loaded form down by hand as ground truth.
# The matching instance files live in ``tests/data/io/loader/``.

def opb_model() -> cp.Model:
    x = cp.boolvar(shape=3, name="x")
    m = cp.Model(x[0] + 2 * x[1] - x[2] >= 2)
    m.minimize(2 * x[0] + x[1])
    return m

def cnf_model() -> cp.Model:
    a, b, c = [cp.boolvar(name=n) for n in "abc"]
    return cp.Model(cp.any([a, b, c]), b.implies(~c), ~a)

def wcnf_model() -> cp.Model:
    x = cp.boolvar(shape=3, name="x")
    m = cp.Model(cp.any(x))
    m.minimize(x[0] + x[1] + x[2])
    return m

def scip_model() -> cp.Model:
    x = cp.intvar(0, 10, name="x")
    y = cp.intvar(0, 10, name="y")
    m = cp.Model(x + y == 5)
    m.maximize(x + 2 * y)
    return m


# --------------------------------------------------------------------------- #
#                                  IOCase                                      #
# --------------------------------------------------------------------------- #

@dataclass
class IOCase:
    """
    A dataclass to help write IO related tests.
    """
    id: str
    model: cp.Model                          # writer input
    filename: str                            # instance under tests/data/io/loader
    expected_repr: str                       # exact str(model) a correct loader produces
    load_format: Optional[str] = None        # key for load(); None -> not loadable
    write_format: Optional[str] = None        # key for write(); None -> not writable
    # loader capabilities
    autodetect: bool = True                  # load(path) (no format) resolves to this format
    string_load: bool = True                 # loader accepts a raw string / open handle (not path-only)
    # round-trip support (solution-equivalence)
    roundtrip_model: bool = True             # write(model) -> load gives an equivalent model
    roundtrip_instance: bool = True          # load(file) -> write -> load gives an equivalent model


# DIMACS/WCNF auto-name variables (BV0/BV1/...), so the ground truth depends on the
# (reset) counter state; OPB/SCIP name them from the file.
CASES = [
    IOCase(
        id="opb",
        model=opb_model(),
        filename="basic.opb",
        expected_repr="Constraints:\n    sum([1, 2, -1] * [x1, x2, x3]) >= 2\n"
                      "Objective: minimize sum([2, 1] * [x1, x2])",
        load_format="opb",
        write_format="opb",
    ),
    IOCase(
        id="cnf",
        model=cnf_model(),
        filename="basic.cnf",
        expected_repr="Constraints:\n    (~BV1) or (~BV2)\n    or(BV2, BV1, BV0)\n    ~BV0\n"
                      "Objective: None",
        load_format="cnf",
        write_format="cnf",
    ),
    IOCase(
        id="cnf_dimacs",
        model=cnf_model(),
        filename="basic.cnf",
        expected_repr="Constraints:\n    (~BV1) or (~BV2)\n    or(BV2, BV1, BV0)\n    ~BV0\n"
                      "Objective: None",
        load_format="dimacs",
        write_format="dimacs",
    ),
    IOCase(
        id="wcnf",
        model=wcnf_model(),
        filename="basic.wcnf",
        expected_repr="Constraints:\n    or(x1, x2, x3)\n"
                      "Objective: minimize sum(x1, x2, x3)",
        load_format="wcnf",
        write_format="wcnf",
        # write_dimacs' WCNF objective encoding does not round-trip through the loader
        # (the soft-clause weights are not recovered), so neither direction is equivalent.
        roundtrip_model=False,
        roundtrip_instance=False,
    ),
    IOCase(
        id="wcnf_dimacs",
        model=wcnf_model(),
        filename="basic.wcnf",
        expected_repr="Constraints:\n    or(x1, x2, x3)\n"
                      "Objective: minimize sum(x1, x2, x3)",
        load_format="dimacs",
        write_format="dimacs",
        roundtrip_model=False,
        roundtrip_instance=False,
    ),
    IOCase(
        id="mps",
        model=scip_model(),
        filename="basic.mps",
        expected_repr="Constraints:\n    (y) + (x) >= 5\n    (y) + (x) <= 5\n"
                      "Objective: maximize sum([2, 1] * [y, x])",
        load_format="mps",
        write_format="mps",
    ),
    IOCase(
        id="lp",
        model=scip_model(),
        filename="basic.lp",
        expected_repr="Constraints:\n    (x) + (y) >= 5\n    (x) + (y) <= 5\n"
                      "Objective: maximize sum([2, 1] * [y, x])",
        load_format="lp",
        write_format="lp",
    ),
    IOCase(
        id="cip",
        model=scip_model(),
        filename="basic.cip",
        expected_repr="Constraints:\n    (x) + (y) >= 5\n    (x) + (y) <= 5\n"
                      "Objective: maximize sum([2, 1] * [y, x])",
        load_format="cip",
        write_format="cip",
    ),
    IOCase(
        id="fzn",
        model=scip_model(),
        filename="basic.fzn",
        expected_repr="Constraints:\n    (x) + (y) >= 5\n    (x) + (y) <= 5\n"
                      "Objective: maximize sum([2, 1] * [y, x])",
        load_format="fzn",
        write_format="fzn",
    ),
    IOCase(
        id="pip",
        model=scip_model(),
        filename="basic.pip",
        expected_repr="Constraints:\n    (x) + (y) >= 5\n    (x) + (y) <= 5\n"
                      "Objective: maximize sum([2, 1] * [y, x])",
        load_format="pip",
        write_format="pip",
    ),
    # gms can be written but not read back (this SCIP build has no GAMS reader plugin).
    IOCase(
        id="gms",
        model=scip_model(),
        filename="basic.gms",
        expected_repr="",
        load_format=None,
        write_format="gms",
    ),
    # XCSP3 is read-only (no writer), reads a file path only, and is registered in the
    # loader dispatch but deliberately not in the extension map: it cannot be auto-detected.
    IOCase(
        id="xcsp3",
        model=None,
        filename="basic.xml",
        expected_repr="Constraints:\n    (x) + (-(y)) < 0\nObjective: None",
        load_format="xcsp3",
        write_format=None,
        autodetect=False,
        string_load=False,
    ),
]


def _case_needs(case, *modes) -> Tuple[str, ...]:
    needs = []
    for mode in modes:
        if mode == "load":
            needs.extend(_format_needs(case.load_format))
        elif mode == "write":
            needs.extend(_format_needs(case.write_format))
        else:
            raise ValueError(f"unknown dependency mode: {mode}")
    return tuple(dict.fromkeys(needs))


def _params(cases, *modes):
    """
    Build pytest params, skipping a case when any required capability is missing.
    """
    out = []
    for c in cases:
        needs = _case_needs(c, *modes)
        missing = _missing(needs)
        marks = [pytest.mark.skip(reason=f"requires {missing}")] if missing else []
        out.append(pytest.param(c, id=c.id, marks=marks))
    return out


LOAD_CASES = [c for c in CASES if c.load_format is not None]
WRITE_CASES = [c for c in CASES if c.write_format is not None]
# loaders that accept a raw string / open handle (not path-only)
STRING_LOAD_CASES = [c for c in LOAD_CASES if c.string_load]
# loaders whose format can / cannot be auto-detected from the file extension
AUTODETECT_CASES = [c for c in LOAD_CASES if c.autodetect]
NON_AUTODETECT_CASES = [c for c in LOAD_CASES if not c.autodetect]
# round-trip cases need both a loader and a writer
RT_MODEL_CASES = [c for c in CASES if c.load_format and c.write_format and c.roundtrip_model]
RT_INSTANCE_CASES = [c for c in CASES if c.load_format and c.write_format and c.roundtrip_instance]


# --------------------------------------------------------------------------- #
#                                   Writer                                    #
# --------------------------------------------------------------------------- #

class TestWriter:

    @pytest.mark.parametrize("case", _params(WRITE_CASES, "write"))
    def test_write_to_string(self, case):
        # the writer returns non-empty content
        text = write(case.model, format=case.write_format)
        assert isinstance(text, str)
        assert text.strip() != ""

    @pytest.mark.parametrize("case", _params(WRITE_CASES, "write"))
    def test_write_to_file(self, case):
        # writing to a file produces exactly what the writer returns (empty header so the
        # file is identical to the returned string)
        with tempfile.NamedTemporaryFile(suffix=f".{case.write_format}") as tmp:
            text = write(case.model, path=tmp.name, format=case.write_format, header="")
            with open(tmp.name, "r") as f:
                written = f.read()
        assert written.strip() != ""
        assert text == written

    @pytest.mark.parametrize("case", _params(WRITE_CASES, "write"))
    def test_header(self, case):
        # the provided header is written out verbatim
        header = "This is a header\n----------------"
        text = write(case.model, format=case.write_format, header=header)
        assert "This is a header" in text
        assert "----------------" in text


# --------------------------------------------------------------------------- #
#                                   Loader                                     #
# --------------------------------------------------------------------------- #

class TestLoader:

    @pytest.mark.parametrize("case", _params(STRING_LOAD_CASES, "load"))
    def test_load_from_string(self, case):
        with open(_fixture(case.filename), "r") as f:
            text = f.read()
        _reset_var_counters()
        model = load(text, format=case.load_format)
        assert model is not None
        assert str(model) == case.expected_repr

    @pytest.mark.parametrize("case", _params(LOAD_CASES, "load"))
    def test_load_from_file(self, case):
        _reset_var_counters()
        model = load(_fixture(case.filename), format=case.load_format)
        assert model is not None
        assert str(model) == case.expected_repr

    @pytest.mark.parametrize("case", _params(STRING_LOAD_CASES, "load"))
    def test_load_from_textio(self, case):
        _reset_var_counters()
        with open(_fixture(case.filename), "r") as f:
            model = load(f, format=case.load_format)
        assert model is not None
        assert str(model) == case.expected_repr

    @pytest.mark.parametrize("case", _params(LOAD_CASES, "load"))
    def test_get_loader(self, case):
        # every load format resolves to a callable loader
        assert callable(_get_loader(case.load_format))

    @pytest.mark.parametrize("case", _params(AUTODETECT_CASES, "load"))
    def test_load_autodetect_from_path(self, case):
        # loading a file path without an explicit format auto-detects it from the extension
        _reset_var_counters()
        explicit = load(_fixture(case.filename), format=case.load_format)
        _reset_var_counters()
        autodetected = load(_fixture(case.filename))
        assert str(autodetected) == str(explicit)

    @pytest.mark.parametrize("case", _params(NON_AUTODETECT_CASES, "load"))
    def test_load_autodetect_unsupported(self, case):
        # formats declared non-auto-detectable (e.g. xcsp3: its .xml extension is not in the
        # format map) must require an explicit format=; loading by path alone fails.
        with pytest.raises((ValueError, KeyError)):
            load(_fixture(case.filename))

    @pytest.mark.parametrize("case", _params(LOAD_CASES, "load"))
    def test_load_updates_var_counters(self, case):
        # After loading, the global name counters must be advanced past every auto-named
        # (BV*/IV*) variable the loader created, so variables created afterwards never
        # collide with the loaded model's variables. 
        # (Formats that name their variablesfrom the file, like OPB's x1.. and SCIP's x,y, 
        # leave the counters untouched, which is fine.)
        _reset_var_counters()
        model = load(_fixture(case.filename), format=case.load_format)
        loaded_names = {v.name for v in get_variables_model(model)}

        def _expected_counter(prefix):
            idxs = [int(n[len(prefix):]) for n in loaded_names
                    if n.startswith(prefix) and n[len(prefix):].isdigit()]
            return max(idxs) + 1 if idxs else 0

        assert _BoolVarImpl.counter >= _expected_counter(_BV_PREFIX)
        assert _IntVarImpl.counter >= _expected_counter(_IV_PREFIX)

        # practical consequence: freshly created variables reuse none of the loaded names
        fresh = {cp.boolvar().name for _ in range(3)} | {cp.intvar(0, 1).name for _ in range(3)}
        assert loaded_names.isdisjoint(fresh)

    def test_load_string_without_format_raises(self):
        # a raw (multi-line) string cannot be auto-detected -> an explicit format is required
        with open(_fixture("basic.opb"), "r") as f:
            text = f.read()
        with pytest.raises(ValueError):
            load(text)

    def test_public_fixtures_are_attributed(self):
        with open(_fixture(ATTRIBUTION_FILE), "r") as f:
            attribution = f.read()
        for filename in PUBLIC_FIXTURES:
            assert filename in attribution


# --------------------------------------------------------------------------- #
#                                 Round-trip                                  #
# --------------------------------------------------------------------------- #

class TestRoundtrip:
    """End-to-end load/write/load consistency, checked by solution-equivalence.

    Writing then reading back renames variables and may add auxiliary encoding variables,
    so the model structure is not preserved verbatim; what must be preserved is the
    observable behaviour (optimal objective, or number of solutions)."""

    @pytest.mark.parametrize("case", _params(RT_MODEL_CASES, "write", "load"))
    def test_write_then_load(self, case):
        # model -> write -> load yields a solution-equivalent model
        text = write(case.model, format=case.write_format)
        loaded = load(text, format=case.load_format)
        _assert_same_solutions(case.model, loaded)

    @pytest.mark.parametrize("case", _params(RT_INSTANCE_CASES, "write", "load"))
    def test_load_write_load(self, case):
        # instance -> load -> write -> load yields a solution-equivalent model
        first = load(_fixture(case.filename), format=case.load_format)
        text = write(first, format=case.write_format)
        second = load(text, format=case.load_format)
        _assert_same_solutions(first, second)


# --------------------------------------------------------------------------- #
#                            Larger instances (coarse)                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LargerInstanceCase:
    id: str
    filename: str
    load_format: str
    expected_vars: int
    expected_constraints: int
    string_load: bool = True
    solve_result: Optional[bool] = None


LARGER_INSTANCE_CASES = [
    LargerInstanceCase(
        id="cnf-tseitin",
        filename="large_tseitin_n18.cnf",
        load_format="cnf",
        expected_vars=27,
        expected_constraints=72,
    ),
    LargerInstanceCase(
        id="dimacs-tseitin",
        filename="large_tseitin_n18.cnf",
        load_format="dimacs",
        expected_vars=27,
        expected_constraints=72,
    ),
    LargerInstanceCase(
        id="wcnf-ramsey",
        filename="large_ramsey_k4_n5.wcnf",
        load_format="wcnf",
        expected_vars=10,
        expected_constraints=0,
    ),
    LargerInstanceCase(
        id="opb-tseitin",
        filename="large_tseitin_n18.opb",
        load_format="opb",
        expected_vars=27,
        expected_constraints=72,
    ),
    LargerInstanceCase(
        id="mps-linear",
        filename="large_linear.mps",
        load_format="mps",
        expected_vars=12,
        expected_constraints=24,
    ),
    LargerInstanceCase(
        id="lp-linear",
        filename="large_linear.lp",
        load_format="lp",
        expected_vars=12,
        expected_constraints=24,
    ),
    LargerInstanceCase(
        id="cip-linear",
        filename="large_linear.cip",
        load_format="cip",
        expected_vars=12,
        expected_constraints=24,
    ),
    LargerInstanceCase(
        id="fzn-linear",
        filename="large_linear.fzn",
        load_format="fzn",
        expected_vars=12,
        expected_constraints=24,
    ),
    LargerInstanceCase(
        id="pip-linear",
        filename="large_linear.pip",
        load_format="pip",
        expected_vars=12,
        expected_constraints=24,
    ),
    LargerInstanceCase(
        id="xcsp3-same-queens-knights",
        filename="large_same_queens_knights_05.xml",
        load_format="xcsp3",
        expected_vars=77,
        expected_constraints=102,
    ),
]


def _larger_params():
    return _params(LARGER_INSTANCE_CASES, "load")


def _larger_string_params():
    return _params([case for case in LARGER_INSTANCE_CASES if case.string_load], "load")


LARGER_WRITER_ONLY_FILES = [
    ("gms-linear", "large_linear.gms"),
]


class TestLargerInstance:
    """Larger real instances: loaders must not fail and models must be well-formed.

    These examples are still small enough for unit tests, but large enough to catch
    parser mistakes that the hand-written toy instances above tend to miss.
    """

    def _check(self, case, model):
        assert model is not None
        assert len(get_variables_model(model)) == case.expected_vars
        assert len(model.constraints) == case.expected_constraints
        if case.solve_result is not None:
            assert model.solve() is case.solve_result

    @pytest.mark.parametrize("case", _larger_params())
    def test_from_file(self, case):
        self._check(case, load(_fixture(case.filename), format=case.load_format))

    @pytest.mark.parametrize("case", _larger_string_params())
    def test_from_string(self, case):
        with open(_fixture(case.filename), "r") as f:
            text = f.read()
        self._check(case, load(text, format=case.load_format))

    @pytest.mark.parametrize("case", _larger_string_params())
    def test_from_textio(self, case):
        with open(_fixture(case.filename), "r") as f:
            self._check(case, load(f, format=case.load_format))

    @pytest.mark.parametrize("case_id,filename", LARGER_WRITER_ONLY_FILES)
    def test_writer_only_file_exists(self, case_id, filename):
        assert case_id
        assert os.path.getsize(_fixture(filename)) > 0
