"""
Tests for ``cpmpy.tools.verify``.
"""
import os
import sys
import tempfile
from shutil import which

import pytest

from cpmpy.tools.verify import verify_prooflog
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.solvers.gcs import CPM_gcs
import cpmpy as cp


@pytest.fixture
def dummy_proof():
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(b"x")
    f.close()
    yield f.name
    os.unlink(f.name)


class TestVerifyTool:

    def test_missing_verifier(self, dummy_proof):
        with pytest.raises(Exception, match="Unable to run"):
            verify_prooflog("cpmpy-no-such-verifier", [dummy_proof])

    def test_valid_exit(self, dummy_proof):
        status = verify_prooflog("true", [dummy_proof])
        assert status["result"] is True
        assert status["timeout"] is False
        assert status["runtime"] >= 0
        assert "error_message" not in status

    def test_invalid_exit(self, dummy_proof):
        status = verify_prooflog("false", [dummy_proof])
        assert status["result"] is False
        assert status["timeout"] is False
        assert status["error_message"] == ""

    def test_timeout(self, dummy_proof):
        status = verify_prooflog(sys.executable, [dummy_proof],
                        time_limit=0.2,
                        verifier_args=["-c", "import time; time.sleep(5)"])
        assert status["result"] is False
        assert status["timeout"] is True
        assert status["runtime"] >= 0.2

    def test_display_output(self, dummy_proof, capfd):
        status = verify_prooflog("echo", [dummy_proof], display_output=True, verifier_args=["hello"])
        assert status["result"] is True
        captured = capfd.readouterr()
        assert "hello" in captured.out

    def test_verifier_args_recorded(self, dummy_proof):
        args = ["-n"]
        status = verify_prooflog("true", [dummy_proof], verifier_args=args)
        assert status["verifier_args"] == args


@pytest.mark.skipif(not CPM_pysat.supported(), reason="PySAT not installed")
@pytest.mark.skipif(which("drat-trim") is None, reason="drat-trim not installed")
class TestVerifyDratTrim:

    def test_valid_pysat_proof(self):
        x, y, z = cp.intvar(1, 5, shape=3)
        m = cp.Model(x < y, y < z, z < x)
        proof = tempfile.NamedTemporaryFile(delete=False).name
        s = cp.SolverLookup.get("pysat", m, proof=proof)
        assert s.solve() is False

        status = verify_prooflog("drat-trim", s.get_proof_files())
        assert status["result"] is True
        assert status["timeout"] is False

    def test_invalid_pysat_proof(self):
        x, y, z = cp.intvar(1, 5, shape=3)
        m = cp.Model(x < y, y < z, z < x)
        proof = tempfile.NamedTemporaryFile(delete=False).name
        s = cp.SolverLookup.get("pysat", m, proof=proof)
        assert s.solve() is False

        cnf, drat = s.get_proof_files()
        with open(drat, "w") as f:
            f.write("")

        status = verify_prooflog("drat-trim", [cnf, drat])
        assert status["result"] is False
        assert "error_message" in status


@pytest.mark.skipif(not CPM_gcs.supported(), reason="GCS not installed")
@pytest.mark.skipif(which("veripb") is None, reason="veripb not installed")
class TestVerifyVeripb:

    def test_valid_gcs_proof(self):
        x, y, z = cp.intvar(1, 5, shape=3)
        m = cp.Model(x < y, y < z, z < x)
        proof = "gcs_verify_proof"
        s = cp.SolverLookup.get("gcs", m, proof=proof)
        assert s.solve() is False

        opb, pbp, _varmap = s.get_proof_files()
        status = verify_prooflog("veripb", [opb, pbp])
        assert status["result"] is True
        assert status["timeout"] is False

    def test_invalid_gcs_proof(self):
        x, y, z = cp.intvar(1, 5, shape=3)
        m = cp.Model(x < y, y < z, z < x)
        proof = "gcs_verify_proof_invalid"
        s = cp.SolverLookup.get("gcs", m, proof=proof)
        assert s.solve() is False

        opb, pbp, _varmap = s.get_proof_files()
        with open(pbp, "w") as f:
            f.write("")

        status = verify_prooflog("veripb", [opb, pbp])
        assert status["result"] is False
        assert "error_message" in status
