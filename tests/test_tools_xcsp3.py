"""
Tests for the XCSP3 parser callbacks (:mod:`cpmpy.tools.xcsp3`).
"""
import numpy as np
import pytest

import cpmpy as cp

pytestmark = pytest.mark.requires_dependency("pycsp3")


@pytest.fixture
def callbacks():
    # pycsp3 parses sys.argv at import time; disable its auto-compile first
    from cpmpy.tools.xcsp3.parser import _disable_pycsp3_auto_compile
    _disable_pycsp3_auto_compile()
    from cpmpy.tools.xcsp3.parser_callbacks import CallbacksCPMPy
    return CallbacksCPMPy()


class TestCallbacksCPMPy:

    def test_get_cpm_exprs_empty(self, callbacks):
        # fix from xcsp3_26 branch (commit bd9b6a723): empty input must return [],
        # both for plain lists and numpy arrays (where `not lst` raises/misbehaves)
        assert callbacks.get_cpm_exprs([]) == []
        assert callbacks.get_cpm_exprs(np.array([])) == []

    def test_get_cpm_var_non_xvar_key(self, callbacks):
        # fix from xcsp3_26 branch (commit 7e72cc896): keys registered in
        # cpm_variables must be mapped even when they are not XVar instances
        v = cp.intvar(0, 3, name="x")
        callbacks.cpm_variables["k"] = v
        assert callbacks.get_cpm_var("k") is v
        # constants still pass through untouched
        assert callbacks.get_cpm_var(5) == 5
