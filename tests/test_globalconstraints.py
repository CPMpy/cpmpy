import unittest
from cppy.solver_interfaces import get_supported_solvers
from cppy.model import Model
from cppy import IntVar
from cppy.globalconstraints import alldifferent, circuit

class TestGlobal(unittest.TestCase):
    def test_alldifferent(self):
        """Test all different constraint with a set of
        unit cases.
        """
        lb = 1
        start = 2
        nTests = 10
        for i in range(start, start + nTests):
            # construct the model vars = lb..i
            vars = IntVar(lb, i, i)

            # CONSTRAINTS
            constraint = [ alldifferent(vars) ]

            # MODEL Transformation to default solver specification
            model = Model(constraint)

            # SOLVE
            for solver in get_supported_solvers():
                _ = model.solve(solver=solver)
                vals = [x.value() for x in vars]

                # ensure all different values
                self.assertEqual(len(vals),len(set(vals)), msg=f"{solver.name} does provide solution validating given constraints.")

    def test_circuit(self):
        """
        Circuit constraint unit tests the hamiltonian circuit on a
        successor array. For example, if

            arr = [3, 0, 5, 4, 2, 1]

        then

            arr[0] = 3

        means that there is a directed edge from 0 -> 3.
        """
        # TODO implement circuit unit test
        vars = IntVar(0, 5, 6)
        constraints = [circuit(vars)]
        constraints += []
        model = Model(constraints)
        for solver in get_supported_solvers():
            _ = model.solve(solver=solver)
