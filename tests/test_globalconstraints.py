import unittest
import cpmpy as cp

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
            vars = cp.IntVar(lb, i, i)

            # CONSTRAINTS
            constraint = [ cp.alldifferent(vars) ]

            # MODEL Transformation to default solver specification
            model = cp.Model(constraint)

            # SOLVE
            if True:
                _ = model.solve()
                vals = [x.value() for x in vars]

                # ensure all different values
                self.assertEqual(len(vals),len(set(vals)), msg=f"solver does provide solution validating given constraints.")

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
        x = cp.IntVar(0, 5, 6)
        constraints = [cp.circuit(x)]
        model = cp.Model(constraints)

        _ = model.solve()

    def test_minimax_python(self):
        from cpmpy import min,max
        iv = cp.IntVar(1,9, 10)
        self.assertIsInstance(min(iv), cp.GlobalConstraint) 
        self.assertIsInstance(max(iv), cp.GlobalConstraint) 

    def test_minimax_cpm(self):
        iv = cp.IntVar(1,9, 10)
        mi = cp.min(iv)
        ma = cp.max(iv)
        self.assertIsInstance(mi, cp.GlobalConstraint) 
        self.assertIsInstance(ma, cp.GlobalConstraint) 

        self.assertEqual(cp.Model([], minimize=mi).solve(), 1)
        self.assertEqual(cp.Model([], minimize=ma).solve(), 1)
        self.assertEqual(cp.Model([], maximize=mi).solve(), 9)
        self.assertEqual(cp.Model([], maximize=ma).solve(), 9)
