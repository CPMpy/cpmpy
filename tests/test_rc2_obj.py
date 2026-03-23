import unittest
import pytest
import cpmpy as cp
from cpmpy import *
from cpmpy.solvers.rc2 import CPM_rc2

# Check if RC2 is available
rc2_available = CPM_rc2.supported()

@pytest.mark.skipif(not rc2_available, reason="RC2 solver not available")
class TestRC2Transform(unittest.TestCase):
    """
    Test cases for RC2 solver objective transformation functionality
    Based on the test cases from rc2.ipynb
    """
    
    def setUp(self):
        """Set up test variables similar to the notebook"""
        self.xs = cp.boolvar(3)
        self.ys = cp.intvar(1, 4, shape=3)
        self.rc2 = CPM_rc2()
        self.rc2.encoding = "direct"
    
    def test_rc2_solver_creation(self):
        """Test that RC2 solver can be created and accessed via SolverLookup"""
        rc2_solver = cp.SolverLookup.get("rc2")
        self.assertIsInstance(rc2_solver, CPM_rc2)
    
    def test_transform_objective_single_bool(self):
        """Test objective transformation with single boolean variable"""
        # Test xs[0] -> flat_obj BV0
        weights, xs, const = self.rc2.transform_objective(self.xs[0])
        self.assertEqual(weights, [1])
        self.assertEqual(xs, [self.xs[0]])
        self.assertEqual(const, 0)

    def test_transform_objective_sum_bool(self):
        """Test objective transformation with sum of boolean variables"""
        # Test sum(xs) -> flat_obj sum([BV0, BV1, BV2])
        weights, xs, const = self.rc2.transform_objective(cp.sum(self.xs))
        self.assertEqual(weights, [1, 1, 1])
        self.assertEqual(xs, self.xs.tolist())
        self.assertEqual(const, 0)
    
    def test_transform_objective_sum_bool_plus_const(self):
        """Test objective transformation with sum of boolean variables plus constant"""
        # Test sum(xs) + 3 -> flat_obj sum([BV0, BV1, BV2, 3])
        weights, xs, const = self.rc2.transform_objective(cp.sum(self.xs) + 3)
        self.assertEqual(weights, [1, 1, 1])
        self.assertEqual(xs, self.xs.tolist())
        self.assertEqual(const, 3)
    
    def test_transform_objective_sum_bool_minus_const(self):
        """Test objective transformation with sum of boolean variables minus constant"""
        # Test sum(xs) - 2 -> flat_obj sum([BV0, BV1, BV2, -2])
        weights, xs, const = self.rc2.transform_objective(cp.sum(self.xs) - 2)
        self.assertEqual(weights, [1, 1, 1])
        self.assertEqual(xs, self.xs.tolist())
        self.assertEqual(const, -2)
    
    def test_transform_objective_scaled_sum_bool(self):
        """Test objective transformation with scaled sum of boolean variables"""
        # Test 3*sum(xs) -> flat_obj sum([3, 3, 3] * [BV0, BV1, BV2])
        weights, xs, const = self.rc2.transform_objective(3 * cp.sum(self.xs))
        self.assertEqual(weights, [3, 3, 3])
        self.assertEqual(xs, self.xs.tolist())
        self.assertEqual(const, 0)
    
    def test_transform_objective_linear_combination_bool(self):
        """Test objective transformation with linear combination of boolean variables"""
        # Test 3*xs[0] + 2*xs[1] - 4*xs[2] -> flat_obj sum([3, 2, -4] * [BV0, BV1, BV2])
        weights, xs, const = self.rc2.transform_objective(3*self.xs[0] + 2*self.xs[1] - 4*self.xs[2])
        self.assertEqual(weights, [3, 2, 4])  # negative weight should be flipped
        self.assertEqual(xs, [self.xs[0], self.xs[1], ~self.xs[2]])  # last variable should be negated
        self.assertEqual(const, -4)  # constant should include the flipped weight
    
    def test_transform_objective_linear_combination_bool_plus_const(self):
        """Test objective transformation with linear combination plus constant"""
        # Test 3*xs[0] + 2*xs[1] + 1*xs[2] + 12 -> flat_obj (IV6) + 12
        weights, xs, const = self.rc2.transform_objective(3*self.xs[0] + 2*self.xs[1] + 1*self.xs[2] + 12)
        # This creates an intermediate variable for the sum, which gets encoded... TODO: could do better without intermediate variable!
        self.assertEqual(len(weights), 6)  # Int encoding weights
        self.assertEqual(len(xs), 6)  # Int encoding variables
        self.assertEqual(const, 12)
    
    def test_transform_objective_single_int(self):
        """Test objective transformation with single integer variable"""
        # Test ys[0] -> flat_obj IV0
        # Integer variables are encoded as weighted sums of boolean variables
        weights, xs, const = self.rc2.transform_objective(self.ys[0])
        self.assertEqual(const, 1)  # offset min domain value of 1
        self.assertEqual(weights, [1,2,3])  # unary encoding weights
        self.assertEqual(len(xs), 3)  # unary encoding variables
    
    def test_transform_objective_sum_int(self):
        """Test objective transformation with sum of integer variables"""
        # Test sum(ys) -> flat_obj sum([IV0, IV1, IV2])
        # Integer variables are encoded as weighted sums of boolean variables
        weights, xs, const = self.rc2.transform_objective(cp.sum(self.ys))
        # Each integer variable is encoded as a weighted sum of boolean variables
        self.assertEqual(const, 3)  # offset each min domain value
        self.assertEqual(weights, [1,2,3]*3)  # unary encoding weights
        self.assertEqual(len(xs), 9)  # unary encoding variables
    
    def test_transform_objective_sum_int_plus_const(self):
        """Test objective transformation with sum of integer variables plus constant"""
        # Test sum(ys) + 3 -> flat_obj sum([IV0, IV1, IV2, 3])
        # Integer variables are encoded as weighted sums of boolean variables
        weights, xs, const = self.rc2.transform_objective(cp.sum(self.ys) + 3)
        # Each integer variable is encoded as a weighted sum of boolean variables
        self.assertEqual(const, 6)  # offset each min domain value + added constant
        self.assertEqual(weights, [1,2,3]*3)  # unary encoding weights
        self.assertEqual(len(xs), 9)  # unary encoding variables
    
    def test_transform_objective_linear_combination_int_plus_const(self):
        """Test objective transformation with linear combination of integer variables plus constant"""
        # Test 3*ys[0] + 2*ys[1] - 4*ys[2] + 12 -> flat_obj (IV8) + 12
        weights, xs, const = self.rc2.transform_objective(3*self.ys[0] + 2*self.ys[1] - 4*self.ys[2] + 12)
        # TODO... This creates an intermediate variable for the sum, which gets encoded
        self.assertGreater(len(weights), 0)  # Should have some weights
        self.assertGreater(len(xs), 0)  # Should have some variables
    
    def test_transform_objective_mixed_vars(self):
        """Test objective transformation with mixed boolean and integer variables"""
        # Test xs[0] + ys[0] + 2*xs[1] - 3*ys[1] -> flat_obj sum([1, 1, 2, -3] * [BV0, IV0, BV1, IV1])
        weights, xs, const = self.rc2.transform_objective(self.xs[0] + self.ys[0] + 2*self.xs[1] + 3*self.ys[1])
        # Integer variables are encoded as weighted sums of boolean variables
        self.assertEqual(weights, [1]+[1,2,3]+[2]+[3,6,9])
        self.assertEqual(len(weights), 1+3+1+3)
        self.assertEqual(const, 4)
    
    def test_transform_objective_mixed_vars_plus_const(self):
        """Test objective transformation with mixed variables plus constant"""
        # Test 3 + xs[0] + ys[0] + 2*xs[1] - 3*ys[1] - 12 -> flat_obj (IV9) + -12
        weights, xs, const = self.rc2.transform_objective(3 + self.xs[0] + self.ys[0] + 2*self.xs[1] - 3*self.ys[1] - 12)
        # TODO... gets auxiliary, can do better
        self.assertGreater(len(weights), 0)  # Should have some weights
        self.assertGreater(len(xs), 0)  # Should have some variables
    
@pytest.mark.skipif(not rc2_available, reason="RC2 solver not available")
@pytest.mark.parametrize("solve_kwargs", (
    {"solver": "mc"},
    {"solver": "g3"},
    # {"solver": "cadical195", "native_card": True},  # TODO if solver args become supported for RC2
))
class TestRC2Solve:
    def test_rc2_solve_simple_maximization(self, solve_kwargs):
        """Test actual solving with RC2 for a simple maximization problem"""
        # Create a simple model: maximize sum of boolean variables
        model = cp.Model()
        x = cp.boolvar(3)
        model.maximize(cp.sum(x))
        # Add some constraints
        model += x[0] | x[1]  # at least one of first two must be true
        model += x[1].implies(x[2])  # if x[1] is true, then x[2] must be true
        
        # Solve with RC2
        rc2 = CPM_rc2(model)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved
        assert rc2.objective_value() is not None
        assert rc2.objective_value() == 3
    
    def test_rc2_solve_minimization(self, solve_kwargs):
        """Test actual solving with RC2 for a minimization problem"""
        # Create a simple model: minimize sum of boolean variables
        model = cp.Model()
        x = cp.boolvar(3)
        model.minimize(cp.sum(x))
        # Add constraints that force some variables to be true
        model += x[0] == 1  # x[0] must be true
        model += x[1] | x[2]  # at least one of x[1] or x[2] must be true
        
        # Solve with RC2
        rc2 = CPM_rc2(model)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved is True
        assert rc2.objective_value() == 2
    
    def test_rc2_solve_with_integer_variables(self, solve_kwargs):
        """Test solving with integer variables in the objective"""
        # Create a model with integer variables
        model = cp.Model()
        x = cp.boolvar(2)
        y = cp.intvar(0, 3, shape=2)
        z = cp.intvar(0, 3)
        model.maximize(cp.sum(x) + cp.sum(y) + 0 * z)
        # Add constraints
        model += (x[0] != x[1])  # both must be different
        model += (y[0] < y[1])  # y[0] must be less than y[1]
        
        # Solve with RC2
        rc2 = CPM_rc2(model)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved is True
        assert rc2.objective_value() == 1+2+3 # 1 from bool, 2+3 from int
    
    def test_rc2_unsatisfiable(self, solve_kwargs):
        """Test RC2 with an unsatisfiable model"""
        # Create an unsatisfiable model
        model = cp.Model()
        x = cp.boolvar(2)
        model.maximize(cp.sum(x))
        # Add contradictory constraints
        model += x[0] == 1
        model += x[0] == 0
        
        # Solve with RC2
        rc2 = CPM_rc2(model)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved is False
        assert rc2.objective_value() is None
    
    def test_rc2_solve_negative_positive_combination(self, solve_kwargs):
        """Test RC2 solving with negative and positive coefficients in objective"""
        # Create model: m = cp.Model()
        m = cp.Model()
        x = cp.boolvar(2)
        m.maximize(-4*x[0] + 3*x[1])
        
        # Solve with RC2
        rc2 = CPM_rc2(m)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved is True
        assert rc2.objective_value() == 3
        assert list(x.value()) == [False, True]
    
    def test_rc2_atmostk(self, solve_kwargs):
        """Test RC2 solving with AtMostK"""
        m = cp.Model()
        x = cp.boolvar(5)
        m += cp.sum(x) <= 3
        m += cp.sum(x) >= 2
        m.minimize(cp.sum(x * [1,2,3,4,5]))
        
        # Solve with RC2
        rc2 = CPM_rc2(m)
        solved = rc2.solve(**solve_kwargs)
        
        assert solved is True
        assert rc2.objective_value() == 3
        assert list(x.value()) == [True, True, False, False, False]
    

if __name__ == '__main__':
    unittest.main()
