import unittest
import cpmpy as cp
from cpmpy.model_tools import flatten_model

class TestFlattenModel(unittest.TestCase):
    def setUp(self):
        self.ivars = cp.IntVar(1, 10, (5,))
        self.bvars = cp.BoolVar((2,))
        self.constraints = [self.ivars != 3]
        #self.model = cp.Model()

    def test_constraints(self):
        model = cp.Model(self.constraints)
        model2 = flatten_model(model)
        self.assertTrue(isinstance(model2, cp.Model))
        self.assertTrue(hasattr(model2, 'constraints'))
        self.assertTrue(len(model2.constraints) > 1)

    def test_objective(self):
        obj = self.ivars.sum()
        model = cp.Model(self.constraints, maximize=obj)
        model2 = flatten_model(model)
        self.assertTrue(model2.objective is not None)
        self.assertTrue(model2.objective_max)


class TestFlattenConstraint(unittest.TestCase):
    def setUp(self):
        a,b,c,d,e =  IntVar(1, 10, shape=(5,))
        f,g,h = BoolVar((3,))
        self.C = [
            a == b,
            h != f,
            (a > 5) == (b < 3),
            ((c % 2) == 0) == (c > 3),
            ((c / 2) < 2  ) == ( (a == 2) | (b == 3)),
            ((a != 0) | (b != 0)) == (c == 1),
            (((a + c == 5 | e > 3)  &  (b > 2) ) == (d < 8) )
        ]
        self.ivars = [a,b,c,d,e]
        self.bvars = [f,g,h]

    def test_base_constraint(self):
        #TODO: very basic
        model = cp.Model(self.c[:2])
        model2 = flatten_model(model)
        self.assertEqual(len(model2.constraints), 2)
    
    def test_flatten_reification(self):
        #TODO more complex test
        model = cp.Model(self.c[1:3])
        model2 = flatten_model(model)
        self.assertGreater(len(model2.constraints), len(model.constraints))

    