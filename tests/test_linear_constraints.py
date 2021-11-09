import unittest
import cpmpy as cp 
from cpmpy.expressions import *
from cpmpy.solvers.pysat import CPM_pysat
from cpmpy.transformations.to_cnf import to_cnf

class TestLinearConstraint(unittest.TestCase):
    def setUp(self):
        self.bv = boolvar(shape=5)


    # def test_pysat_simple_atmost(self):

    #     atmost = cp.Model(
    #         ## < This does not 
    #         2 * self.bv[0] < 3,
    #         # self.bv[0] *2 < 3,
    #         ## <=
    #         3 * self.bv[1] <= 3,
    #         # self.bv[1] *3 <= 3,
    #         ## >
    #         2 * self.bv[2] > 3,
    #         # self.bv[2] * 2 > 3,
    #         ## >=
    #         4 * self.bv[2] >= 3,
    #         # self.bv[3] * 4 >= 3
    #     )

    #     ps = CPM_pysat(atmost)
    #     ps.solve()
    #     print(self.bv.value())


    def test_pysat_boolean_linear_sum(self):
        n = 5
        color = intvar(1,n,shape=n,name="color")
        red,green,yellow,blue,ivory = color

        nationality = intvar(1,n,shape=n,name="nationality")
        englishman,spaniard,japanese,ukrainian,norwegian = nationality
        # This is used in the solution below.
        englishman.name="englishman"; spaniard.name="spaniard";
        japanese.name="japanese";ukrainian.name="ukrainan";
        norwegian.name="norwegian"

        animal = intvar(1,n,shape=n,name="animal")
        dog,snails,fox,zebra,horse = animal

        drink = intvar(1,n,shape=n,name="drink")
        tea,coffee,water,milk,fruit_juice = drink

        smoke = intvar(1,n,shape=n,name="smoke")
        old_gold,kools,chesterfields,lucky_strike,parliaments = smoke

        ls = cp.Model(
            # 2 * self.bv[0] + 3 * self.bv[1] <= 3,
            [1 == abs(norwegian - blue)]
        )
        ps = CPM_pysat(ls)


if __name__ == '__main__':
    unittest.main()

