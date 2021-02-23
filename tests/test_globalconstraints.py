from cppy.model import Model
from cppy import IntVar
from cppy.globalconstraints import alldifferent


def test_alldifferent():
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
        _ = model.solve()
        vals = [x.value() for x in vars]

        # ensure all different values
        assert len(vals) == len(set(vals))

