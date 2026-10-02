import time

import pytest
import cpmpy as cp
from cpmpy.tools.local_search import large_neighborhood_search


class TestLNS:

    def test_maximize(self):
        x = cp.boolvar(shape=4)
        model = cp.Model(2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] <= 5,
                         maximize=3*x[0] + 4*x[1] + 5*x[2] + 6*x[3])
        for var in x:
            var._value = 0
        init_obj = model.objective_value()

        def destroy():
            x[0].clear()
            x[1].clear()

        large_neighborhood_search(model, destroy, max_iterations=5)

        assert all(var.value() is not None for var in x)
        assert (2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] <= 5).value() is True
        assert model.objective_value() > init_obj
        assert model.objective_value() == 3*x[0].value() + 4*x[1].value() + 5*x[2].value() + 6*x[3].value()

    def test_minimize(self):
        x = cp.boolvar(shape=4)
        model = cp.Model(2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] >= 5,
                         minimize=3*x[0] + 4*x[1] + 5*x[2] + 6*x[3])
        for var in x:
            var._value = 1
        init_obj = model.objective_value()

        def destroy():
            x[0].clear()
            x[1].clear()

        large_neighborhood_search(model, destroy, max_iterations=5)

        assert all(var.value() is not None for var in x)
        assert (2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] >= 5).value() is True
        assert model.objective_value() < init_obj
        assert model.objective_value() == 3*x[0].value() + 4*x[1].value() + 5*x[2].value() + 6*x[3].value()

    def test_non_incremental(self):
        x = cp.boolvar(shape=4)
        model = cp.Model(2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] <= 5,
                         maximize=3*x[0] + 4*x[1] + 5*x[2] + 6*x[3])
        for var in x:
            var._value = 0

        def destroy():
            x[0].clear()
            x[1].clear()

        large_neighborhood_search(model, destroy, max_iterations=5, incremental=False)

        assert all(var.value() is not None for var in x)
        assert (2*x[0] + 3*x[1] + 4*x[2] + 5*x[3] <= 5).value() is True
        assert model.objective_value() >= 0

    def test_time_limit(self):
        x = cp.boolvar(shape=8)
        model = cp.Model(cp.sum(x) <= 4, maximize=cp.sum(x))
        for var in x:
            var._value = 0

        def destroy():
            for var in x[:4]:
                var.clear()

        start = time.time()
        large_neighborhood_search(model, destroy, total_time_limit=0.5, incremental=False)
        elapsed = time.time() - start

        assert elapsed < 5
        assert all(var.value() is not None for var in x)
        assert (cp.sum(x) <= 4).value() is True
        assert model.objective_value() >= 0

    def test_invalid_args(self):
        x = cp.boolvar()
        sat = cp.Model(x)
        opt = cp.Model(x, maximize=x)

        def destroy():
            x.clear()

        with pytest.raises(ValueError):
            large_neighborhood_search(sat, destroy, max_iterations=1)
        with pytest.raises(ValueError):
            large_neighborhood_search(opt, destroy)
