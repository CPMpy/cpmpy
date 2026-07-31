import cpmpy as cp
from cpmpy.tools.explain.mus import quickxplain_naive

if __name__ == "__main__":
    p = cp.boolvar(name="p")

    soft = [p, ]
    hard = [~p & p]

    mus = quickxplain_naive(soft=soft, hard=hard)
    print(mus)