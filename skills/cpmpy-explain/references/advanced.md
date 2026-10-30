# Explanation use cases beyond "why UNSAT?"

Most explanation questions can be reduced to *"explain why this set of constraints is UNSAT"*: add the right extra constraints, then call a MUS/MCS tool. This page lists those reductions, gives a compact pattern for each, and links a reference implementation.

## Contents

- [Overview: question → reduction → tool](#overview-question--reduction--tool)
- [Step-wise explanations (hints) for a satisfiable problem](#step-wise-explanations-hints-for-a-satisfiable-problem)
- [Why is there no better solution?](#why-is-there-no-better-solution)
- [Why X and not Y?](#why-x-and-not-y)
- [Repairing an UNSAT model interactively](#repairing-an-unsat-model-interactively)
- [Relaxing instead of removing: slack](#relaxing-instead-of-removing-slack)
- [Counterfactuals on the objective: inverse optimisation](#counterfactuals-on-the-objective-inverse-optimisation)
- [Combining with machine learning: visual sudoku](#combining-with-machine-learning-visual-sudoku)
- [Reference material](#reference-material)

## Overview: question → reduction → tool

| User question | Reduce to | CPMpy tool | Reference example |
|---|---|---|---|
| "What is the next easiest step / give me a hint" | `rules + known facts + ¬(one new fact)` is UNSAT | `ocus(..., meta_constraint=sum(negs)==1)` | [ACP26 notebook §3a][acp26], [IJCAI22 sudoku][ijcai], [`ocus_explanations.py`][ocus-ex] |
| "Explain the whole solution, step by step" | Repeat the hint, add each derived fact | loop over `ocus` / `mus` | [`explain_stepwise_csp.ipynb`][stepwise-nb], [IJCAI22 sudoku][ijcai] |
| "Explain this UNSAT proof step by step" | Step-wise sequence over a MUS | `mus` + XCP `stepwise.find_sequence` | [ECAI24 notebook][ecai], [XCP `explanations/stepwise`][xcp-stepwise] |
| "Why can't the objective be better?" | `constraints + (objective < opt)` is UNSAT | `mus`, `mcs` | [ACP26 §3b][acp26] |
| "Why is X true in the solution, not Y?" | `constraints + Y` is UNSAT, or re-solve with `Y` and compare | `mus` / re-solve | [ACP26 §4b, §4c][acp26] |
| "Help me fix my infeasible model" | Iterate: MUS → user drops one → repeat | `mus` in a loop, `mcs`, `mcs_opt` | XCP [`diagnosis.py`][xcp-diag] |
| "Fix it with as few/small changes as possible" | Add slack variables, minimise them | plain CPMpy model | [ACP26 §4a][acp26], [ECAI24 part 2][ecai2] |
| "What weights would make my preferred solution optimal?" | Inverse optimisation | XCP [`counterfactual.py`][xcp-cf], [`counterfactual_explain.py`][cf-ex] | [ACP26 §4c][acp26] |
| "Read and solve a sudoku from a photo, with hints" | Max-likelihood COP over a CNN's outputs + step-wise hints | `Model.maximize` + `ocus` | [`visual_sudoku.ipynb`][vsudoku], [Sudoku assistant][vs-site] |

## Step-wise explanations (hints) for a satisfiable problem

A step explains one new fact `n` from a few constraints `S` and known facts `E`, so that `S ∧ E → n`. That is the same as saying `S ∧ E ∧ ¬n` is UNSAT. Here `¬n` is the negation of one fact still to be explained. To pick the *easiest* step, give each constraint and fact a cost and ask OCUS for the cheapest unsatisfiable subset that contains **exactly one** negated target. This is the method behind the hint button of the [Sudoku assistant][vs-site] (Bogaerts et al., AIJ 2021; Gamba et al., IJCAI 2021).

```python
import numpy as np
import cpmpy as cp
from cpmpy.tools.explain import ocus

def explain_steps(rules, facts, targets, rule_cost=10, fact_cost=1):
    """Yield (facts_used, rules_used, new_fact), easiest step first.
    `targets` must be implied by rules + facts (e.g. the cells of a unique solution)."""
    facts, targets = list(facts), list(targets)
    while targets:
        neg = {~t: t for t in targets}           # negated target -> target
        soft = rules + facts + list(neg)
        weights = [rule_cost] * len(rules) + [fact_cost] * len(facts) + [0] * len(neg)
        step = ocus(soft, weights=weights, meta_constraint=cp.sum(list(neg)) == 1)
        new = next(neg[c] for c in step if c in neg)
        yield ([c for c in step if c in facts], [c for c in step if c in rules], new)
        facts.append(new)
        targets.remove(new)

grid = np.array([[0, 3, 4, 0], [4, 0, 0, 2], [1, 0, 0, 3], [0, 2, 1, 0]])
cells = cp.intvar(1, 4, shape=grid.shape, name="cell")
rules = [cp.AllDifferent(r) for r in cells] + [cp.AllDifferent(c) for c in cells.T]
rules += [cp.AllDifferent(cells[i:i+2, j:j+2]) for i in (0, 2) for j in (0, 2)]
givens = [cells[i, j] == v for (i, j), v in np.ndenumerate(grid) if v]

assert cp.Model(rules + givens).solve()
targets = [cells[i, j] == cells[i, j].value() for (i, j), v in np.ndenumerate(grid) if not v]

for used_facts, used_rules, new in explain_steps(rules, givens, targets):
    print(new, "because", used_facts, "and", used_rules)
# e.g. cell[0,0] == 2 because [cell[0,1] == 3, cell[1,3] == 2]
#      and [alldifferent(cell[1,0],...,cell[1,3]), alldifferent(cell[0,0],cell[0,1],cell[1,0],cell[1,1])]
# (step order may vary between runs: several steps can have the same cost)
```

Design choices:

- **Targets** must be facts that *follow* from the problem. For a puzzle with a unique solution these are the solution's assignments. Otherwise, use the facts shared by all solutions (`solution_intersection` in [`explain_stepwise_csp.ipynb`][stepwise-nb]).
- **The cost function defines "easy".** Constraint cost ≫ fact cost prefers steps that use few rules. Give global constraints, or ones that are hard to understand, a higher cost. The Sudoku assistant counts the constraints used.
- **Granularity**: targets can be positive assignments (`x == v`, used in the sudoku examples) or negative ones (`x != v`, for "pencil-mark" reasoning). Use `cp.all(...)` to group constraints that the user sees as one rule.
- **Show a hint without the answer**: highlight `used_facts` and `used_rules` and hide `new` (the Sudoku assistant's UX).
- **Lighter variant**: to pick a step without full optimality, compute `mus(rules + facts, hard=[~t])` for each target `t` and keep the cheapest. This is not cost-optimal, but each step is cheaper to compute.

Scaling. Each step is an implicit-hitting-set loop, so a full sequence is expensive:

| Instance | Approach | Time measured (ortools hitting set) |
|---|---|---|
| 4x4 sudoku | `explain_steps` above | 1–8 s per step |
| 9x9 sudoku | `explain_steps` above | > 5 min for the first step |
| 9x9 sudoku | cheapest per-target `mus`, `solver="pindakaas"` | ~17 s per step |

To speed things up:
- use an incremental hitting-set solver (`hs_solver="gurobi"` or `"exact"`);
- use an incremental SAT-based `solver` (`"pindakaas"`, `"exact"`, `"pysat"`), which is roughly 10x faster than ortools per MUS here. Pass `do_solution_hint=False` to `ocus` for solvers without `solution_hint`;
- put facts that can't be part of the step in `hard`, or drop them;
- keep one hitting-set solver per target alive between calls, and always advance the most promising one (`split_ocus` in the [IJCAI22 notebook][ijcai]);
- for large problems, use the propagation-based sequence construction in [XCP `explanations/stepwise`][xcp-stepwise] (Bleukx et al., CP 2023).

## Why is there no better solution?

Turn optimality into infeasibility: forbid any better objective value, then explain that UNSAT problem.

```python
import cpmpy as cp
from cpmpy.tools.explain import mus, mcs

edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, 3), (3, 4)]
x = cp.intvar(1, 5, shape=5, name="x")
m = cp.Model([x[i] != x[j] for i, j in edges], minimize=cp.max(x))
m.solve()
better = [m.objective_ < m.objective_value()]

print(mus(m.constraints, hard=better))   # the K4 clique (0,1,2,3) forces 4 colours
print(mcs(m.constraints, hard=better))   # drop one of its edges to allow 3 colours
```

For realistic models, MUSes of optimality tend to be large. Group constraints (`cp.all`) per user-level concept before explaining.

## Why X and not Y?

The user proposes a foil `Y`, e.g. "why isn't node 4 the same colour as node 3?":

- **Deductive**: `mus(model.constraints, hard=[Y])` returns the constraints that rule `Y` out.
- **Counterfactual**: re-solve with `Y` added (`m2 = model.copy(); m2 += Y; m2.solve()`), then show how the solution and objective change. The XCP notebooks highlight the changed cells (`highlight_changes`).

## Repairing an UNSAT model interactively

One MUS is one conflict. Fixing it may reveal the next one. An interactive repair loop:

```python
import cpmpy as cp
from cpmpy.tools.explain import mus

def diagnose(soft, hard=[], choose=lambda conflict: conflict[0]):
    soft, removed = list(soft), []
    while not cp.Model(hard + soft).solve():
        conflict = mus(soft, hard)
        c = choose(conflict)                 # ask the user, or use a heuristic
        soft.remove(c)
        removed.append(c)
    return removed
```

An assumption-based, incremental version with user input is in XCP [`diagnosis.py`][xcp-diag]; it also has `fastdiag`, the MCS counterpart of QuickXplain. When the user doesn't need to choose, `mcs` / `mcs_opt(weights=priority)` gives the cheapest repair in one call.

## Relaxing instead of removing: slack

For numeric constraints (coverage, capacity), removing a constraint is too coarse. Add a slack variable to each constraint and minimise how much the constraints are violated:

```python
under = cp.intvar(0, UB, shape=n, name="under")
over = cp.intvar(0, UB, shape=n, name="over")
model += [cp.sum(assigned[d]) + under[d] - over[d] == required[d] for d in range(n)]

slack = cp.cpm_array(list(under) + list(over))
model.minimize(cp.sum(slack))                          # total violation
model.minimize(cp.sum(slack != 0))                     # number of violated constraints
model.minimize(10000 * cp.max(slack) + 1000 * cp.sum(slack != 0) + cp.sum(slack))  # lexicographic
```

Full nurse rostering version: `NurseSchedulingFactory.get_slack_model` in [XCP `factory.py`][xcp], used in [ACP26 §4a][acp26].

## Counterfactuals on the objective: inverse optimisation

"What objective weights would have to change so that my preferred assignment becomes optimal?" This is solved by alternating a master problem (new weights, close to the old ones) with a sub-problem (re-optimise under those weights) (Korikov & Beck, CP 2021).

- Knapsack example: [`examples/advanced/counterfactual_explain.py`][cf-ex]
- Generic `inverse_optimize(model, user_sol, allowed_to_change)`: XCP [`counterfactual.py`][xcp-cf], used in [ACP26 §4c][acp26].

## Combining with machine learning: visual sudoku

The [Sudoku assistant][vs-site] reads a photographed puzzle and then gives hints:

1. A CNN outputs, for each cell, a probability for each digit 1–9 and for "empty".
2. **Joint inference**: instead of taking the argmax per cell, solve a COP that maximises the log-likelihood of the chosen digits subject to the sudoku rules. This corrects misread digits that would make the puzzle UNSAT or ambiguous (see [`visual_sudoku.ipynb`][vsudoku], Mulamba et al., CPAIOR 2020).
3. **Hints**: [step-wise explanations](#step-wise-explanations-hints-for-a-satisfiable-problem) on the recognised grid, with the cost = number of constraints used.

## Reference material

- [XCP-explain][xcp]: explainable constraint solving tutorial repo (Bleukx, Tsouros, Guns). Nurse rostering running example, with:
  - [ACP 2026 notebook][acp26]: the most recent; covers all of the above;
  - [ECAI 2024 notebook][ecai] and practice parts [1][ecai1], [2][ecai2], [3][ecai3] (Colab-ready);
  - `explanations/`: less mature algorithms not (yet) in `cpmpy.tools.explain`: step-wise sequences, `diagnose`, `fastdiag`, inverse optimisation, proof-log based cores with Pumpkin.
- CPMpy examples: [`ocus_explanations.py`][ocus-ex], [`explain_stepwise_csp.ipynb`][stepwise-nb], [`counterfactual_explain.py`][cf-ex], [`visual_sudoku.ipynb`][vsudoku], [IJCAI22 tutorial: explain sudoku][ijcai]
- Papers:
  - Bogaerts, Gamba, Guns. *A framework for step-wise explaining how to solve constraint satisfaction problems.* AIJ 2021.
  - Gamba, Bogaerts, Guns. *Efficiently explaining CSPs with unsatisfiable subset optimization.* IJCAI 2021 / JAIR 2023.
  - Bleukx, Devriendt, Gamba, Bogaerts, Guns. *Simplifying step-wise explanation sequences.* CP 2023.
  - Korikov, Beck. *Counterfactual explanations via inverse constraint programming.* CP 2021.
  - Mulamba, Mandi, Canoy, Guns. *Hybrid classification and reasoning for image-based constraint solving.* CPAIOR 2020.

[xcp]: https://github.com/CPMpy/XCP-explain
[acp26]: https://github.com/CPMpy/XCP-explain/blob/main/acp26_presentation.ipynb
[ecai]: https://github.com/CPMpy/XCP-explain/blob/main/ecai2024_presentation.ipynb
[ecai1]: https://colab.research.google.com/github/CPMpy/XCP-explain/blob/main/ecai2024_practice_part1.ipynb
[ecai2]: https://colab.research.google.com/github/CPMpy/XCP-explain/blob/main/ecai2024_practice_part2.ipynb
[ecai3]: https://colab.research.google.com/github/CPMpy/XCP-explain/blob/main/ecai2024_practice_part3.ipynb
[xcp-stepwise]: https://github.com/CPMpy/XCP-explain/tree/main/explanations/stepwise
[xcp-diag]: https://github.com/CPMpy/XCP-explain/blob/main/explanations/diagnosis.py
[xcp-cf]: https://github.com/CPMpy/XCP-explain/blob/main/explanations/counterfactual.py
[ocus-ex]: https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/ocus_explanations.py
[stepwise-nb]: https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/explain_stepwise_csp.ipynb
[cf-ex]: https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/counterfactual_explain.py
[vsudoku]: https://github.com/CPMpy/cpmpy/blob/master/examples/advanced/visual_sudoku.ipynb
[ijcai]: https://github.com/CPMpy/cpmpy/blob/master/examples/tutorial_ijcai22/6_explain_sudoku.ipynb
[vs-site]: https://visualsudoku.cs.kuleuven.be/behind-the-scenes.html
