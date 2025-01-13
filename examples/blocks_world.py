"""
    Blocks world

    Given a k piles of blocks in a starting configuration, find the shortest sequence of block moves
        to arrive at at a given goal configuration.
    A block can only move to another block if it is on top of the pile.
    At most k piles can be used (some piles may be empty)

    This file showcases two ways of solving the problem.
    The first model is an optimization model with a heuristic upper bound on the number of moves as initial bound.
    The second appraoch is a "planning as SAT" approach where the model is solved iteratively with an increasing horizon

    Both models are inspired by the MiniZinc model:
        https://github.com/MiniZinc/mzn-challenge/blob/develop/2022/blocks-world/blocks.mzn
"""

def get_model(start, goal, n_piles=None):
    import cpmpy as cp

    n = len(start)
    if n_piles is None:
        n_piles = n # no limit on number of piles

    n_cubes, horizon = n + 1, n * n_piles + 1  # n_cubes includes the dummy cube 0

    # state[b,t] shows the block where block b is on at time t
    #   state[b,t] = 0 means the block is on the table at that  time
    state = cp.intvar(0, n, shape=(n_cubes, horizon), name="state")
    # count[b,t] counts the number of blocks on b at time t (always <= 1 except for table)
    count = cp.intvar(0, n_piles, shape=(n_cubes, horizon), name="count")
    # whether we are done at time step t
    done = cp.boolvar(shape=horizon, name="done")
    # shows the move at time t, moves are interleaved between states
    move = cp.intvar(0, n, shape=(horizon, 2), name="move")

    model = cp.Model()
    model.minimize(cp.sum(~done)) # minimize the time we are not done

    # don't move dummy block
    model += state[0, :] == 0

    # define start and end states
    model += state[1:, 0] == start
    model += state[1:, -1] == goal

    for t in range(horizon):
        model += done[t] == cp.all(state[1:,t] == goal)

    # define moves
    for t in range(1, horizon):
        model += move[t-1, 1] == state[move[t-1, 0], t]

    # computing the number of times a cube occurs in the configuration
    for t in range(horizon):
        model += cp.GlobalCardinalityCount(state[1:, t], list(range(0, n_cubes)), count[:, t], closed=True)

    # count nb of times a block occurs
    model += count[0, :] >= 1
    model += count[1:, :] <= 1

    # ensure we have two phases
    model += cp.Increasing(done)

    for t in range(1, horizon):
        # don't move if we are done
        model += ~done[t-1] | (cp.all(state[1:, t-1] == state[1:, t]))
        #no more moves once finished
        model += done[t-1] == (move[t-1, 0] == 0)
        # can only move if block is free
        model += done[t-1] | (count[move[t-1,0], t-1] == 0)
        # if the state change for a block, it is moved
        for b in range(n_cubes):
            model += done[t-1] | ((state[b, t-1] != state[b,t]) == (b == move[t-1,0]))


    # redundant constraints, should speed up search
    # prevent do-undo moves
    for t in range(0, horizon - 1):
        model += (~done[t]).implies(cp.all([move[t, 0] != move[t + 1, 0],
                                            move[t, 1] != move[t + 1, 0],
                                            move[t, 1] != move[t + 1, 1],
                                            move[t, 0] != move[t, 1]]))

    # some cubes are locked in end position
    # locked means the block and all blocks below are at their end position
    locked = cp.boolvar(shape=(n_cubes, horizon), name="locked")
    for t in range(horizon):
        for b in range(1, n_cubes):
            model += locked[b, t] == ((state[b, t] == goal[b-1]) & (locked[goal[b-1], t] if goal[b-1] != 0 else True))

    # don't move locked blocks
    for t in range(horizon - 1):
        for b in range(1, n_cubes):
            model += locked[b, t].implies(state[b, t + 1] == goal[b-1])
            model += locked[b, t].implies(move[t + 1, 0] != b)

    # objective lower bound
    # we will need at least as many time-steps as non-fixed blocks
    # seems to mostly help for Choco and Gecode
    for t in range(horizon):
        model += done[t] | (cp.sum(~locked[:, t]) + t <= cp.sum(~done))

    return model, (state, move)



# Planning as SAT model
def get_sat_model(start, goal, horizon, n_piles=None):
    import cpmpy as cp

    n = len(start)
    if n_piles is None:
        n_piles = n  # no limit on number of piles

    n_cubes = n + 1  # n_cubes includes the dummy cube 0
    horizon += 1

    #  state[b,t] shows the block where block b is on at time t
    #   state[b,t] = 0 means the block is on the table at that  time
    state = cp.intvar(0, n, shape=(n_cubes, horizon), name="state")
    # count[b,t] counts the number of blocks on b at time t (always <= 1 except for table)
    count = cp.intvar(0, n_piles, shape=(n_cubes, horizon), name="count")
    # shows the move at time t, moves are interleaved between states
    move = cp.intvar(0, n, shape=(horizon, 2), name="move")

    model = cp.Model()

    # don't move dummy block
    model += state[0, :] == 0

    # define start and end states
    model += state[1:, 0] == start
    model += state[1:, -1] == goal

    # define moves
    for t in range(1, horizon):
        model += move[t-1, 1] == state[move[t-1, 0], t]

    # computing the number of times a cube occurs in the configuration
    for t in range(horizon):
        model += cp.GlobalCardinalityCount(state[1:, t], list(range(0, n_cubes)), count[:, t], closed=True)

    # count nb of times a block occurs
    model += count[0, :] >= 1
    model += count[1:, :] <= 1

    for t in range(1, horizon):
        # can only move if block is free
        model += count[move[t-1,0], t-1] == 0
        # if the state change for a block, it is moved
        for b in range(n_cubes):
            model += (state[b, t-1] != state[b,t]) == (b == move[t-1,0])


    # redundant constraints, should up search
    # prevent do-undo moves
    for t in range(0, horizon - 1):
        model += (cp.all([move[t, 0] != move[t + 1, 0],
                          move[t, 1] != move[t + 1, 0],
                          move[t, 1] != move[t + 1, 1],
                          move[t, 0] != move[t, 1]]))

    # some cubes are locked in end position
    # locked means the block and all blocks below are at their end position
    locked = cp.boolvar(shape=(n_cubes, horizon), name="locked")
    for t in range(horizon):
        for b in range(1, n_cubes):
            model += locked[b, t] == ((state[b, t] == goal[b - 1]) & (locked[goal[b - 1], t] if goal[b - 1] != 0 else True))

    # don't move locked blocks
    for t in range(horizon - 1):
        for b in range(1, n_cubes):
            model += locked[b, t].implies(state[b, t + 1] == goal[b - 1])
            model += locked[b, t].implies(move[t + 1, 0] != b)

    # objective lower bound
    # we will need at least as many time-steps as non-fixed blocks
    # seems to mostly help for Choco and Gecode
    for t in range(horizon):
        model += cp.sum(~locked[:, t]) + t <= horizon

    return model, (state, move)

#### stuff for plotting

def _get_piles(state):
    state = [x - 1 for x in state]
    towers = []
    for next_idx in range(len(state)):
        if next_idx in state: continue
        tower = [next_idx + 1]
        while next_idx >= 0:
            x = state[next_idx]
            tower.append(int(x + 1))
            next_idx = x
        towers.append(tower[:-1])
    return sorted(towers, key=lambda x: x[-1])

def show_solution(states):
    # Find where the states start repeating by checking from the end
    states = states.T
    last_state = states[-1]
    end_idx = len(states)-1
    while end_idx > 0 and (states[end_idx-1] == last_state).all():
        end_idx -= 1
    
    # Print states up to where they start repeating
    for state in states[:end_idx+1]:
        piles = _get_piles(state[1:])
        for h in range(max(len(pile) for pile in piles)-1,-1,-1):
            for pile in piles:
                pile = list(reversed(pile))
                if len(pile) > h:
                    print(int(pile[h]), end='\t')
                else:
                    print(" ", end="\t")
            print()
        print("----------")

if __name__ == "__main__":

    n = 9
    k = 3
    start = [2, 0, 9, 6, 3, 1, 5, 4, 0]
    end = [3, 8, 4, 5, 0, 9, 0, 0, 1]

    # optimization model
    model, (state, move) = get_model(start, end, n_piles=k)
    assert model.solve(solver="ortools") is True
    n_steps = model.objective_value()
    print(f"Found optimal solution with {n_steps} steps in {round(model.status().runtime,3)}s")

    # iterative planning with increasing horizon
    horizon = sum(x != y for x,y in zip(start, end)) # count difference between start and end state
    total_time = 0
    while 1:
        model, (state, move) = get_sat_model(start, end, horizon, n_piles=k)
        has_sol = model.solve()
        runtime = model.status().runtime
        total_time += runtime
        if has_sol:
            print(f"Found optimal  solution with {horizon} steps ({round(runtime,3)}s)")
            print("Total solving time:", total_time)
            break
        else:
            print(f"No solution with {horizon} steps ({round(runtime,3)}s)")
            horizon += 1

    show_solution(state.value())
