"""
    Blocks world

    Given a k piles of blocks in a starting configuration, find the shortest sequence of block moves
        to arrive at at a given goal configuration.
    A block can only move to another block if it is on top of the pile.
    At most k piles can be used (some piles may be empty)

    Roughly a translation from the MiniZinc model:
    https://github.com/MiniZinc/mzn-challenge/blob/develop/2022/blocks-world/blocks.mzn


"""


def get_model(n, k, start, goal):
    import cpmpy as cp

    nCubes, horizon = n + 1, n * k + 1  # nCubes includes the dummy cube 0

    state = cp.intvar(0, n, shape=(nCubes, horizon), name="state")
    count = cp.intvar(0, k, shape=(nCubes, horizon), name="count")
    done = cp.boolvar(shape=horizon, name="done")
    move = cp.intvar(0, n, shape=(horizon, 2), name="move")
    locked = cp.boolvar(shape=(nCubes, horizon), name="locked")

    model = cp.Model()
    model.minimize(horizon - cp.sum(done))

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
        model += cp.GlobalCardinalityCount(state[1:, t], list(range(0, nCubes)), count[:, t])

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
        for b in range(nCubes):
            model += done[t-1] | ((state[b, t-1] != state[b,t]) == (b == move[t-1,0]))


    # redundant constraints, should up search
    # prevent do-undo moves
    for t in range(0, horizon - 1):
        model += (~done[t]).implies(cp.all([move[t, 0] != move[t + 1, 0],
                                            move[t, 1] != move[t + 1, 0],
                                            move[t, 1] != move[t + 1, 1],
                                            move[t, 0] != move[t, 1]]))

    # some cubes are locked in end position
    for t in range(horizon):
        for b in range(1, nCubes):
            model += locked[b, t] == ((state[b, t] == goal[b-1]) & (locked[goal[b-1], t] if goal[b-1] != 0 else True))

    # don't move locked blocks
    for t in range(horizon - 1):
        for b in range(1, nCubes):
            model += locked[b, t].implies(state[b, t + 1] == goal[b-1])
            model += locked[b, t].implies(move[t + 1, 0] != b)

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

    for state in states.T:
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

    model, (state, move) = get_model(n, k, start, end)

    if model.solve() is True:
        print(f"Found solution with {model.objective_value()} steps")
        show_solution(state.value())
    else:
        print("Model is UNSAT!")


