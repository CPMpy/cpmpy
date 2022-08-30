"""
    Basketball schedule in CPMpy

    Problem 011 on CSPlib

    The problem is finding a timetable for the 1997/98 Atlantic Coast Conference (ACC) in basketball. It was first tackled by Nemhauser and Trick.

    Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""

from cpmpy import *
import numpy as np

def basketball_schedule():

    n_teams = 9
    n_days = 18


    # Teams
    teams = np.arange(n_teams)
    CLEM, DUKE, FSU, GT, UMD, UNC, NCSt, UVA, WAKE = teams
    rivals = [GT, UNC, FSU, CLEM, UVA, DUKE, WAKE, UMD, NCSt]

    # Days
    days = np.arange(n_days)
    weekdays = np.where(days % 2 == 0)[0]
    weekends = np.where(days % 2 == 1)[0]

    # matrix indicating which teams play against each other at what date
    # config[d,i] == j iff team i plays team j on day d of the tournament
    # config[d,i] == i iff team i plays bye on day d of the tournament
    config = intvar(0,n_teams-1, shape=(n_days, n_teams), name="config")
    # home[d,i] == True iff team i plays home on day d of the tournament
    where = intvar(0,2, shape=(n_days, n_teams), name="where")
    HOME, BYE, AWAY = 0,1,2




    model = Model()

    # a team cannot have different opponents on the same day
    for day_conf in config:
        model += AllDifferent(day_conf)

    # team plays itself when playing BYE
    for day in range(n_days):
        model += (config[day] == teams) == (where[day] == BYE)

    # symmetry
    for day in range(n_days):
        for t in range(n_teams):
            model += config[day,config[day,t]] == t


    # 1. mirroring constraint
    scheme = np.array([7, 8, 11, 12, 13, 14, 15, 0, 1, 16, 17, 2, 3, 4, 5, 6, 9, 10])
    model += config == config[scheme]
    model += where == (2 - where[scheme])

    # 2. no two final days away
    for t in range(n_teams):
        model += sum(where[-2:,t] == AWAY) <= 1

    # 3. home/away/bye pattern constraint
    for t in teams:
        for d in days[:-3]:
            # No team may have more than two home matches in a row
            model += sum(where[d:d+3,t] == HOME) <= 2
            # No team may have more than two away matches in a row
            model += sum(where[d:d+3,t] == AWAY) <= 2

        for d in days[:-4]:
            # No team may have more than three away matches or byes in a row
            model += sum((where[d:d+4,t] == AWAY) |
                         (where[d:d+4,t] == BYE)) <= 3

        for d in days[:-5]:
            # No team may have more than four home matches or byes in a row.
            model += sum((where[d:d+5,t] == HOME) |
                         (where[d:d+5,t] == BYE)) <= 4


    # 4. weekend pattern constraint
    # Of the weekends, each team plays four at home, four away, and one bye.
    for t in range(n_teams):
        model += sum(where[weekends, t] == HOME) == 4
        model += sum(where[weekends, t] == AWAY) == 4
        model += sum(where[weekends, t] == BYE) == 1

    # 5. first weekends constraint
    # Each team must have home matches or byes at least on two of the first five weekends.
    for t in range(n_teams):
        model += (sum(where[weekends[:5], t] == HOME) +
                  sum(where[weekends[:5], t] == BYE))  >= 2

    # 6. rival matches constraint
    # In the last date, every team except FSU plays against its rival, unless it plays against FSU or has a bye.
    for t in teams:
        if t != FSU:
            model += (config[-1,t] == rivals[t]) | \
                     (config[-1,t] == FSU) | \
                     (where[-1,t] == BYE)

    # 7. Constrained matches
    # The following pairings must occur at least once in dates 11 to 18:
    # Wake-UNC, Wake-Duke, GT-UNC, and GT-Duke.
    model += sum(config[10:,WAKE] == UNC) >= 1
    model += sum(config[10:,WAKE] == DUKE) >= 1
    model += sum(config[10:,GT] == UNC) >= 1
    model += sum(config[10:,GT] == DUKE) >= 1

    # 8. Opponent Sequence constraints
    for t in teams:
        for d in days[:-2]:
            if t != DUKE and t != UNC:
                # No team plays in two consecutive dates away against UNC and Duke
                model += ~((config[d, t] == UNC) & (where[d,t] == AWAY) &
                           (config[d+1, t] == DUKE) & (where[d+1,t] == AWAY))
                model += ~((config[d, t] == DUKE) & (where[d,t] == AWAY) &
                           (config[d+1, t] == UNC) & (where[d+1,t] == AWAY))
        for d in days[:-3]:
            if t not in [UNC, DUKE, WAKE]:
                # No team plays in three consecutive dates against UNC, Duke and Wake (independent of home/away).
                model += ~((config[d,t] == UNC)  & (config[d+1,t] == DUKE) & (config[d+2] == WAKE))
                model += ~((config[d,t] == UNC)  & (config[d+1,t] == WAKE) & (config[d+2] == DUKE))
                model += ~((config[d,t] == DUKE) & (config[d+1,t] == UNC)  & (config[d+2] == WAKE))
                model += ~((config[d,t] == DUKE) & (config[d+1,t] == WAKE) & (config[d+2] == UNC))
                model += ~((config[d,t] == WAKE) & (config[d+1,t] == UNC)  & (config[d+2] == DUKE))
                model += ~((config[d,t] == WAKE) & (config[d+1,t] == DUKE) & (config[d+2] == UNC))


    # 9. Other constraints
    # UNC plays its rival Duke in the last date and in date 11
    model += config[10, UNC] == DUKE
    model += config[-1, UNC] == DUKE
    # UNC plays Clem in the second date
    model += config[1, UNC] == CLEM
    # Duke has a bye in date 16
    model += where[15,DUKE] == BYE
    # Wake does not play home in date 17
    model += where[16,WAKE] != HOME
    # Wake has a bye in the first date
    model += where[0,WAKE] == BYE
    # Clem, Duke, UMD and Wake do not play away in the last date
    model += where[-1,[CLEM, DUKE, UMD, WAKE]] != AWAY
    # Clem, FSU, GT and Wake do not play away in the first date
    model += where[0, [CLEM, FSU, GT, WAKE]] != WAKE
    # Neither FSU nor NCSt have a bye in last date
    model += where[-1,[FSU, NCSt]] != BYE
    # UNC does not have a bye in the first date.
    model += where[0, UNC] != BYE

    return model, (config, where)

def print_solution(config, where):
    n_days = len(config)
    team_map = np.array(["CLEM", "DUKE", "FSU", "GT", "UMD", "UNC", "NCSt", "UVA", "WAKE"])
    where_map = np.array(["HOME","BYE","AWAY"])
    fmt = "{:<7}"*len(config[0])
    # print header
    print(" "*6, end="|\t")
    print(fmt.format(*team_map), end="\t|\t")
    print(fmt.format(*team_map))
    print("-"*137)
    for day in range(n_days):
        print("Day {:<2}".format(day), end="|\t")
        print(fmt.format(*team_map[config[day]]), end="\t|\t")
        print(fmt.format(*where_map[where[day]]))


if __name__ == "__main__":

    model, (config, where) = basketball_schedule()

    if model.solve():
        print_solution(config.value(), where.value())
    else:
        raise ValueError("Model is unsatisfiable")





