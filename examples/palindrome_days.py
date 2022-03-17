# how many 'palindrome days' are there in a century?
# for example, today: 121121 (12 november 2021)
# Thanks for asking Michiel : )

from cpmpy import *

v = intvar(0,9, shape=6)
day = 10*v[0] + v[1]
month = 10*v[2] + v[3]
year = 10*v[4] + v[5]
# for the American version:
#month = 10*v[0] + v[1]
#day = 10*v[2] + v[3]

m = Model(
    day >= 1, day <= 31,
    month >= 1, month <= 12,
    year >= 0, year <= 99,
    v[0] == v[-1],
    v[1] == v[-2],
    v[2] == v[-3],
    (month == 2).implies(day <= 28) # february
)
for no31 in [2,4,6,9,11]:
    m += [(month == no31).implies(day<=30)]

c = 0
while m.solve():
    c += 1
    print(c, v.value())
    m += ~all(v == v.value())
