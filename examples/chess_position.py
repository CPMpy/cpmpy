from cpmpy import *
import numpy as np
from enum import Enum
from functools import total_ordering

data11 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

K = 1
Q = 2
R = 3
N = 4
B = 5
P = 6
E = 7
p = 8
b = 9
n = 10
r = 11
q = 12
k = 13


def fen_to_board(fen):
    board = []
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c == 'P':
                brow.append(P)
            elif c == 'p':
                brow.append(p)
            elif c == 'K':
                brow.append(K)
            elif c == 'k':
                brow.append(k)
            elif c == 'R':
                brow.append(R)
            elif c == 'r':
                brow.append(r)
            elif c == 'N':
                brow.append(N)
            elif c == 'n':
                brow.append(n)
            elif c == 'B':
                brow.append(B)
            elif c == 'b':
                brow.append(b)
            elif c == 'Q':
                brow.append(Q)
            elif c == 'q':
                brow.append(q)
            else:
                brow += [E] * int(c)
        board.append(brow)
    return np.array(board)

board = fen_to_board(data)

pieces = intvar(1, 13, (8, 8), "pieces")


#all the constraints that a chess board has to follow
model = Model(
    #both colors have only one king
    (sum(pieces[r,c] == k for r in range(0,8) for c in range(0,8)) == 1),
    (sum(pieces[r,c] == K for r in range(0,8) for c in range(0,8)) == 1),
    #both colors have maximum 16 pieces
    (sum(pieces[r,c] <= 6 for r in range(0,8) for c in range(0,8)) <= 16),
    (sum(pieces[r,c] >= 8 for r in range(0,8) for c in range(0,8)) <= 16),
    #both colors have maximum 8 pawns
    (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) <= 8),
    (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) <= 8),
    #pawns can't be on the edge of the board
    (sum(pieces[0, c] == p for c in range(0,8)) == 0),
    (sum(pieces[0, c] == P for c in range(0,8)) == 0),
    (sum(pieces[7, c] == p for c in range(0,8)) == 0),
    (sum(pieces[7, c] == P for c in range(0,8)) == 0),
    #the number of pieces of each piece without promotion
    (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == b for r in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == B for r in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[ro,c] == r for ro in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == R for r in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == n for r in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == N for r in range(0,8) for c in range(0,8)) <= 2),
    (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == q for r in range(0,8) for c in range(0,8)) <= 1),
    (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8).implies(sum(pieces[r,c] == Q for r in range(0,8) for c in range(0,8)) <= 1),
    #bishops can't have moved if the pawns are still in starting position
    ((pieces[1,1] == p) & (pieces[1,3] == p) & (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8) & (sum(pieces[r,c] == b for r in range(0,8) for c in range(0,8)) == 2)).implies(pieces[0,2]==b),
    ((pieces[6,1] == P) & (pieces[6,3] == P) & (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8) & (sum(pieces[r,c] == B for r in range(0,8) for c in range(0,8)) == 2)).implies(pieces[7,2]==B),
    ((pieces[1,4] == p) & (pieces[1,6] == p) & (sum(pieces[r,c] == b for r in range(0,8) for c in range(0,8)) == 2) & (sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) == 8)).implies(pieces[0,5]==b),
    ((pieces[6,4] == P) & (pieces[6,6] == P) & (sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) == 8) & (sum(pieces[r,c] == B for r in range(0,8) for c in range(0,8)) == 2)).implies(pieces[7,5]==B),
    #if bishop isn't able to get out and pawns are still in starting position, rook is stuck
    (((pieces[6,4] == P) & (pieces[6,6] == P) & (pieces[6,7] == P) & (pieces[7,5]==B)).implies(Xor([pieces[7,6]==R,pieces[7,7] == R]))),
    (((pieces[6,1] == P) & (pieces[6,3] == P) & (pieces[6,0] == P) & (pieces[7,2]==B)).implies(Xor([pieces[7,1]==R,pieces[7,0] == R]))),
    (((pieces[1,1] == p) & (pieces[1,3] == p) & (pieces[1,0] == p) & (pieces[0,2]==b)).implies(Xor([pieces[0,1]==r,pieces[0,0] == r]))),
    (((pieces[1,4] == p) & (pieces[1,6] == p) & (pieces[1,7] == p) & (pieces[0,5]==b)).implies(Xor([pieces[0,6]==r,pieces[0,7] == r]))),
    #if both bishops are stuck and pawns are in starting position, king and queen are also in starting position
    (((pieces[1,1] == p) & (pieces[1,2] == p) & (pieces[1,3] == p) & (pieces[1,4] == p) & (pieces[1,5] == p) & (pieces[1,6] == p)).implies((pieces[0,3]==q) | (pieces[0,4]==k))),
    (((pieces[6,1] == P) & (pieces[6,2] == P) & (pieces[6,3] == P) & (pieces[6,4] == P) & (pieces[6,5] == P) & (pieces[6,6] == P)).implies((pieces[7,3]==Q) | (pieces[7,4]==K))),
    #if the pawns in column 2 or 7 didn't move, bishop of own color can't be in the corners of the board
    (pieces[6,1] == P).implies(~((pieces[7,0] == B))),
    (pieces[6,6] == P).implies(~((pieces[7,7] == B))),
    (pieces[1,6] == p).implies(~((pieces[0,7] == b))),
    (pieces[1,1] == p).implies(~((pieces[0,0] == b))),
    #if the pawns in column 2 or 7 didn't move, bishop of other color can't be in the corners of the board excpet when promoted
    ((pieces[6,1] == P) & (pieces[7,0] == B)).implies(sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) <= 7),
    ((pieces[6,6] == P) & (pieces[7,7] == B)).implies(sum(pieces[r,c] == p for r in range(0,8) for c in range(0,8)) <= 7),
    ((pieces[1,6] == p) & (pieces[0,7] == b)).implies(sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) <= 7),
    ((pieces[1,1] == p) & (pieces[0,0] == b)).implies(sum(pieces[r,c] == P for r in range(0,8) for c in range(0,8)) <= 7),
)   

#fill the intvar with the board
for (r,c), val in np.ndenumerate(board):
    model += pieces[r,c] == val

#if there is a doubled pawn, the other team can't have all their pieces
for col in range(0,8):
    model += (sum(pieces[r, col] == p for r in range(0,8)) >= 2).implies(sum(pieces[r,c] <= 6 for r in range(0,8) for c in range(0,8)) <= (16-(sum(pieces[r, col] == p for r in range(0,8)))))
    model += (sum(pieces[r, col] == P for r in range(0,8)) >= 2).implies(sum(pieces[r,c] <= 6 for r in range(0,8) for c in range(0,8)) <= (16-(sum(pieces[r, col] == P for r in range(0,8))))),


if model.solve():
    print("We found models")
    print(pieces.value())
else:
    print("No solution found")