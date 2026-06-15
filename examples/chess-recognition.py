#!/usr/bin/env python
"""
Example of evaluating a chess position in CPMpy

Evaluating a chess position by getting the probabilities of which piece it is and the places the pieces are placed
"""
import time

from cpmpy import *
import numpy as np
from enum import Enum
from functools import total_ordering


# Replace numerals by letters for easier interpretation
def replace_value(x):
    return dict.get(x, x)


# calculate the softmax of a vector
def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


piece_probabilities = np.array([[-1.1558e+01, -1.8940e-01, -1.0785e+01, -8.4097e+00, -1.3034e+01, -8.7986e+00,
                                 -2.6694e+00, 1.2009e+01, -3.5726e+00, -9.1412e-01, -2.0600e+00, -5.9005e-01],
                                [-1.4162e+01, -1.7926e+01, -1.4494e+01, -2.1050e+00, -1.6337e+01, -1.7570e+01,
                                 -5.1813e+00, -1.4064e+01, -5.9986e+00, 1.4458e+01, -9.9950e+00, -1.4928e+01],
                                [-1.2823e+01, -1.6611e+01, -1.3248e+01, -2.6784e+00, -1.3868e+01, -1.7606e+01,
                                 -4.4053e+00, -1.1934e+01, -5.1489e+00, 1.2227e+01, -7.1553e+00, -1.3355e+01],
                                [-1.5258e+00, -1.4117e+01, -1.5366e+01, -9.6915e+00, -1.2290e+01, -1.5584e+01,
                                 1.5208e+01, -4.8226e+00, -6.0717e+00, 1.6145e-02, -4.0168e+00, -6.2375e+00],
                                [-1.2323e+01, -1.5209e+01, -1.1199e+01, -2.0871e+00, -1.3257e+01, -1.5116e+01,
                                 -4.8906e+00, -1.1009e+01, -2.9934e+00, 1.1106e+01, -7.4524e+00, -1.1324e+01],
                                [-1.0075e+01, -1.2352e+01, -5.5208e+00, 1.5060e+01, -1.4162e+01, -1.3731e+01,
                                 -1.8131e+01, -2.2809e+01, -2.0523e+01, -9.2885e+00, -2.2656e+01, -2.3837e+01],
                                [-1.1237e+00, -1.5327e+01, -1.6454e+01, -1.0157e+01, -1.2902e+01, -1.6999e+01,
                                 1.6460e+01, -5.9819e+00, -6.6868e+00, 3.4977e-01, -4.5292e+00, -7.6471e+00],
                                [-5.8989e+00, -8.2629e+00, -6.0951e+00, -2.0709e+00, -6.8444e+00, -6.9492e+00,
                                 -1.4980e+00, -6.4349e+00, -2.8802e+00, 6.5927e+00, -5.2023e+00, -5.9768e+00],
                                [-1.5690e+01, -1.9345e+01, -1.4696e+01, -9.7584e-01, -1.6753e+01, -1.8871e+01,
                                 -7.1721e+00, -1.5371e+01, -5.9698e+00, 1.4855e+01, -1.0443e+01, -1.5127e+01],
                                [-1.5975e+01, -1.9709e+01, -1.6952e+01, -1.3250e+00, -1.7329e+01, -1.8364e+01,
                                 -6.4735e+00, -1.5004e+01, -7.9697e+00, 1.5087e+01, -1.0322e+01, -1.4676e+01],
                                [-9.6248e+00, -1.3684e+01, -8.5521e+00, 1.4719e+01, -1.1945e+01, -1.3185e+01,
                                 -1.6002e+01, -2.2853e+01, -1.9399e+01, -6.8938e+00, -1.9636e+01, -2.1333e+01],
                                [-2.8619e+00, -5.1141e+00, 1.3954e+01, 9.9080e-01, -5.9906e+00, -2.2784e+00,
                                 -1.1103e+01, -1.2708e+01, -8.1810e-01, -6.1267e+00, -1.4330e+01, -1.3472e+01],
                                [-1.1058e+01, -1.4450e+01, -1.0945e+01, -1.8521e+00, -1.2169e+01, -1.3680e+01,
                                 -4.4671e+00, -1.1790e+01, -5.1938e+00, 1.1196e+01, -8.3586e+00, -1.1404e+01],
                                [-7.1482e+00, -1.1688e+01, -6.5647e+00, 1.1144e+01, -9.7372e+00, -1.1413e+01,
                                 -1.0610e+01, -1.6807e+01, -1.3919e+01, -5.2683e+00, -1.4328e+01, -1.5948e+01],
                                [-3.4779e+00, -6.1333e+00, -1.9338e+00, 7.6642e+00, -5.6737e+00, -4.8410e+00,
                                 -8.8620e+00, -1.1995e+01, -8.9921e+00, -4.4965e+00, -1.1039e+01, -1.0286e+01],
                                [-5.4471e+00, -7.3961e+00, -4.0091e+00, 1.0008e+01, -8.2364e+00, -7.7552e+00,
                                 -1.1562e+01, -1.4871e+01, -1.2803e+01, -4.9956e+00, -1.4799e+01, -1.4024e+01],
                                [-1.0975e+01, -1.4695e+01, -7.0197e+00, 1.6814e+01, -1.4073e+01, -1.4442e+01,
                                 -1.8432e+01, -2.4743e+01, -2.0148e+01, -6.8818e+00, -2.2039e+01, -2.4336e+01],
                                [-1.8632e+00, 1.4883e+00, -2.7764e+00, -1.0044e+00, 1.1251e+01, -2.1246e+00,
                                 -9.2420e+00, -1.0971e+01, -8.4337e+00, -4.7349e+00, -1.9218e+00, -9.4412e+00],
                                [-5.1858e+00, 1.4478e+01, -6.0291e+00, -2.7641e+00, -3.3264e+00, -3.5799e+00,
                                 -1.1613e+01, 6.5402e-01, -1.1856e+01, -6.3279e+00, -1.2178e+01, -1.0819e+01],
                                [-8.3045e+00, -1.1004e+01, -5.4839e+00, 1.2478e+01, -1.1314e+01, -1.0851e+01,
                                 -1.3917e+01, -1.9016e+01, -1.5132e+01, -3.3467e+00, -1.7661e+01, -1.8405e+01],
                                [-1.0578e+01, -9.6113e+00, -1.0476e+01, -6.9775e+00, -3.4868e+00, -9.2788e+00,
                                 2.2917e-01, 1.6189e+00, -1.9423e+00, 9.8658e-01, 8.7442e+00, 1.9664e+00],
                                [1.7821e+00, 2.5520e+00, 2.3929e+00, -3.7030e-01, -3.3090e-01, 1.9798e-01, -6.8749e+00,
                                 -6.4092e+00, -5.7582e+00, -5.6637e+00, -8.7920e+00, -7.5211e+00]])

# make them actual probabilities
for i in range(len(piece_probabilities)):
    piece_probabilities[i] = softmax(piece_probabilities[i])

occupancy_in_board = np.array([[False, False, False, False, False, False, True, False],
                               [True, False, False, False, False, False, False, False],
                               [False, True, False, False, True, True, False, False],
                               [False, True, True, False, True, False, True, True],
                               [True, False, True, True, True, False, False, False],
                               [False, False, False, True, False, True, False, True],
                               [False, False, True, False, True, False, True, False],
                               [True, False, False, False, True, False, False, False]])

# Variable to represent the whole board
board = intvar(0, 12, (8, 8), "board")

# take the log of the NN output values
logprobs = np.log(piece_probabilities)

# Convert log probabilities to ints using a desired precision
PRECISION = 1e-4
lprobs = -logprobs / PRECISION

# convert to cpm_array to be able to use it in the objective function
lprobs = cpm_array(np.int64(lprobs))

# Assign values to pieces
b, k, n, p, q, r = 0, 1, 2, 3, 4, 5  # Black pieces
B, K, N, P, Q, R = 6, 7, 8, 9, 10, 11  # White pieces
E = 12  # Empty square

# Lowercases are black pieces, uppercases white pieces, and E is empty square
dict = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R', 12: 'E'}

# The constraints that a chess board has to follow
model = Model(
    # Both colors have only one king
    Count(board, k) == 1,
    Count(board, K) == 1,
    # Both colors have maximum 16 pieces
    (sum(board[l, c] < 6 for l in range(0, 8) for c in range(0, 8)) <= 16),
    (sum(((board[l, c] >= 6) & (board[l, c] < 12)) for l in range(0, 8) for c in range(0, 8)) <= 16),
    # Both colors have maximum 8 pawns
    Count(board, p) <= 8,
    Count(board, P) <= 8,
    # Pawns can't be on the edge of the board
    (Count(board[0], p) == 0),
    (Count(board[0], P) == 0),
    (Count(board[7], p) == 0),
    (Count(board[7], P) == 0),
    # The number of pieces can't exceed the starting number plus the number of promotions possible
    (
        (Count(board, b) + Count(board, r) + Count(board, n) + Count(board, q) <= 2+2+1+1 + 8-Count(board, p))
    ),
    (
        (Count(board, B) + Count(board, R) + Count(board, N) + Count(board, Q) <= 2+2+1+1 + 8-Count(board, P))
    ),
    # Bishops can't have moved if the pawns are still in starting position
    ((board[1, 1] == p) & (board[1, 3] == p) & (
        Count(board, p) == 8) & (Count(board, b) == 2)
     ).implies(board[0, 2] == b),
    ((board[6, 1] == P) & (board[6, 3] == P) & (
        Count(board, P) == 8) & (Count(board, B) == 2)
     ).implies(board[7, 2] == B),
    ((board[1, 4] == p) & (board[1, 6] == p) & (
        Count(board, b) == 2) & (Count(board, p) == 8)
     ).implies(board[0, 5] == b),
    ((board[6, 4] == P) & (board[6, 6] == P) & (
        Count(board, P) == 8) & (Count(board, B) == 2)
     ).implies(board[7, 5] == B),
    # If bishop isn't able to get out and pawns are still in starting position, rook is stuck
    (((board[6, 4] == P) & (board[6, 6] == P) & (board[6, 7] == P) & (board[7, 5] == B) & (
        Count(board, P) == 8) & (Count(board, R) == 2) ).implies(
        Xor([board[7, 6] == R, board[7, 7] == R]))),
    (((board[6, 1] == P) & (board[6, 3] == P) & (board[6, 0] == P) & (board[7, 2] == B) & (
        Count(board, P) == 8) & (Count(board, R) == 2) ).implies(
        Xor([board[7, 1] == R, board[7, 0] == R]))),
    (((board[1, 1] == p) & (board[1, 3] == p) & (board[1, 0] == p) & (board[0, 2] == b) & (
        Count(board, p) == 8) & (Count(board, r) == 2) ).implies(
        Xor([board[0, 1] == r, board[0, 0] == r]))),
    (((board[1, 4] == p) & (board[1, 6] == p) & (board[1, 7] == p) & (board[0, 5] == b) & (
        Count(board, p) == 8) & (Count(board, r) == 2) ).implies(
        Xor([board[0, 6] == r, board[0, 7] == r]))),
    # If both bishops are stuck and pawns are in starting position, king and queen are also in starting position
    (((board[1, 1] == p) & (board[1, 2] == p) & (board[1, 3] == p) & (board[1, 4] == p) & (board[1, 5] == p) & (
            board[1, 6] == p)).implies((board[0, 3] == q) & (board[0, 4] == k))),
    (((board[6, 1] == P) & (board[6, 2] == P) & (board[6, 3] == P) & (board[6, 4] == P) & (board[6, 5] == P) & (
            board[6, 6] == P)).implies((board[7, 3] == Q) & (board[7, 4] == K))),
    # If the pawns in column 2 or 7 didn't move, bishop of own color can't be in the corners of the board
    (board[6, 1] == P).implies(~((board[7, 0] == B))),
    (board[6, 6] == P).implies(~((board[7, 7] == B))),
    (board[1, 6] == p).implies(~((board[0, 7] == b))),
    (board[1, 1] == p).implies(~((board[0, 0] == b))),
    # If the pawns in column 2 or 7 didn't move, bishop of other color can't be in the corners of the board
    # except when promoted
    ((board[6, 1] == P) & (board[7, 0] == B)).implies(Count(board, p) <= 7),
    ((board[6, 6] == P) & (board[7, 7] == B)).implies(Count(board, p) <= 7),
    ((board[1, 6] == p) & (board[0, 7] == b)).implies(Count(board, P) <= 7),
    ((board[1, 1] == p) & (board[0, 0] == b)).implies(Count(board, P) <= 7),
)

# Constraints for the empty squares
model += board[occupancy_in_board == False] == E

# Maximize the probabilities of the pieces to get the most likely board
obj = sum(lp[v] for lp, v in zip(lprobs, board[occupancy_in_board]))

model.minimize(obj)

start = time.time()
sat = model.solve()
end = time.time()

print(f"{end-start} seconds needed to solve")
# Solve the model
if sat:
    print(board.value())

    v_replace_value = np.vectorize(replace_value)
    board_string = v_replace_value(board.value())

    print(board_string)

else:
    print("no model found")
