##
"""
Type aliases used to annotate
"""

import cpmpy as cp
import numpy as np

# types
from typing import List, Tuple, Union, TypeVar
from typing_extensions import TypeAlias

BoolConstant: TypeAlias = Union[cp.BoolVal, bool, np.bool_]
IntegerConstant: TypeAlias = Union[BoolConstant, int, np.integer]
NumericConstant: TypeAlias = Union[IntegerConstant, float, np.floating]
ExpressionLike: TypeAlias = Union[cp.expressions.core.Expression, NumericConstant]

T = TypeVar("T")
FlatList: TypeAlias = Union[List[T], Tuple[T], np.ndarray] # TODO: Numpy has better typing too
AnyList: TypeAlias = FlatList[Union[T, "AnyList[T]"]]

FlatExprList: TypeAlias = AnyList[ExpressionLike]
AnyExprList: TypeAlias = AnyList[Union["AnyExprList", ExpressionLike]]


