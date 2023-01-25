'''
Custom exception classes, for finer grained error handling
'''


class CPMpyException(Exception):
    '''Parent class for all our exceptions'''
    pass


class MinizincPathException(CPMpyException):
    pass

class ConstraintNotImplementedError(CPMpyException):
    """
        Indicates a constraint is not implemented for a specific solver.
        Should ONLY be thrown from a solvers _post_constraint method.
    """
