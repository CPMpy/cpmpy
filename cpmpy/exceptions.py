'''
Custom exception classes, for finer grained error handling
'''


class CPMpyException(Exception):
    '''Parent class for all our exceptions'''
    pass


class MinizincPathException(CPMpyException):
    '''Raised when `minizinc` is not added to PATH'''
    pass

class MinizincNameException(CPMpyException):
    '''Raised when a variable is a keyword or otherwise violates Minizinc naming rules'''
    pass

class MinizincBoundsException(CPMpyException):
    '''Raised when an integer overflows MiniZinc's bounds of (-2147483646..2147483646)'''
    pass

class ChocoBoundsException(CPMpyException):
    '''Raised when an integer overflows Choco's integer bounds of (-2147483646..2147483646)'''
    pass

class NotSupportedError(CPMpyException):
    '''Raised when a solver does not support a certain feature'''
    pass

class IncompleteFunctionError(CPMpyException):
    '''Raised when an expression's value is not defined for its sub-expressions (e.g. `x div y` where `y` is assigned 0)'''
    pass

class TypeError(CPMpyException):
    '''Raised when an expression receives sub-expressions of the wrong type'''
    pass

class GCSVerificationException(CPMpyException):
    '''Raised when GCS fails proof logging by VeriPB'''
    pass

class TransformationNotImplementedError(CPMpyException):
    '''Raised when a transformation is not implemented for a certain expression'''
    pass