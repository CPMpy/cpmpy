'''
Custom exception classes, for finer grained error handling
'''


class CPMpyException(Exception):
    '''Parent class for all our exceptions'''
    pass


class MinizincPathException(CPMpyException):
    pass

class MinizincNameException(CPMpyException):
    pass


class NotSupportedError(CPMpyException):
    pass

class IncompleteFunctionError(CPMpyException):
    pass