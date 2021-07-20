"""
    Methods to transform CPMpy expressions in simpler CPMpy expressions

    Input and output are always CPMpy expressions, so transformations can
    be chained and called multiple times, as needed.

    A transformation can not modify expressions in place but in that case
    should create and return new expression objects. In this way, the
    expressions prior to the transformation remain intact, and could be
    used for other purposes too.

    ==================
    List of submodules
    ==================
    .. autosummary::
        :nosignatures:

        flatten_model
        get_variables
"""
