"""
    Transformations are used by solvers to convert (high-level) CPMpy expressions
    into the low-level constraints they support.

    Typical users never need to use these functions directly.

    **CPMpy’s transformations** selectively rewrite only those constraint expressions that a solver does not support. While solvers can use any transformation they need, lower-level solvers largely reuse those of higher-level ones, creating a waterfall pattern:

    .. image:: ../waterfall.png
        :width: 480
        :alt: Waterfall from model to solvers

    The rest of this documentation is for solver developers and other advanced users.
    Make sure you read `Adding a new solver <../adding_solver.html>`_ first.
     
    A transformation can not modify expressions in-place but in that case
    should create and return new expression objects (copy-on-write). In this way, the
    expressions prior to the transformation remain intact, and could be
    used for other purposes too.

    ==================
    List of submodules
    ==================

    Input and output to transformations are always CPMpy expressions, so transformations can
    be chained and called multiple times, as needed. While there is no fixed ordering,
    the following ordering corresponds to the waterfall:

    .. autosummary::
        :nosignatures:

        get_variables
        normalize
        safening
        decompose_global
        negation
        flatten_model
        reification
        comparison
        linearize
        to_cnf
"""
