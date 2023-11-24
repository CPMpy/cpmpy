import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list


def make_assump_model(soft, hard=[]):
    """
        Construct implied version of all soft constraints
        Can be used to extract cores (see tools.mus)
    """
    # ensure toplevel list
    soft2 = toplevel_list(soft, merge_and=False)

    # make assumption variables
    assump = cp.boolvar(shape=(len(soft),))

    # hard + implied soft constraints
    model = cp.Model(hard + [assump.implies(soft2)])  # each assumption variable implies a candidate

    return model, soft2, assump
