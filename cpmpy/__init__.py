# Tias Guns, 2019
# current version
#==============================================================================
VERSION = (0, 5, 2, "dev")

__version__ = "%d.%d.%d.%s" % VERSION if len(VERSION) == 4 else \
              "%d.%d.%d" % VERSION

from .variables import *
from .expressions import *
from .globalconstraints import *
from .model import *
