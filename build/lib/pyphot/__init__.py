"""
pyphot package initialization.

The current main purpose of this is to provide package-level globals
that can be imported by submodules.
"""

# Imports for signal and log handling
import warnings

# Set version
__version__ = '0.5.0'

# Import and instantiate the logger
from pyphot import pypmsgs
msgs = pypmsgs.Messages()

from pyphot import check_requirements  # THIS IMPORT DOES THE CHECKING.  KEEP IT

warnings.resetwarnings()
warnings.simplefilter('ignore')
