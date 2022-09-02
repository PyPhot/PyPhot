'''
Bookkeeping from PyPeIt.
'''

from pyphot.cameras import camera

# The import of all the camera modules here is what enables the dynamic
# compiling of all the available cameras below
from pyphot.cameras import cfht_wircam
from pyphot.cameras import keck_nires
from pyphot.cameras import keck_lris
from pyphot.cameras import lbt_lbc
from pyphot.cameras import magellan_imacs
from pyphot.cameras import mmt_mmirs

# Build the list of names for the available cameras
import numpy as np

def all_subclasses(cls):
    """
    Thanks to:
    https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

def camera_classes():
    # Recursively collect all subclasses
    spec_c = np.array(list(all_subclasses(camera.Camera)))
    # Select spectrograph classes with a defined name; cameras without a
    # name are either undefined or a base class.
    spec_c = spec_c[[c.name is not None for c in spec_c]]
    # Construct a dictionary with the spectrograph name and class
    srt = np.argsort(np.array([c.name for c in spec_c]))
    return dict([ (c.name,c) for c in spec_c[srt]])

available_cameras = list(camera_classes().keys())

