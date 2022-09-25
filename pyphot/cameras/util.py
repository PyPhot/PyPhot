"""
Camera utility methods.
Bookkeeping from PyPeIt.
"""

from pyphot import cameras
from pyphot import msgs


def load_camera(cam):
    """
    Instantiate a spectrograph from the available subclasses of
    :class:`~pypeit.spectrographs.spectrograph.Spectrograph`.

    Args:
        spec (:obj:`str`, :class:`~pypeit.spectrographs.spectrograph.Spectrograph`):
            The spectrograph to instantiate. If the input object is ``None``
            or has :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
            as a base class, the instance is simply returned. If it is a
            string, the string is used to instantiate the relevant
            spectrograph instance.

    Returns:
        :class:`~pypeit.spectrographs.spectrograph.Spectrograph`: The
        spectrograph used to obtain the data to be reduced.

    Raises:
        PypeItError:
            Raised if the input is a string that does not select a recognized
            ``PypeIt`` spectrograph.
    """
    if cam is None or isinstance(cam, cameras.camera.Camera):
        return cam

    classes = cameras.camera_classes()
    if cam in classes.keys():
        return classes[cam]()

    msgs.error('{0} is not a supported camera.'.format(cam))


