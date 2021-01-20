""" Uber object for calibration images, e.g. arc, flat

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from pyphot import msgs
from pyphot.par import pyphotpar
from pyphot.images import combineimage
from pyphot.images import pyphotimage
#from pyphot import utils

from IPython import embed

class BiasImage(pyphotimage.PyPhotImage):
    """
    Simple DataContainer for the Tilt Image
    """
    # Set the version of this class
    version = pyphotimage.PyPhotImage.version

    # Output to disk
    output_to_disk = ('BIAS_IMAGE', 'BIAS_DETECTOR')
    hdu_prefix = 'BIAS_'
    master_type = 'Bias'
    master_file_format = 'fits'


class DarkImage(pyphotimage.PyPhotImage):
    """
    Simple DataContainer for the Dark Image
    """
    # Set the version of this class
    version = pyphotimage.PyPhotImage.version

    # Output to disk
    output_to_disk = ('DARK_IMAGE', 'DARK_DETECTOR')
    hdu_prefix = 'DARK_'
    master_type = 'Dark'
    master_file_format = 'fits'



def buildimage_fromlist(spectrograph, det, frame_par, file_list,
                        bias=None, bpm=None, dark=None,
                        flatimages=None,
                        maxiters=5,
                        ignore_saturation=True, slits=None):
    """
    Build a PyPhotImage from a list of files (and instructions)

    Args:
        spectrograph (:class:`pyphot.spectrographs.spectrograph.Spectrograph`):
            Spectrograph used to take the data.
        det (:obj:`int`):
            The 1-indexed detector number to process.
        frame_par (:class:`pyphot.par.pyphotpar.FramePar`):
            Parameters that dictate the processing of the images.  See
            :class:`pyphot.par.pyphotpar.ProcessImagesPar` for the
            defaults.
        file_list (list):
            List of files
        bpm (np.ndarray, optional):
            Bad pixel mask.  Held in ImageMask
        bias (np.ndarray, optional):
            Bias image
        flatimages (:class:`pyphot.flatfield.FlatImages`, optional):  For flat fielding
        maxiters (int, optional):
        ignore_saturation (bool, optional):
            Should be True for calibrations and False otherwise

    Returns:
        :class:`pyphot.images.pyphotimage.PyPhotImage`:  Or one of its children

    """
    # Check
    if not isinstance(frame_par, pyphotpar.FrameGroupPar):
        msgs.error('Provided ParSet for must be type FrameGroupPar.')
    # Do it
    combineImage = combineimage.CombineImage(spectrograph, det, frame_par['process'], file_list)
    pyphotImage = combineImage.run(bias=bias, bpm=bpm, dark=dark,
                                   flatimages=flatimages,
                                   sigma_clip=frame_par['process']['clip'],
                                   sigrej=frame_par['process']['comb_sigrej'], maxiters=maxiters,
                                   ignore_saturation=ignore_saturation, slits=slits,
                                   combine_method=frame_par['process']['combine'])
    #
    # Decorate according to the type of calibration
    #   Primarily for handling MasterFrames
    #   WARNING, any internals in pyphotImage are lost here
    if frame_par['frametype'] == 'bias':
        finalImage = BiasImage.from_pyphotimage(pyphotImage)
    elif frame_par['frametype'] == 'dark':
        finalImage = DarkImage.from_pyphotimage(pyphotImage)
    elif frame_par['frametype'] in ['pixelflat', 'science', 'standard', 'illumflat']:
        finalImage = pyphotImage
    else:
        finalImage = None
        embed()

    # Internals
    finalImage.process_steps = pyphotImage.process_steps
    finalImage.files = file_list
    finalImage.rawheadlist = pyphotImage.rawheadlist
    finalImage.head0 = pyphotImage.head0

    # Return
    return finalImage

