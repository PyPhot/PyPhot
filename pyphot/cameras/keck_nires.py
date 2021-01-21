"""
Module for Keck NIRES slit view camera

.. include:: ../include/links.rst
"""
import glob

import numpy as np
from scipy.signal import savgol_filter

from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from pyphot import msgs
from pyphot import parse
from pyphot import telescopes
from pyphot.par import framematch

from pyphot.cameras import camera


class KECKNIRESCamera(camera.Camera):
    """
    Child to handle MMT/MMIRS specific code
    """
    ndet = 1
    name = 'keck_nires'
    telescope = telescopes.KeckTelescopePar()
    camera = 'NIRES'
    supported = True

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['filter'] = dict(ext=0, card=None, default='K')
        self.meta['binning'] = dict(ext=0, card=None, default='1,1')

        self.meta['mjd'] = dict(ext=0, card='MJD-OBS')
        self.meta['exptime'] = dict(ext=0, card='ITIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='OBSTYPE')

    def get_detector_par(self, hdu, det):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        # TODO: Could this be detector dependent??
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict = dict(
            binning=binning,
            det=1,
            dataext=1,
            platescale=0.125,
            darkcurr=0.01,
            saturation=160000.,  # 155400.,
            nonlinear=0.1,
            mincounts=-1e10,
            numamplifiers=1,
            gain=np.atleast_1d(4.98),
            ronoise=np.atleast_1d(26),
            datasec=np.atleast_1d('[:,:]'),
            oscansec=np.atleast_1d('[:,:]')
        )
        return dict(det01=detector_dict)

    @classmethod
    def default_pyphot_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pyphot.par.pyphotpar.PyPhotPar`: Parameters required by
            all of ``PyPhot`` methods.
        """
        par = super().default_pyphot_par()

        # Image processing steps
        turn_off = dict(use_illumflat=False, use_biasimage=False, use_overscan=False,
                        use_darkimage=False)
        par.reset_all_processimages_par(**turn_off)
        par['scienceframe']['process']['use_darkimage'] = False
        par['scienceframe']['process']['use_pixelflat'] = True
        par['scienceframe']['process']['use_illumflat'] = True


        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 2]
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [2, None]

        # dark
        par['calibrations']['darkframe']['process']['apply_gain'] = True

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0
        par['scienceframe']['process']['grow'] = 0.5

        return par

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'bias':
            # No bias frames
            return np.zeros(len(fitstbl), dtype=bool)
        #if ftype in ['pixelflat', 'trace', 'illumflat']:
        #    return good_exp & (fitstbl['idname'] == 'flat')
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'Object')
        if ftype in ['science', 'pixelflat', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'Object')
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'dark')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        msgs.info("Using hard-coded BPM for det=1 on MMIRS")

        # Get the binning
        #hdu = fits.open(filename)
        #binning = hdu[1].header['CCDSUM']
        #hdu.close()

        # Apply the mask
        #xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
        #bpm_img[:, 187 // ybin] = 1

        return bpm_img

    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : :class:`pypeit.images.detector_container.DetectorContainer`
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector (after overscan subtraction).
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time read from the file header
        """
        # Check for file; allow for extra .gz, etc. suffix
        fil = glob.glob(raw_file + '*')
        if len(fil) != 1:
            msgs.error("Found {:d} files matching {:s}".format(len(fil)))

        # Read
        msgs.info("Reading NIRES file: {:s}".format(fil[0]))
        hdu = fits.open(fil[0])
        head1 = fits.getheader(fil[0], 0)

        detector_par = self.get_detector_par(hdu, det if det is None else 1)

        data = hdu[0].data
        # First read over the header info to determine the size of the output array...
        array = data[:986, :]
        #array = np.copy(data)

        gainimage = np.ones_like(array) * detector_par['det01']['gain'][0]
        rnimage = np.ones_like(array) * detector_par['det01']['ronoise'][0]

        # Need the exposure time
        exptime = hdu[0].header[self.meta['exptime']['card']]

        # Return, transposing array back to orient the overscan properly
        return detector_par, array, head1, exptime, gainimage, rnimage