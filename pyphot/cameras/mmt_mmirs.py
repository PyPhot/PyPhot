"""
Module for MMT MMIRS

Modified from PyPeIt
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


class MMTMMIRSCamera(camera.Camera):
    """
    Child to handle MMT/MMIRS specific code
    """
    ndet = 1
    name = 'mmt_mmirs'
    telescope = telescopes.MMTTelescopePar()
    camera = 'MMIRS'
    supported = True

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=1, card='RA')
        self.meta['dec'] = dict(ext=1, card='DEC')
        self.meta['target'] = dict(ext=1, card='OBJECT')
        self.meta['filter'] = dict(ext=1, card='FILTER')
        self.meta['binning'] = dict(ext=1, card=None, default='1,1')

        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['exptime'] = dict(ext=1, card='EXPTIME')
        self.meta['airmass'] = dict(ext=1, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=1, card='IMAGETYP')

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        # TODO: This should be how we always deal with timeunit = 'isot'. Are
        # we doing that for all the relevant spectrographs?
        if meta_key == 'mjd':
            try:
                time = headarr[1]['DATE-OBS']
            except:
                time = headarr[0]['DATE-OBS']
            ttime = Time(time, format='isot')
            return ttime.mjd
        msgs.error("Not ready for this compound meta")

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
            platescale=0.2012,
            darkcurr=0.01,
            saturation=700000.,  # 155400.,
            nonlinear=1.0,
            mincounts=-1e10,
            numamplifiers=1,
            gain=np.atleast_1d(2.68),
            ronoise=np.atleast_1d(3.14),
            datasec=np.atleast_1d('[:,:]'),
            oscansec=np.atleast_1d('[:,:]')
        )
        #return dict(det01=detector_dict)
        return detector_dict

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
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_supersky'] = False


        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 2]
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [2, None]

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0
        par['scienceframe']['process']['grow'] = 0.5

        # astrometry
        par['postproc']['astrometry']['position_maxerr'] = 0.5
        par['postproc']['astrometry']['posangle_maxerr'] = 10.0
        par['postproc']['astrometry']['detect_thresh'] = 7
        par['postproc']['astrometry']['analysis_thresh'] = 7
        par['postproc']['astrometry']['detect_minarea'] = 5
        par['postproc']['astrometry']['crossid_radius'] = 1
        par['postproc']['astrometry']['astrefmag_limits'] = [17, 21]
        par['postproc']['astrometry']['delete'] = False
        par['postproc']['astrometry']['log'] = True

        # Photometry
        par['postproc']['photometry']['external_flag'] = False
        par['postproc']['photometry']['nstar_min'] = 5

        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PypeIt`` parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        par = super().config_specific_par(scifile, inp_par=inp_par)

        if self.get_meta_value(scifile, 'filter') == 'J':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'J'
            par['postproc']['photometry']['secondary'] = 'H'
            par['postproc']['photometry']['zpt'] = 26.85
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            par['postproc']['photometry']['coefficients'] = [0.,0.,0.]
        elif self.get_meta_value(scifile, 'filter') == 'H':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'H'
            par['postproc']['photometry']['secondary'] = 'K'
            par['postproc']['photometry']['zpt'] = 21.0
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
        elif self.get_meta_value(scifile, 'filter') == 'K':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'K'
            par['postproc']['photometry']['secondary'] = 'H'
            par['postproc']['photometry']['zpt'] = 21.0
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]

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
            return good_exp & (fitstbl['idname'] == 'object')
        if ftype in ['science', 'pixelflat', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'object')
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

        msgs.info("Using hard-coded BPM for det={:} on MMIRS".format(det))

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
        msgs.info("Reading MMIRS file: {:s}".format(fil[0]))
        hdu = fits.open(fil[0])
        try:
            head1 = fits.getheader(fil[0], 1)
        except:
            head1 = fits.getheader(fil[0], 0)

        detector_par = self.get_detector_par(hdu, det if det is None else 1)

        # get the x and y binning factors...
        binning = head1['CCDSUM']
        xbin, ybin = [int(ibin) for ibin in binning.split(' ')]

        # First read over the header info to determine the size of the output array...
        datasec = head1['DATASEC']
        x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()

        # ToDo: I am currently using the standard double correlated frame, that is a difference between
        # the first and final read-outs. In the future need to explore up-the-ramp fitting.
        '''
        if len(hdu) > 2:
            data = hdu[1].data.astype('float64') - hdu[2].data.astype('float64')
        else:
            data = hdu[1].data.astype('float64')
        '''
        if len(hdu) > 2:
            data = mmirs_read_amp(hdu[1].data.astype('float64')) - mmirs_read_amp(hdu[2].data.astype('float64'))
        elif len(hdu)==2:
            data = mmirs_read_amp(hdu[1].data.astype('float64'))
        else:
            data = mmirs_read_amp(hdu[0].data.astype('float64'))

        array = data[x1 - 1:x2, y1 - 1:y2]

        rawdatasec_img = np.ones_like(array) #* detector_par['gain'][0]
        oscansec_img = np.ones_like(array) #* detector_par['ronoise'][0]

        # Need the exposure time
        try:
            exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]
        except:
            exptime = hdu[0].header[self.meta['exptime']['card']]

        # Return, transposing array back to orient the overscan properly

        #ToDo: need to return ramp_image which will be used for procimg

        return detector_par, array, head1, exptime, rawdatasec_img, oscansec_img

def mmirs_read_amp(img, namps=32):
    """
    MMIRS has 32 reading out channels. Need to deal with this issue a little
    bit. I am not using the pypeit overscan subtraction since we need to do
    the up-the-ramp fitting in the future.

    Imported from MMIRS IDL pipeline refpix.pro
    """

    ## ToDo: Import ramp fitting from https://github.com/spacetelescope/stcal/tree/main/src/stcal/ramp_fitting

    # number of channels for reading out
    if namps is None:
        namps = 32

    data_shape = np.shape(img)
    ampsize = int(data_shape[0] / namps)

    refpix1 = np.array([1, 2, 3])
    refpix2 = np.arange(4) + data_shape[0] - 4
    refpix_all = np.hstack([[0, 1, 2, 3], np.arange(4) + data_shape[0] - 4])
    refvec = np.sum(img[:, refpix_all], axis=1) / np.size(refpix_all)
    svec = savgol_filter(refvec, 11, polyorder=5)

    refvec_2d = np.reshape(np.repeat(svec, data_shape[0], axis=0), data_shape)
    img_out = img - refvec_2d

    for amp in range(namps):
        img_out_ref = img_out[np.hstack([refpix1, refpix2]), :]
        ref1, med1, std1 = sigma_clipped_stats(img_out_ref[:, amp * ampsize + 2 * np.arange(int(ampsize / 2))],
                                               sigma=3)
        ref2, med2, std2 = sigma_clipped_stats(img_out_ref[:, amp * ampsize + 2 * np.arange(int(ampsize / 2)) + 1],
                                               sigma=3)
        ref12 = (ref1 + ref2) / 2.
        img_out[:, amp * ampsize:(amp + 1) * ampsize] -= ref12

    return img_out



