"""
Module for Magellan IMACS

"""
import glob

import numpy as np

from astropy import wcs
from astropy.time import Time
from astropy.io import fits

from pyphot import msgs
from pyphot import parse
from pyphot import telescopes
from pyphot.par import framematch

from pyphot.cameras import camera


class MagellanIMACSCamera(camera.Camera):
    """
    Child to handle Magellan/IMACS specific code
    """
    ndet = 8
    name = 'magellan_imacs'
    telescope = telescopes.MagellanTelescopePar()
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
        self.meta['filter'] = dict(ext=0, card='FILTER')
        self.meta['binning'] = dict(ext=0, card='BINNING', default='1x1')

        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')

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
        ## a specific column indicates whether its flat or not
        flats = np.zeros(len(fitstbl),dtype='bool')
        for i in range(len(fitstbl)):
            if 'flat' in fitstbl[i]['idname'].lower():
                flats[i] = True

        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'Bias')
        if ftype in ['pixelflat', 'illumflat']:
            return good_exp  & flats #& (fitstbl['idname'] == 'Object')
        #if ftype == 'standard':
        #    return good_exp & (fitstbl['idname'] == 'Object') & np.invert(flats)
        if ftype in ['science','supersky','fringe']:
            return good_exp & (fitstbl['idname'] == 'Object') & np.invert(flats)
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'Dark')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

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
        raw_file = raw_file.replace('c1.fits','c{:01d}.fits'.format(det))
        fil = glob.glob(raw_file + '*')
        if len(fil) != 1:
            msgs.error("Found {:d} files matching {:s}".format(len(fil)))

        # Read
        msgs.info("Reading IMACS F2 file: {:s}".format(fil[0]))
        hdu = fits.open(fil[0], memmap=False)
        head1 = fits.getheader(fil[0], 0)

        # get the x and y binning factors...
        detector_par = self.get_detector_par(hdu, det if det is not None else 1)
        ## ToDo: Need to tweak with binned data
        xbin, ybin = parse.parse_binning(detector_par['binning'])

        # Update header with an initial WCS information.
        crpix1 = 1024/xbin
        crpix2 = 2048/xbin
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [crpix1, crpix2]
        if (head1['DEC-D']<-29.01597) & (head1['ROTANGLE']==43.85):
            # Need to flip the WCS
            if det>4:
                cdelt1 = -detector_par['platescale'] * xbin / 3600.
                cdelt2 = -detector_par['platescale'] * ybin / 3600.
            else:
                cdelt1 = detector_par['platescale'] * xbin / 3600.
                cdelt2 = detector_par['platescale'] * ybin / 3600.
            w.wcs.crval = [head1['RA-D']-head1['CHOFFX']/np.cos(head1['DEC-D']/180.* np.pi) / 3600.,
                           head1['DEC-D']+head1['CHOFFY']/3600.]
        else:
            if det>4:
                cdelt1 = detector_par['platescale'] * xbin / 3600.
                cdelt2 = detector_par['platescale'] * ybin / 3600.
            else:
                cdelt1 = -detector_par['platescale'] * xbin / 3600.
                cdelt2 = -detector_par['platescale'] * ybin / 3600.
            w.wcs.crval = [head1['RA-D']+head1['CHOFFX']/np.cos(head1['DEC-D']/180.*np.pi)/3600.,
                           head1['DEC-D']-head1['CHOFFY']/3600.]

        w.wcs.cdelt = np.array([cdelt1, cdelt2])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header_wcs = w.to_header()
        for i in range(len(header_wcs)):
            head1.append(header_wcs.cards[i])

        # First read over the header info to determine the size of the output array...
        #head1.pop('BZERO')
        #head1.pop('BSCALE')
        datasec = head1['DATASEC']
        x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()

        data = hdu[detector_par['dataext']].data*1.0
        array = data[y1-1:y2, x1-1:x2]

        # datasec_img and oscansec_img
        rawdatasec_img = np.ones_like(array) #* detector_par['gain'][0]
        oscansec_img = np.ones_like(array) #* detector_par['ronoise'][0]

        #from pyphot import io
        #io.save_fits('test_c{:01d}.fits'.format(det), data, head1, 'Science', overwrite=True)

        # Need the exposure time
        try:
            exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]
        except:
            exptime = head1[self.meta['exptime']['card']]

        # Return, transposing array back to orient the overscan properly
        return detector_par, array, head1, exptime, rawdatasec_img, oscansec_img


class MagellanIMACSF2Camera(MagellanIMACSCamera):
    """
    Child to handle IMACS/F2 specific code
    """
    name = 'magellan_imacsf2'
    camera = 'IMACS'
    supported = True
    comment = 'IMACS f2 camera'

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
        # TODO: Could this be detector dependent?
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict1 = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,
            specaxis        = 0,
            specflip        = False,
            spatflip        = False,
            platescale      = 0.2,
            darkcurr        = 2.28,
            saturation      = 65535., # ADU
            nonlinear       = 0.95,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(1.56),
            ronoise         = np.atleast_1d(5.4),
            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            darkcurr=1.,
            gain            = np.atleast_1d(1.56),
            ronoise         = np.atleast_1d(5.6),
        ))
        # Detector 3
        detector_dict3 = detector_dict1.copy()
        detector_dict3.update(dict(
            det=3,
            darkcurr=1.,
            gain            = np.atleast_1d(1.68),
            ronoise         = np.atleast_1d(5.4),
        ))
        # Detector 4
        detector_dict4 = detector_dict1.copy()
        detector_dict4.update(dict(
            det=4,
            darkcurr=1.,
            gain            = np.atleast_1d(1.59),
            ronoise         = np.atleast_1d(6.8),
        ))
        # Detector 5
        detector_dict5 = detector_dict1.copy()
        detector_dict5.update(dict(
            det=5,
            darkcurr=1.,
            gain            = np.atleast_1d(1.67), #old:1.58, updated based NB919 observations in May 2022
            ronoise         = np.atleast_1d(5.6),
        ))
        # Detector 6
        detector_dict6 = detector_dict1.copy()
        detector_dict6.update(dict(
            det=6,
            darkcurr=1.,
            gain            = np.atleast_1d(1.70), #old:1.61, updated based NB919 observations in May 2022
            ronoise         = np.atleast_1d(5.9),
        ))
        # Detector 7
        detector_dict7 = detector_dict1.copy()
        detector_dict7.update(dict(
            det=7,
            darkcurr=1.,
            gain            = np.atleast_1d(1.47),#old:1.59, updated based NB919 observations in May 2022
            ronoise         = np.atleast_1d(6.3),
        ))
        # Detector 8
        detector_dict8 = detector_dict1.copy()
        detector_dict8.update(dict(
            det=8,
            darkcurr=1.,
            gain            = np.atleast_1d(1.53), #old:1.65, updated based NB919 observations in May 2022
            ronoise         = np.atleast_1d(6.7),
        ))
        detectors = [detector_dict1, detector_dict2, detector_dict3, detector_dict4,
                     detector_dict5, detector_dict6, detector_dict7, detector_dict8]
        # Return
        return detectors[det-1]
        #return dict('det{:02d}'.format(det) = detectors[det-1] )


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
        par['scienceframe']['process']['use_biasimage'] = True
        par['scienceframe']['process']['use_darkimage'] = False
        par['scienceframe']['process']['use_pixelflat'] = True
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_supersky'] = True
        par['calibrations']['superskyframe']['process']['window_size'] = [256, 256]

        ## We use dome flat for the pixel flat and thus do not need mask bright stars.
        par['calibrations']['pixelflatframe']['process']['mask_brightstar']=False

        # Skybackground
        par['scienceframe']['process']['use_medsky'] = False
        par['scienceframe']['process']['back_size'] = [401, 401]

        # Vignetting
        par['scienceframe']['process']['mask_vig'] = True
        par['scienceframe']['process']['minimum_vig'] = 0.3
        #par['scienceframe']['process']['replace'] = 'zero'
        # sometimes the guider introduce vignetting regions that cannot be fully masked with mask_vig
        par['scienceframe']['process']['mask_negative_star'] = True

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0

        # astrometry
        par['postproc']['astrometry']['mosaic'] = True
        par['postproc']['astrometry']['mosaic_type'] = 'UNCHANGED'
        par['postproc']['astrometry']['astref_catalog'] = 'GAIA-EDR3'
        par['postproc']['astrometry']['astrefmag_limits'] = [18, 21]
        par['postproc']['astrometry']['detect_thresh'] = 10
        par['postproc']['astrometry']['analysis_thresh'] = 10
        par['postproc']['astrometry']['detect_minarea'] = 5
        par['postproc']['astrometry']['crossid_radius'] = 2

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['superskyframe']['exprng'] = [10, None]
        par['calibrations']['fringeframe']['exprng'] = [10, None]
        par['scienceframe']['exprng'] = [None, None]

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

        if self.get_meta_value(scifile, 'filter') == 'NB919':
            # There is no need to subtract fringing for NB919.
            # ToDo: measure the color-term using PS1 z and y bands.
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'z'
            par['postproc']['photometry']['secondary'] = 'y'
            #par['postproc']['photometry']['zpt'] = 24.30 # Meausred from the observations of J1526-2050 on UT 03/09/2021
            par['postproc']['photometry']['zpt'] = 24.45 # Meausred from the observations of J1526-2050 on UT 07/28/2021
                                                         # this is the average of the mosaic. det08 has zeropoint of 24.55
                                                         # and a range of 24.36-24.55 for all detectors
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            # pyphot_colorterm IMACSF2-NB919 PS1-Z PS1-Y --path /Volumes/Work/Imaging/all_dr2_fits
            par['postproc']['photometry']['coefficients'] = [0.015,-0.618,0.]
            # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
            # I use z-band as a proximation for NB919. It actually does not matter since
            # PyPhot calibrates individual chip of each exposure to the ZPT first and then coadds all chips and exposures.
            par['postproc']['photometry']['coeff_airmass'] = 0.02
        elif self.get_meta_value(scifile, 'filter') == 'Sloan_u':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 23.55
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.48
        elif self.get_meta_value(scifile, 'filter') == 'Sloan_g':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['zpt'] = 27.72
            par['postproc']['photometry']['coefficients'] = [0.016, 0.160, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.18
        elif self.get_meta_value(scifile, 'filter') == 'Sloan_r':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['zpt'] = 27.77
            par['postproc']['photometry']['coefficients'] = [0.002, 0.024, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.10
        elif self.get_meta_value(scifile, 'filter') == 'Sloan_i':
            # There is no need to subtract fringing for i-band and other bands
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'i'
            par['postproc']['photometry']['secondary'] = 'z'
            par['postproc']['photometry']['zpt'] = 27.53
            par['postproc']['photometry']['coefficients'] = [0.,0.058,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.04
        elif self.get_meta_value(scifile, 'filter') == 'Sloan_z':
            par['scienceframe']['process']['use_fringe'] = True # Subtract fringing if using z-band
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'z'
            par['postproc']['photometry']['secondary'] = 'y'
            par['postproc']['photometry']['zpt'] = 26.97
            par['postproc']['photometry']['coefficients'] = [-0.011,-0.258,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.02

        return par

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

        msgs.info("Using hard-coded BPM for det={:} on IMACS".format(det))

        # Get the binning
        #hdu = fits.open(filename)
        #binning = hdu[1].header['CCDSUM']
        #hdu.close()

        # Apply the mask
        #xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
        #bpm_img[:, 187 // ybin] = 1

        return bpm_img