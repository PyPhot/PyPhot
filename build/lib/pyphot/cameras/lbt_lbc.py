"""
Module for LBT/LBC

"""
import glob,gc
import numpy as np

from astropy import wcs
from astropy.time import Time
from astropy.io import fits

from pyphot import msgs
from pyphot import parse
from pyphot import telescopes
from pyphot.par import framematch
from pyphot.cameras import camera


class LBTLBCCamera(camera.Camera):
    """
    Child to handle Magellan/IMACS specific code
    """
    ndet = 4
    name = 'lbt_lbc'
    telescope = telescopes.LBTTelescopePar()
    supported = True

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='OBSRA')
        self.meta['dec'] = dict(ext=0, card='OBSDEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['filter'] = dict(ext=0, card='FILTER')
        self.meta['binning'] = dict(ext=0, card='LBCBIN', default='1x1')

        #self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['mjd'] = dict(ext=0, card='MJD_OBS')
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='IMAGETYP')

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
            if 'flat' in fitstbl[i]['target'].lower():
                flats[i] = True

        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        copoint_exp = (fitstbl['target'] == 'Co-point')
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'zero') & np.invert(copoint_exp)
        if ftype in ['pixelflat', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'flat') & flats & np.invert(copoint_exp)
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'standard') & np.invert(copoint_exp)
        if ftype in ['science','supersky']:
            return good_exp & (fitstbl['idname'] == 'object') & np.invert(copoint_exp)
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'dark') & np.invert(copoint_exp)
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
        fil = glob.glob(raw_file + '*')
        if len(fil) != 1:
            msgs.error("Found {:d} files matching {:s}".format(len(fil)))

        # Read
        msgs.info("Reading LBT LBC file: {:s}".format(fil[0]))
        hdu = fits.open(fil[0], memmap=False)
        head = hdu[0].header
        head_det = hdu[det].header

        detector_par = self.get_detector_par(hdu, det if det is not None else 1)
        '''
        # get the x and y binning factors...
        msgs.work('Need to tweak with binned data.')
        if len(detector_par['binning'])<=1:
            binning = '1,1'
        else:
            binning = detector_par['binning']
        xbin, ybin = parse.parse_binning(binning)
        '''

        w = wcs.WCS(head_det)
        header_wcs = w.to_header()
        for i in range(len(header_wcs)):
            head.append(header_wcs.cards[i])

        # First read over the header info to determine the size of the output array...
        datasec = head_det['DATASEC']
        x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=True)).flatten()

        if x2>4608: # LBC occationally has reading out problem and would result in some bad rows.
            x2=4608

        ## ToDo: Check whether all data need to do the trim. It seems most of the data have problem at the edges.
        ## Trim some edge pixels for LBT. This can be improved by providing BPM mask or mask using pixelflat,
        ##  i.e. with maskpixvar=0.03
        #x1, x2, y1, y2 = x1+15, x2-10, y1+5, y2-10
        x1, x2, y1, y2 = x1+10, x2-10, y1+10, y2-10

        data = hdu[det].data*1.0
        array = data[x1:x2,y1:y2]

        # datasec_img and oscansec_img
        rawdatasec_img = np.ones_like(array) #* detector_par['gain'][0]
        oscansec_img = np.ones_like(array) #* detector_par['ronoise'][0]

        #from IPython import embed
        #embed()
        #from pyphot import io
        #head = io.initialize_header(hdr=None, primary=False)
        #head1.pop('DATASEC')
        #head1.pop('TRIMSEC')
        #io.save_fits('test_c{:01d}.fits'.format(det), array, head, 'Science', overwrite=True)

        # Need the exposure time
        try:
            exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]
        except:
            exptime = head_det[self.meta['exptime']['card']]

        # release the memory
        del hdu[1].data
        del hdu[2].data
        del hdu[3].data
        del hdu[4].data
        hdu.close()
        gc.collect()

        # Return, transposing array back to orient the overscan properly
        return detector_par, array, head, exptime, rawdatasec_img, oscansec_img


class LBTLBCBCamera(LBTLBCCamera):
    """
    Child to handle LBC_B specific code
    """
    name = 'lbt_lbcb'
    camera = 'LBC'
    supported = True
    comment = 'LBC blue camera'

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
            platescale      = 0.224,
            darkcurr        = 0.01,
            saturation      = 65535., # ADU
            nonlinear       = 0.95,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(1.96),
            ronoise         = np.atleast_1d(5.2),
            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            darkcurr=1.,
            gain            = np.atleast_1d(2.09),
            ronoise         = np.atleast_1d(4.8),
        ))
        # Detector 3
        detector_dict3 = detector_dict1.copy()
        detector_dict3.update(dict(
            det=3,
            darkcurr=1.,
            gain            = np.atleast_1d(2.06),
            ronoise         = np.atleast_1d(4.8),
        ))
        # Detector 4
        detector_dict4 = detector_dict1.copy()
        detector_dict4.update(dict(
            det=4,
            darkcurr=1.,
            gain            = np.atleast_1d(1.98),
            ronoise         = np.atleast_1d(5.0),
        ))
        detectors = [detector_dict1, detector_dict2, detector_dict3, detector_dict4]
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
        par['scienceframe']['process']['use_fringe'] = False
        par['calibrations']['superskyframe']['process']['window_size'] = [101, 101]

        # Background type for image processing
        par['scienceframe']['process']['use_medsky'] = False

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0

        # astrometry
        par['postproc']['astrometry']['mosaic'] = True
        par['postproc']['astrometry']['mosaic_type'] = 'UNCHANGED'
        par['postproc']['astrometry']['astref_catalog'] = 'PANSTARRS-1'
        par['postproc']['astrometry']['astrefmag_limits'] = [18, 23.5] # change the bright end limit if your image is shallow
        par['postproc']['astrometry']['astrefsn_limits'] = [7, 10.0]
        par['postproc']['astrometry']['posangle_maxerr'] = 5.0
        par['postproc']['astrometry']['position_maxerr'] = 0.5
        par['postproc']['astrometry']['pixscale_maxerr'] = 1.1
        par['postproc']['astrometry']['detect_thresh'] = 20 # increasing this can improve the solution if your image is deep
        par['postproc']['astrometry']['analysis_thresh'] = 20
        par['postproc']['astrometry']['detect_minarea'] = 7
        par['postproc']['astrometry']['crossid_radius'] = 2

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 10]
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [10, None]

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

        # https://sites.google.com/a/lbto.org/lbc/phase-ii-guidelines/sensitivities
        # LBT gives ZP for 1 ADU/s, PyPhot use 1 e/s
        # ZP_ADU = ZP_e - 2.5*np.log10(gain)
        if self.get_meta_value(scifile, 'filter') == 'SDT_Uspec':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 28.13 #2.5*np.log10(2.09)+27.33
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.47 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'U-BESSEL':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 27.03 #2.5*np.log10(2.09)+26.23
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            par['postproc']['photometry']['coefficients'] = [0.,0.,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.48 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'B-BESSEL':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['zpt'] = 28.73 #2.5*np.log10(2.09)+27.93
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.22 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'V-BESSEL':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['zpt'] = 28.93 #2.5*np.log10(2.09)+28.13
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.15 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'g-SLOAN':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'g'
            #par['postproc']['photometry']['secondary'] = 'r'
            #par['postproc']['photometry']['coefficients'] = [0., -0.086, 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['coefficients'] = [0.016, 0.160, 0.]
            par['postproc']['photometry']['zpt'] = 29.11 #2.5*np.log10(2.09)+28.31
            par['postproc']['photometry']['coeff_airmass'] = 0.17 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'r-SLOAN':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'r'
            #par['postproc']['photometry']['secondary'] = 'g'
            #par['postproc']['photometry']['coefficients'] = [0., 0.016, 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['coefficients'] = [0.002, 0.024, 0.]
            par['postproc']['photometry']['zpt'] = 28.55 #2.5*np.log10(2.09)+27.75, consistent with J0100, 27.67 for 1ADU/s
            par['postproc']['photometry']['coeff_airmass'] = 0.11 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)

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
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        msgs.info("Using hard-coded BPM for det={:} on LBCB".format(det))

        # Get the binning
        #hdu = fits.open(filename)
        #binning = hdu[1].header['CCDSUM']
        #hdu.close()

        # Apply the mask
        #xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
        #bpm_img[:, 187 // ybin] = 1

        return bpm_img

class LBTLBCRCamera(LBTLBCCamera):
    """
    Child to handle LBC_Rspecific code
    """
    name = 'lbt_lbcr'
    camera = 'LBC'
    supported = True
    comment = 'LBC red camera'

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
            platescale      = 0.224,
            darkcurr        = 0.01,
            saturation      = 65535., # ADU
            nonlinear       = 0.95,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(2.08),
            ronoise         = np.atleast_1d(5.0),
            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            darkcurr=1.,
            gain            = np.atleast_1d(2.14),
            ronoise         = np.atleast_1d(5.0),
        ))
        # Detector 3
        detector_dict3 = detector_dict1.copy()
        detector_dict3.update(dict(
            det=3,
            darkcurr=1.,
            gain            = np.atleast_1d(2.13),
            ronoise         = np.atleast_1d(5.3),
        ))
        # Detector 4
        detector_dict4 = detector_dict1.copy()
        detector_dict4.update(dict(
            det=4,
            darkcurr=1.,
            gain            = np.atleast_1d(2.09),
            ronoise         = np.atleast_1d(4.8),
        ))
        detectors = [detector_dict1, detector_dict2, detector_dict3, detector_dict4]
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

        # Calibrations
        # PyPhot default is 0.1. Use 0.03 to remove more bad pixels
        #par['calibrations']['pixelflatframe']['process']['maskpixvar'] =0.03 # This masks too many pixels. Not set maskpixvar

        # Image processing steps
        turn_off = dict(use_illumflat=False, use_biasimage=False, use_overscan=False,
                        use_darkimage=False)
        par.reset_all_processimages_par(**turn_off)
        par['scienceframe']['process']['use_biasimage'] = True
        par['scienceframe']['process']['use_darkimage'] = False
        par['scienceframe']['process']['use_pixelflat'] = True
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_supersky'] = True
        par['scienceframe']['process']['use_fringe'] = True

        # Background type for image processing
        par['scienceframe']['process']['use_medsky'] = False

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0
        par['scienceframe']['process']['grow'] = 0.5

        # astrometry
        par['postproc']['astrometry']['mosaic'] = True
        par['postproc']['astrometry']['mosaic_type'] = 'UNCHANGED'
        par['postproc']['astrometry']['astref_catalog'] = 'PANSTARRS-1'
        par['postproc']['astrometry']['astrefmag_limits'] = [18, 23] # change the bright end limit if your image is shallow
        par['postproc']['astrometry']['astrefsn_limits'] = [7, 10.0]
        par['postproc']['astrometry']['posangle_maxerr'] = 5.0
        par['postproc']['astrometry']['position_maxerr'] = 0.5
        par['postproc']['astrometry']['pixscale_maxerr'] = 1.1
        par['postproc']['astrometry']['detect_thresh'] = 10 # increasing this can improve the solution if your image is deep
        par['postproc']['astrometry']['analysis_thresh'] = 10
        par['postproc']['astrometry']['detect_minarea'] = 5
        par['postproc']['astrometry']['crossid_radius'] = 2

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 10]
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [10, None]

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

        # https://sites.google.com/a/lbto.org/lbc/phase-ii-guidelines/sensitivities
        # LBT gives ZP for 1 ADU/s, PyPhot use 1 e/s
        # ZP_ADU = ZP_e - 2.5*np.log10(gain)
        if self.get_meta_value(scifile, 'filter') == 'V-BESSEL':
            par['postproc']['photometry']['photref_catalog'] = 'Sloan'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 28.77 #2.5*np.log10(2.14)+27.94
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            par['postproc']['photometry']['coefficients'] = [0.,0.,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.16 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'R-BESSEL':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'r'
            #par['postproc']['photometry']['secondary'] = 'g'
            #par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['coefficients'] = [-0.010,-0.218, 0.]
            par['postproc']['photometry']['zpt'] = 28.69 #2.5*np.log10(2.14)+27.86
            par['postproc']['photometry']['coeff_airmass'] = 0.13 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'I-BESSEL':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'i'
            #par['postproc']['photometry']['secondary'] = 'r'
            #par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'i'
            par['postproc']['photometry']['secondary'] = 'z'
            par['postproc']['photometry']['coefficients'] = [-0.003,-0.411,0.]
            #par['postproc']['photometry']['zpt'] = 28.42 #2.5*np.log10(2.14)+27.59
            par['postproc']['photometry']['zpt'] = 28.56 #measured from 2020A observations of J0706
            par['postproc']['photometry']['coeff_airmass'] = 0.04 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'r-SLOAN':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'r'
            #par['postproc']['photometry']['secondary'] = 'i'
            #par['postproc']['photometry']['coefficients'] = [0., -0.014, 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['coefficients'] = [0.002, 0.024, 0.]
            par['postproc']['photometry']['zpt'] = 28.86 #2.5*np.log10(2.14)+28.03
            par['postproc']['photometry']['coeff_airmass'] = 0.09 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'i-SLOAN':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'i'
            #par['postproc']['photometry']['secondary'] = 'z'
            #par['postproc']['photometry']['coefficients'] = [0.,0.072, 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'i'
            par['postproc']['photometry']['secondary'] = 'z'
            par['postproc']['photometry']['coefficients'] = [0.,0.058,0.]
            par['postproc']['photometry']['zpt'] = 28.66 #2.5*np.log10(2.14)+27.83, measured from J0100 observations
            par['postproc']['photometry']['coeff_airmass'] = 0.03 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'z-SLOAN':
            #par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            #par['postproc']['photometry']['primary'] = 'z'
            #par['postproc']['photometry']['secondary'] = 'i'
            #par['postproc']['photometry']['coefficients'] = [0., 0.020, 0.]
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'z'
            par['postproc']['photometry']['secondary'] = 'y'
            par['postproc']['photometry']['zpt'] = 28.03 # For 1 e/s, 2.5*np.log10(2.14)+27.2, consistent with J0100, 27.25 for 1ADU/s
            par['postproc']['photometry']['coefficients'] = [-0.011,-0.258,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.04 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)

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
        ## a specific column indicates whether its flat or not
        flats = np.zeros(len(fitstbl),dtype='bool')
        for i in range(len(fitstbl)):
            if 'flat' in fitstbl[i]['target'].lower():
                flats[i] = True

        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        copoint_exp = (fitstbl['target'] == 'Co-point')
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'zero') & np.invert(copoint_exp)
        if ftype in ['pixelflat', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'flat') & flats & np.invert(copoint_exp)
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'standard') & np.invert(copoint_exp)
        if ftype in ['science','supersky','fringe']:
            return good_exp & (fitstbl['idname'] == 'object') & np.invert(copoint_exp)
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'dark') & np.invert(copoint_exp)
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
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        msgs.info("Using hard-coded BPM for det={:} on LBCR".format(det))

        # Get the binning
        #hdu = fits.open(filename)
        #binning = hdu[1].header['CCDSUM']
        #hdu.close()

        # Apply the mask
        #xbin, ybin = int(binning.split(' ')[0]), int(binning.split(' ')[1])
        #bpm_img[:, 187 // ybin] = 1

        return bpm_img