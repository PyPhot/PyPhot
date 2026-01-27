"""
Module for CFHT WIRCam

"""
import glob
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from pyphot import msgs
from pyphot import parse
from pyphot import telescopes
from pyphot.par import framematch
from pyphot.cameras import camera


class CFHTWIRCAMCamera(camera.Camera):
    """
    Child to handle CFHT/WIRCam specific code
    """
    ndet = 4
    name = 'cfht_wircam'
    telescope = telescopes.CFHTTelescopePar()
    camera = 'WIRCAM'
    supported = True

    def init_meta(self):
        """
        Define how metadata are derived from the camera files.

        That is, this associates the ``PyPhot``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA')
        self.meta['dec'] = dict(ext=0, card='DEC')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        self.meta['filter'] = dict(ext=0, card='FILTER')
        self.meta['binning'] = dict(ext=0, card=None, compound=True)

        self.meta['mjd'] = dict(ext=0, card='MJD-OBS')
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')

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

        N_microdither = hdu[det].header['MDCOORDS']
        xbin = np.sqrt(N_microdither)

        # Detector 1
        detector_dict1 = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,
            specaxis        = 0,
            specflip        = False,
            spatflip        = False,
            platescale      = 0.306/xbin,
            darkcurr        = 0.05,
            saturation      = 65535., # ADU
            nonlinear       = 0.5,
            mincounts       = -1e10,
            numamplifiers   = 1,
            gain            = np.atleast_1d(3.8),
            ronoise         = np.atleast_1d(30.0),
            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
        ))
        # Detector 3
        detector_dict3 = detector_dict1.copy()
        detector_dict3.update(dict(
            det=3,
        ))
        # Detector 4
        detector_dict4 = detector_dict1.copy()
        detector_dict4.update(dict(
            det=4,
        ))
        detectors = [detector_dict1, detector_dict2, detector_dict3, detector_dict4]
        # Return
        return detectors[det-1]

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
        if meta_key == 'binning':
            return '{:}x{:}'.format(headarr[0]['CCDBIN1'],headarr[0]['CCDBIN2'])
        msgs.error("Not ready for this compound meta")

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
        ## ToDo: I am currently using the ??????s.fits images which are pre-processed images (dark subtracted, flat fielded, etc)
        ## You can also use ??????p.fits images which are sky-background subtracted images. However, the sky subtraction
        ##  by CFHT is not ideal in some cases (i.e. >60s exposures), so use that with caution.
        ## In order to get a better sky-subtraction, you can group your science images accordingly, so you can perform the
        ## fringe subtraction for a list of images that obtained at similar time-scale. The fringe subtraction here is
        ## for subtracting off dirty stuff not the same with fringe in optical CCD.
        par['scienceframe']['process']['use_biasimage'] = False
        par['scienceframe']['process']['use_darkimage'] = False
        par['scienceframe']['process']['use_pixelflat'] = False
        par['scienceframe']['process']['use_illumflat'] = False
        par['scienceframe']['process']['use_supersky'] = False
        par['scienceframe']['process']['use_fringe'] = True
        par['scienceframe']['process']['mask_negative_star'] = True # detector 4 is very dirty, need to mask out some negative stars

        # Vignetting
        par['scienceframe']['process']['mask_vig'] = False
        par['scienceframe']['process']['minimum_vig'] = 0.7
        par['scienceframe']['process']['brightstar_nsigma'] = 3

        # The WIRCam processed replace bad pixels with zero, so I would also replace with zeros.
        par['scienceframe']['process']['replace'] = 'zero'

        # cosmic ray rejection
        # I set to False for WIRCam since WIRCam processed data set bad pixels to zero
        # and all pixels close to bad pixels will be masked as cosmic rays
        par['scienceframe']['process']['mask_cr'] = False
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0
        par['scienceframe']['process']['grow'] = 0.5

        # astrometry
        par['postproc']['astrometry']['mosaic'] = True
        par['postproc']['astrometry']['mosaic_type'] = 'UNCHANGED'
        par['postproc']['astrometry']['astref_catalog'] = 'GAIA-EDR3'
        par['postproc']['astrometry']['position_maxerr'] = 1.0
        par['postproc']['astrometry']['detect_thresh'] = 5
        par['postproc']['astrometry']['analysis_thresh'] = 5
        par['postproc']['astrometry']['detect_minarea'] = 5
        par['postproc']['astrometry']['crossid_radius'] = 2
        par['postproc']['astrometry']['delete'] = True
        par['postproc']['astrometry']['log'] = False

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['standardframe']['exprng'] = [None, 6]
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [6, None]

        return par


    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PyPhot`` parameters to hard-wired values used for
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

        # https://www.cfht.hawaii.edu/Instruments/Imaging/WIRCam/quickinformation.html
        if self.get_meta_value(scifile, 'filter') == 'Y':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'J'
            par['postproc']['photometry']['secondary'] = 'H'
            par['postproc']['photometry']['zpt'] = 25.75
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            # I used the UKIRT Y-band as a proximation, which was derived by Hodgkin et al. 2009, MNRAS, 394, 675
            # Hodgkin+09 using vega system, I corrected it to AB using Hewett+06 vega-AB offset and derived that
            #  Y_AB = J_AB,2MASS + 0.029 + 0.5*(J_AB,2MASS-H_AB,2MASS)
            # This is consistent with what I derived below
            # pyphot_colorterm UKIRT-Y TMASS-J TMASS-H --path /Volumes/Work/Imaging/all_dr2_fits
            par['postproc']['photometry']['coefficients'] = [0.003,0.694,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.02 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'J':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'J'
            par['postproc']['photometry']['secondary'] = 'H'
            par['postproc']['photometry']['zpt'] = 26.078
            par['postproc']['photometry']['coefficients'] = [-0.001, -0.059, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.05
        elif self.get_meta_value(scifile, 'filter') == 'H':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'H'
            par['postproc']['photometry']['secondary'] = 'J'
            par['postproc']['photometry']['zpt'] = 26.72
            par['postproc']['photometry']['coefficients'] = [-0.007, -0.035, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.03
        elif self.get_meta_value(scifile, 'filter') == 'Ks':
            par['postproc']['photometry']['photref_catalog'] = 'TwoMass'
            par['postproc']['photometry']['primary'] = 'K'
            par['postproc']['photometry']['secondary'] = 'H'
            par['postproc']['photometry']['zpt'] = 26.54
            par['postproc']['photometry']['coefficients'] = [0.023,-0.031, 0.] # I used VISTA Ks as an approximation.
            par['postproc']['photometry']['coeff_airmass'] = 0.05

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
        if ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'zero')
        if ftype in ['pixelflat', 'illumflat']:
            return good_exp & (fitstbl['idname'] == 'flat') & flats
        if ftype == 'standard':
            return good_exp & (fitstbl['idname'] == 'standard')
        if ftype in ['science','supersky','fringe']:
            return good_exp & (fitstbl['idname'] == 'OBJECT')
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
        msgs.info("Reading CFHT WIRCam processed image: {:s}".format(fil[0]))
        hdu = fits.open(fil[0], memmap=False)
        head = fits.getheader(fil[0], 0)
        head_det = fits.getheader(fil[0], det)

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
        # First read over the header info to determine the size of the output array...
        datasec = head_det['DATASEC']
        x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=True)).flatten()

        N_microdither = head_det['MDCOORDS']
        if N_microdither == 1:
            #header
            w = wcs.WCS(head_det)
            header_wcs = w.to_header()
            for i in range(len(header_wcs)):
                head.append(header_wcs.cards[i])

            # data
            data = hdu[det].data*1.0 #- hdu[det]['CHIPBIAS']

        elif N_microdither ==4:
            ## deal with microdither, it should also works for N_microdither=9, but who knows!!!
            xbin, ybin = np.sqrt(N_microdither).astype('int'), np.sqrt(N_microdither).astype('int')
            # header
            w = wcs.WCS(naxis=2)
            w.wcs.crpix = [head_det['CRPIX1']*xbin, head_det['CRPIX2']*ybin]
            w.wcs.cdelt = [head_det['CD1_1']/xbin, head_det['CD2_2']/ybin]
            w.wcs.crval = [head_det['CRVAL1'], head_det['CRVAL2']]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            header_wcs = w.to_header()
            for i in range(len(header_wcs)):
                head.append(header_wcs.cards[i])

            # data
            cube = np.zeros((N_microdither, head_det['NAXIS1']*xbin,head_det['NAXIS2']*ybin))
            cube_flag = np.zeros((N_microdither, head_det['NAXIS1']*xbin,head_det['NAXIS2']*ybin))
            pos0 = head_det['MDCOORD1'].split(',')
            x_pos0, y_pos0 = float(pos0[0][1:]), float(pos0[1][:-1])
            for ii in range(N_microdither):
                tmp1 = np.repeat(hdu[det].data[ii], xbin, axis=0).reshape(cube.shape[1],head_det['NAXIS2'])
                tmp2 = np.repeat(tmp1, ybin, axis=1).reshape(cube.shape[1],cube.shape[2])
                this_dither = tmp2 / N_microdither  # to reserve the total number of counts
                # shift the data
                this_pos = head_det['MDCOORD{:1d}'.format(ii+1)].split(',')
                x_this, y_this = float(this_pos[0][1:]), float(this_pos[1][:-1])
                # ToDo: Not sure whether the following is fixed or depends on CD1_1 and CD2_2
                #       I assume this is correct since it moves on pixel basis but not WCS.
                x_shift, y_shift = 2*(x_this-x_pos0), 2*(y_this-y_pos0)
                this_dither = np.roll(this_dither,int(x_shift),axis=1)
                this_dither = np.roll(this_dither,int(y_shift),axis=0)
                cube[ii, :, :] = this_dither - np.median(this_dither[:7,:7])
                cube_flag[ii, :, :] = (this_dither==0.) ## True for pixels=0
            #data_flag = np.sum(cube_flag, axis=0)
            #data = np.median(cube, axis=0)
            #data[data_flag.astype('bool')] = 0 ## make pixels affected by bad pixels to be zero
            #from IPython import embed
            #embed()
            _, data, _ = sigma_clipped_stats(cube, cube_flag, sigma=3, maxiters=1,
                                             cenfunc='median', stdfunc='std', axis=0)
            data[np.isnan(data)] = 0
            x1, x2, y1, y2 = x1*xbin, x2*xbin, y1*ybin, y2*ybin
            detector_par['platescale'] = detector_par['platescale'] / xbin
        else:
            msgs.error('Microdither with {:} points is not supported yet.'.format(N_microdither))

        # Final data
        array = data[x1:x2,y1:y2]

        # datasec_img and oscansec_img
        rawdatasec_img = np.ones_like(array) #* detector_par['gain'][0]
        oscansec_img = np.ones_like(array) #* detector_par['ronoise'][0]

        # Need the exposure time
        exptime = head_det[self.meta['exptime']['card']]

        # Return, transposing array back to orient the overscan properly
        return detector_par, array, head, exptime, rawdatasec_img, oscansec_img
