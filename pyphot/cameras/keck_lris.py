"""
Module for Keck/LRIS

Modified from PyPeIt.
"""
import os
import glob,gc
import numpy as np

import json, gzip
from pkg_resources import resource_filename

from astropy import wcs
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from pyphot import msgs
from pyphot import parse
from pyphot import procimg
from pyphot import telescopes
from pyphot.par import framematch
from pyphot.cameras import camera


class KeckLRISCamera(camera.Camera):
    """
    Child to handle Keck/LRIS specific code
    """
    ndet = 2
    name = 'keck_lris'
    telescope = telescopes.KeckTelescopePar()
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
        self.meta['target'] = dict(ext=0, card='TARGNAME')
        #self.meta['filter'] = dict(ext=0, card='FILTER') # see below
        self.meta['binning'] = dict(ext=0, card='BINNING')

        self.meta['mjd'] = dict(ext=0, card='MJD-OBS')
        self.meta['exptime'] = dict(ext=0, card='ELAPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='OBJECT')
        #self.meta['flamp1'] = dict(ext=0, card='FLAMP1')
        #self.meta['flamp2'] = dict(ext=0, card='FLAMP2')

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
            return framematch.check_frame_exptime(fitstbl['exptime'], [-0.01, 0.01])
        if ftype in ['pixelflat', 'illumflat']:
            return good_exp #& (fitstbl['idname'] == 'flat') & flats
        if ftype in ['science','supersky']:
            return good_exp #& (fitstbl['idname'] == 'object')
        if ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'dark')
        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

class KeckLRISBCamera(KeckLRISCamera):
    """
    Child to handle LBC_B specific code
    """
    name = 'keck_lris_blue'
    camera = 'LRIS BLUE'
    supported = True
    comment = 'LRIS blue camera'

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
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict1 = dict(
            binning         = binning,
            det             = 1,
            dataext         = 1,
            specaxis        = 0,
            specflip        = False,
            spatflip        = False,
            platescale      = 0.135,
            darkcurr        = 0.0,
            saturation      = 65535.,
            nonlinear       = 0.95,
            mincounts       = -1e10,
            numamplifiers   = 2,
            #gain            = np.atleast_1d([1.55, 1.56]),
            gain            = np.atleast_1d([1.641, 1.623]), ## New gain measured by FW using flat observed on the night of March, 2022.
            #gain            = np.atleast_1d([1.758, 1.739]), ## New gain measured by FW using flat observed on the night of Jan 27, 2022.
            #gain            = np.atleast_1d([1.55, 1.533]), ## New gain measured by FW using flat observed on the night of Jan 27, 2022.
            ronoise         = np.atleast_1d([3.9, 4.2]),
            )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            dataext=2,
            #gain=np.atleast_1d([1.63, 1.70]),
            gain=np.atleast_1d([1.667, 1.697]), ## New gain measured by FW using flat observed on the night of March, 2022.
            #gain=np.atleast_1d([1.63, 1.66]), ## New gain measured by FW using flat observed on the night of Jan 27, 2022.
            ronoise=np.atleast_1d([3.6, 3.6])
        ))

        detectors = [detector_dict1, detector_dict2]

        detector = detectors[det-1]

        # Deal with number of amps
        namps = hdu[0].header['NUMAMPS']
        # The website does not give values for single amp per detector so we take the mean
        #   of the values provided
        if namps == 2 or (namps==4 and len(hdu)==3):  # Longslit readout mode is the latter.  This is a hack..
            detector.numamplifiers = 1
            detector.gain = np.atleast_1d(np.mean(detector.gain))
            detector.ronoise = np.atleast_1d(np.mean(detector.ronoise))
        elif namps == 4:
            pass
        else:
            msgs.error("Did not see this namps coming..")

        # Return
        return detector

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

        # Vignetting
        par['scienceframe']['process']['mask_vig'] = True
        par['scienceframe']['process']['minimum_vig'] = 0.2
        par['scienceframe']['process']['conv'] = 'sex995' # Used for bright star mask

        # Background type for image processing
        par['scienceframe']['process']['use_medsky'] = True
        #par['scienceframe']['process']['back_type'] = 'GlobalMedian'
        #par['scienceframe']['process']['back_type'] = 'median'

        # cosmic ray rejection
        par['scienceframe']['process']['sigclip'] = 5.0
        par['scienceframe']['process']['objlim'] = 2.0

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [10, None]
        par['calibrations']['pixelflatframe']['exprng'] = [0.1, 10]
        par['calibrations']['illumflatframe']['exprng'] = [0.1, 10]
        par['calibrations']['superskyframe']['exprng'] = [10, None]
        par['calibrations']['superskyframe']['process']['window_size'] = [101, 101]
        ## We use dome flat for the pixel flat and thus do not need mask bright stars.
        par['calibrations']['pixelflatframe']['process']['mask_brightstar']=False
        par['calibrations']['illumflatframe']['process']['mask_brightstar']=False

        # astrometry
        par['postproc']['astrometry']['mosaic'] = True
        par['postproc']['astrometry']['mosaic_type'] = 'UNCHANGED'
        par['postproc']['astrometry']['astref_catalog'] = 'PANSTARRS-1'
        par['postproc']['astrometry']['astrefmag_limits'] = [17, 22.5]
        par['postproc']['astrometry']['astrefsn_limits'] = [5, 10.0]
        par['postproc']['astrometry']['posangle_maxerr'] = 10.0
        par['postproc']['astrometry']['position_maxerr'] = 1.0
        par['postproc']['astrometry']['pixscale_maxerr'] = 1.1
        par['postproc']['astrometry']['detect_thresh'] = 10  # increasing this can improve the solution if your image is deep
        par['postproc']['astrometry']['analysis_thresh'] = 10
        par['postproc']['astrometry']['detect_minarea'] = 11
        par['postproc']['astrometry']['crossid_radius'] = 2.0

        par['postproc']['detection']['conv'] = 'sex995' # Should be set for 1x1 binning
        # photometry
        par['postproc']['photometry']['external_flag'] = False
        par['postproc']['photometry']['nstar_min'] = 5

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
            :class:`~pyphot.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        par = super().config_specific_par(scifile, inp_par=inp_par)

        # https://sites.google.com/a/lbto.org/lbc/phase-ii-guidelines/sensitivities
        # LBT gives ZP for 1 ADU/s, PyPhot use 1 e/s
        # ZP_ADU = ZP_e - 2.5*np.log10(gain)
        if self.get_meta_value(scifile, 'filter') == 'u':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 27.03 #ToDo: measure it
            # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
            par['postproc']['photometry']['coefficients'] = [0.,0.,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.48 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'B':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['zpt'] = 28.73 #ToDo: measure it
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.22 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'V':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['zpt'] = 28.93 #ToDo: measure it
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.15 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'G':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['coefficients'] = [0.016, 0.160, 0.]
            par['postproc']['photometry']['zpt'] = 28.62 # Estimated from data obtained on Jan 27, 2022
            par['postproc']['photometry']['coeff_airmass'] = 0.17 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)

        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PyPhot``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        super().init_meta()
        # Add the name of the dispersing element
        self.meta['filter'] = dict(ext=0, card='BLUFILT')

    def get_rawimage(self, raw_file, det):
        """
        Read raw images and generate a few other bits and pieces
        that are key for image processing.

        Based on readmhdufits.pro

        Parameters
        ----------
        raw_file : :obj:`str`
            File to read
        det : :obj:`int`
            1-indexed detector to read

        Returns
        -------
        detector_par : dict
            Detector metadata parameters.
        raw_img : `numpy.ndarray`_
            Raw image for this detector.
        hdu : `astropy.io.fits.HDUList`_
            Opened fits file
        exptime : :obj:`float`
            Exposure time read from the file header
        rawdatasec_img : `numpy.ndarray`_
            Data (Science) section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        oscansec_img : `numpy.ndarray`_
            Overscan section of the detector as provided by setting the
            (1-indexed) number of the amplifier used to read each detector
            pixel. Pixels unassociated with any amplifier are set to 0.
        """
        # Check for file; allow for extra .gz, etc. suffix
        fil = glob.glob(raw_file + '*')
        if len(fil) != 1:
            msgs.error("Found {:d} files matching {:s}".format(len(fil)))

        # Read
        msgs.info("Reading LRIS file: {:s}".format(fil[0]))
        hdu = fits.open(fil[0])
        head = hdu[0].header

        detector_par = self.get_detector_par(hdu, det if det is not None else 1)

        # Get post, pre-pix values
        precol = head['PRECOL']
        postpix = head['POSTPIX']
        preline = head['PRELINE']
        postline = head['POSTLINE']

        # get the x and y binning factors...
        binning = head['BINNING']
        xbin, ybin = [int(ibin) for ibin in binning.split(',')]

        # First read over the header info to determine the size of the output array...
        extensions = []
        for kk, ihdu in enumerate(hdu):
            if 'VidInp' in ihdu.name:
                extensions.append(kk)
        n_ext = len(extensions)
        xcol = []
        xmax = 0
        ymax = 0
        xmin = 10000
        ymin = 10000

        for i in extensions:
            theader = hdu[i].header
            detsec = theader['DETSEC']
            if detsec != '0':
                # parse the DETSEC keyword to determine the size of the array.
                x1, x2, y1, y2 = np.array(parse.load_sections(detsec, fmt_iraf=False)).flatten()

                # find the range of detector space occupied by the data
                # [xmin:xmax,ymin:ymax]
                xt = max(x2, x1)
                xmax = max(xt, xmax)
                yt = max(y2, y1)
                ymax = max(yt, ymax)

                # find the min size of the array
                xt = min(x1, x2)
                xmin = min(xmin, xt)
                yt = min(y1, y2)
                ymin = min(ymin, yt)
                # Save
                xcol.append(xt)

        # determine the output array size...
        nx = xmax - xmin + 1
        ny = ymax - ymin + 1

        # change size for binning...
        nx = nx // xbin
        ny = ny // ybin

        # Update PRECOL and POSTPIX
        precol = precol // xbin
        postpix = postpix // xbin

        # Deal with detectors
        if det in [1, 2]:
            nx = nx // 2
            n_ext = n_ext // 2
            det_idx = np.arange(n_ext, dtype=np.int) + (det - 1) * n_ext
        elif det is None:
            det_idx = np.arange(n_ext).astype(int)
        else:
            raise ValueError('Bad value for det')

        # change size for pre/postscan...
        nx += n_ext * (precol + postpix)
        ny += preline + postline

        # allocate output arrays...
        array = np.zeros((nx, ny))
        order = np.argsort(np.array(xcol))
        rawdatasec_img = np.zeros_like(array, dtype=int)
        oscansec_img = np.zeros_like(array, dtype=int)
        #gainimage = np.zeros_like(array)
        #rnimage = np.zeros_like(array)

        # insert extensions into master image...
        for amp, i in enumerate(order[det_idx]):

            # grab complete extension...
            data, predata, postdata, x1, y1 = lris_read_amp(hdu, i + 1)

            # insert predata...
            buf = predata.shape
            nxpre = buf[0]
            xs = amp * precol
            xe = xs + nxpre
            # predata (ignored)
            array[xs:xe, :] = predata

            # insert data...
            buf = data.shape
            nxdata = buf[0]
            xs = n_ext * precol + amp * nxdata  # (x1-xmin)/xbin
            xe = xs + nxdata
            array[xs:xe, :] = data
            rawdatasec_img[xs:xe, preline:ny-postline] = amp+1
            #gainimage[xs:xe, preline:ny-postline] = detector_par['gain'][amp]
            #rnimage[xs:xe, preline:ny-postline] = detector_par['ronoise'][amp]

            # ; insert postdata...
            buf = postdata.shape
            nxpost = buf[0]
            xs = nx - n_ext * postpix + amp * postpix
            xe = xs + nxpost
            array[xs:xe, :] = postdata
            oscansec_img[xs:xe, preline:ny-postline] = amp+1

        # Need the exposure time
        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]

        # Trim the overscan region and get wcs corrrect!
        mask = rawdatasec_img==0.
        raw_img_trim =  procimg.trim_frame(array, rawdatasec_img < 0.1)
        datasec_img_trim =  procimg.trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
        oscansec_img_trim =  procimg.trim_frame(oscansec_img, rawdatasec_img < 0.1)

        # Build WCS
        ##ToDo: This might be not true for other data, i.e. might depends on rotator!
        head['EXPTIME'] = (exptime, 'Exposure time') # This is required
        c = SkyCoord(head['RA'], head['DEC'], frame="icrs", unit=(u.hourangle, u.deg))

        xbin, ybin = int(detector_par['binning'].split(',')[0]), int(detector_par['binning'].split(',')[1])
        # header
        w = wcs.WCS(naxis=2)
        if det==1:
            w.wcs.crpix = array.shape[1]/2, array.shape[0] - 100//ybin
        elif det==2:
            w.wcs.crpix = array.shape[1]/2, 0
        w.wcs.cdelt = [-detector_par['platescale'] / 3600. / xbin, detector_par['platescale'] / 3600. / ybin]
        # LRIS red pointing is about 1.3 arcmin away for the norminal imaging orientation
        w.wcs.crval = [c.ra.value-float(head['DRA'])-0.35/60., c.dec.value-float(head['DDEC'])-1.25/60.]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header_wcs = w.to_header()
        # remove old values
        for ikey in ['CD1_1','CD1_2','CD2_1','CD2_2','CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2']:
            if ikey in head.keys():
                head.remove(ikey, remove_all=True)
        # append new keys
        for i in range(len(header_wcs)):
            head.append(header_wcs.cards[i]) #

        #hdu = fits.PrimaryHDU(raw_img_trim, header=head)
        #hdu.writeto('testb.fits', overwrite=True)

        # Return
        return detector_par, array, head, exptime, rawdatasec_img, oscansec_img

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

        # Only defined for det=1
        if det == 1:
            msgs.info("Using hard-coded BPM for det=1 on LRISb")
            bpm_img[:,:3] = 1

        return bpm_img

class KeckLRISRCamera(KeckLRISCamera):
    """
    Child to handle Keck LRIS Red specific code
    """
    ndet = 1
    name = 'keck_lris_red'
    camera = 'LRIS RED'
    supported = True
    comment = 'LRIS red camera, new Mark4 detector, circa Spring 2021'

    def init_meta(self):
        super().init_meta()
        # Over-ride a pair
        self.meta['mjd'] = dict(ext=0, card='MJD')
        self.meta['exptime'] = dict(ext=0, card='TELAPSE')
        self.meta['filter'] = dict(ext=0, card='REDFILT')

    def get_detector_par(self, hdu, det):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            dict
            Object with the detector metadata.
        """
        # Binning
        # TODO: Could this be detector dependent?
        binning = '1,1' if hdu is None else self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        detector_dict1 = dict(
            binning=binning,
            det=1,
            dataext=0,
            specaxis=0,
            specflip=True,
            spatflip=False,
            platescale=0.123,  # From the web page
            darkcurr=0.0,
            saturation=65535.,
            nonlinear=0.90,
            mincounts=-1e10,
            numamplifiers=2,  # These are defaults but can modify below
            gain=np.atleast_1d([1.61, 1.67*0.959]), # Assumes AMPMODE=HSPLIT,VUP;  Corrected by JXP using 2x1 binned flats
            ronoise=np.atleast_1d([3.65, 3.52]),
        )

        # Header
        # Date of Mark4 installation
        t2021_upgrade = Time("2021-04-15", format='isot')
        # TODO -- Update with the date we transitioned to the correct ones
        t_gdhead = Time("2023-01-01", format='isot')
        date = Time(hdu[0].header['MJD'], format='mjd')

        if date < t2021_upgrade:
            msgs.error("This is not the Mark4 detector.  Use a different keck_lris_red spectrograph")

        # Deal with the intermediate headers
        if date < t_gdhead:
            amp_mode = hdu[0].header['AMPMODE']
            msgs.info("AMPMODE = {:s}".format(amp_mode))
            # Load up translation dict
            ampmode_translate_file = os.path.join(
                resource_filename('pyphot', 'data'), 'cameras',
                'keck_lris_red_mark4', 'dict_for_ampmode.json')
            ampmode_translate_dict = loadjson(ampmode_translate_file)
            # Load up the corrected header
            swap_binning = f"{binning[-1]},{binning[0]}"  # LRIS convention is oppopsite ours
            header_file = os.path.join(
                resource_filename('pyphot', 'data'), 'cameras',
                'keck_lris_red_mark4',
                f'header{ampmode_translate_dict[amp_mode]}_{swap_binning.replace(",","_")}.fits')
            correct_header = fits.getheader(header_file)
        else:
            correct_header = hdu[0].header

        # Deal with number of amps
        detector_dict1['numamplifiers'] = correct_header['TAPLINES']

        # The website does not give values for single amp per detector so we take the mean
        #   of the values provided
        if detector_dict1['numamplifiers'] == 2:
            pass
        elif detector_dict1['numamplifiers'] == 4:
            # From the web page on 2021-10-04 (L1, L2, U1, U2)
            ## Corrected by JXP and SS using chk_lris_mark4_gain.py in the DevSuite
            #detector_dict1['gain'] = np.atleast_1d([1.710,
            #                                        1.64 * 1.0245,  # L2
            #                                        1.61 * 1.0185,  # U1
            #                                        1.67 * 1.0052])  # U2
            detector_dict1['ronoise'] = np.atleast_1d([3.64, 3.45, 3.65, 3.52])
            ## New gain measured by FW using flat observed on the night of Jan 27, 2022.
            detector_dict1['gain'] = np.atleast_1d([1.71, 1.68018, 1.6427817073170732, 1.678684])
        else:
            msgs.error("Did not see this namps coming..")

        detector_dict1['datasec'] = []
        detector_dict1['oscansec'] = []

        # Parse which AMPS were used
        used_amps = []
        for amp in range(4):
            if f'AMPNM{amp}' in correct_header.keys():
                used_amps.append(amp)
        # Check
        assert detector_dict1['numamplifiers'] == len(used_amps)

        # Reverse engenieering to translate LRIS DSEC, BSEC
        #  into ones friendly for PypeIt...
        binspec = int(binning[0])
        binspatial = int(binning[-1])

        for iamp in used_amps:
            # These are column, row
            dsecs = correct_header[f'DSEC{iamp}'].split(',')
            d_rows = [int(item) for item in dsecs[1][:-1].split(':')]
            d_cols = [int(item) for item in dsecs[0][1:].split(':')]
            bsecs = correct_header[f'BSEC{iamp}'].split(',')
            o_rows = [int(item) for item in bsecs[1][:-1].split(':')]
            o_cols = [int(item) for item in bsecs[0][1:].split(':')]

            # Deal with binning (heaven help me!!)
            d_rows = [str(item * binspec) if item != 1 else str(item) for item in d_rows]
            o_rows = [str(item * binspec) if item != 1 else str(item) for item in o_rows]
            d_cols = [str(item * binspatial) if item != 1 else str(item) for item in d_cols]
            o_cols = [str(item * binspatial) if item != 1 else str(item) for item in o_cols]

            # These are now row, column
            #  And they need to be native!!  i.e. no binning accounted for..
            detector_dict1['datasec'] += [f"[{':'.join(d_rows)},{':'.join(d_cols)}]"]
            detector_dict1['oscansec'] += [f"[{':'.join(o_rows)},{':'.join(o_cols)}]"]

        detector_dict1['datasec'] = np.array(detector_dict1['datasec'])
        detector_dict1['oscansec'] = np.array(detector_dict1['oscansec'])

        # Return
        return detector_dict1

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
        par['scienceframe']['process']['use_fringe'] = True

        # Vignetting
        par['scienceframe']['process']['mask_vig'] = True
        par['scienceframe']['process']['minimum_vig'] = 0.1

        # Background type for image processing
        par['scienceframe']['process']['use_medsky'] = True
        #par['scienceframe']['process']['back_type'] = 'GlobalMedian'
        #par['scienceframe']['process']['back_type'] = 'median'
        par['scienceframe']['process']['conv'] = 'sex995' # Used for bright star mask

        # cosmic ray rejection
        #ToDo: Need to tweak these parameters for LRIS red.
        par['scienceframe']['process']['lamaxiter'] = 1
        par['scienceframe']['process']['cr_threshold'] = 5
        par['scienceframe']['process']['neighbor_threshold'] = 2
        par['scienceframe']['process']['contrast'] = 1.
        par['scienceframe']['process']['grow'] = 1.5

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['darkframe']['exprng'] = [None, None]
        par['scienceframe']['exprng'] = [10, None]
        par['calibrations']['pixelflatframe']['exprng'] = [0.1, 10]
        par['calibrations']['illumflatframe']['exprng'] = [0.1, 10]
        par['calibrations']['superskyframe']['exprng'] = [10, None]
        par['calibrations']['superskyframe']['process']['window_size'] = [101, 101]
        ## We use dome flat for the pixel flat and thus do not need mask bright stars.
        par['calibrations']['pixelflatframe']['process']['mask_brightstar']=False
        par['calibrations']['illumflatframe']['process']['mask_brightstar']=False

        # astrometry
        par['postproc']['astrometry']['astref_catalog'] = 'PANSTARRS-1'
        par['postproc']['astrometry']['astrefmag_limits'] = [17, 22.5]
        par['postproc']['astrometry']['astrefsn_limits'] = [7, 10.0]
        par['postproc']['astrometry']['posangle_maxerr'] = 10.0
        par['postproc']['astrometry']['position_maxerr'] = 1.0
        par['postproc']['astrometry']['pixscale_maxerr'] = 1.1
        par['postproc']['astrometry']['detect_thresh'] = 10  # increasing this can improve the solution if your image is deep
        par['postproc']['astrometry']['analysis_thresh'] = 10
        par['postproc']['astrometry']['detect_minarea'] = 13
        par['postproc']['astrometry']['crossid_radius'] = 2.0

        par['postproc']['detection']['conv'] = 'sex995' # Should be set for 1x1 binning
        # photometry
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

        # https://sites.google.com/a/lbto.org/lbc/phase-ii-guidelines/sensitivities
        # The coefficients are from lbt_lbc
        # LBT gives ZP for 1 ADU/s, PyPhot use 1 e/s
        # ZP_ADU = ZP_e - 2.5*np.log10(gain)
        # Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2
        if self.get_meta_value(scifile, 'filter') == 'B':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['zpt'] = 28.73 #ToDo: measure it
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.22 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'V':
            par['postproc']['photometry']['photref_catalog'] = 'Sloan'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 28.77 #ToDo: measure it
            par['postproc']['photometry']['coefficients'] = [0.,0.,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.16 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'R':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['coefficients'] = [-0.010,-0.218, 0.]
            par['postproc']['photometry']['zpt'] = 28.69 #ToDo: measure it
            par['postproc']['photometry']['coeff_airmass'] = 0.13 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)
        elif self.get_meta_value(scifile, 'filter') == 'I':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'i'
            par['postproc']['photometry']['secondary'] = 'z'
            par['postproc']['photometry']['coefficients'] = [-0.003,-0.411,0.]
            par['postproc']['photometry']['zpt'] = 28.03 # Measured from J0252 data observed on Jan 27, 2022
            par['postproc']['photometry']['coeff_airmass'] = 0.04 # extinction, i.e. mag_real=mag_obs-coeff_airmass*(airmass-1)

        return par

    def get_rawimage(self, raw_file, det):

        detector, raw_img, headarr, exptime, datasec_img, oscansec_img = camera.Camera.get_rawimage(self, raw_file, det)

        # ToDo: should put the namp and trim in procimg.py
        # trim the the bias sections
        mask = datasec_img==0.
        raw_img_trim =  procimg.trim_frame(raw_img, datasec_img < 0.1)
        datasec_img_trim =  procimg.trim_frame(datasec_img, datasec_img < 0.1)
        oscansec_img_trim =  procimg.trim_frame(oscansec_img, datasec_img < 0.1)
        #raw_img_trim = raw_img[np.logical_not(np.all(mask, axis=1)), :][:, np.logical_not(np.all(mask, axis=0))]
        #datasec_img_trim = datasec_img[np.logical_not(np.all(mask, axis=1)), :][:, np.logical_not(np.all(mask, axis=0))]
        #oscansec_img_trim = oscansec_img[np.logical_not(np.all(mask, axis=1)), :][:, np.logical_not(np.all(mask, axis=0))]

        # Deal with WCS
        head = headarr[0]
        head['EXPTIME'] = (exptime, 'Exposure time') # This is required
        c = SkyCoord(head['RA'], head['DEC'], frame="icrs", unit=(u.hourangle, u.deg))

        xbin, ybin = int(detector['binning'].split(',')[0]), int(detector['binning'].split(',')[1])
        # header
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = raw_img_trim.shape[1]/2, raw_img_trim.shape[0]/2
        w.wcs.cdelt = [detector['platescale'] / 3600. / xbin, -detector['platescale'] / 3600. / ybin]
        # LRIS red pointing is about 1.3 arcmin away for the norminal imaging orientation
        w.wcs.crval = [c.ra.value-head['DRA'], c.dec.value-head['DDEC']-1.3/60.]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header_wcs = w.to_header()
        # remove old values
        for ikey in ['CD1_1','CD1_2','CD2_1','CD2_2','CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2']:
            if ikey in head.keys():
                head.remove(ikey, remove_all=True)
        # append new keys
        for i in range(len(header_wcs)):
            head.append(header_wcs.cards[i]) #

        ##ToDo: This might be not true for other data, i.e. might depends on rotator!
        # Need rotate in order to get the direction of WCS correct.
        #hdu = fits.PrimaryHDU(raw_img_trim.T, header=head)
        #hdu.writeto('testr.fits', overwrite=True)
        return detector, raw_img_trim.T, head, exptime, datasec_img_trim.T, oscansec_img_trim.T

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

        return bpm_img


def lris_read_amp(inp, ext):
    """
    Read one amplifier of an LRIS multi-extension FITS image

    Args:
        inp (str, astropy.io.fits.HDUList):
            filename or HDUList
        ext (int):
            Extension index

    Returns:
        tuple:
            data
            predata
            postdata
            x1
            y1

    """
    # Parse input
    if isinstance(inp, str):
        hdu = io.fits_open(inp)
    else:
        hdu = inp
    # Count the number of extensions
    n_ext = np.sum(['VidInp' in h.name for h in hdu])

    # Get the pre and post pix values
    # for LRIS red POSTLINE = 20, POSTPIX = 80, PRELINE = 0, PRECOL = 12
    head0 = hdu[0].header
    precol = head0['precol']
    postpix = head0['postpix']

    # Deal with binning
    binning = head0['BINNING']
    xbin, ybin = [int(ibin) for ibin in binning.split(',')]
    precol = precol//xbin
    postpix = postpix//xbin

    # get entire extension...
    temp = hdu[ext].data.transpose() # Silly Python nrow,ncol formatting
    tsize = temp.shape
    nxt = tsize[0]

    # parse the DETSEC keyword to determine the size of the array.
    header = hdu[ext].header
    detsec = header['DETSEC']
    x1, x2, y1, y2 = np.array(parse.load_sections(detsec, fmt_iraf=False)).flatten()

    # parse the DATASEC keyword to determine the size of the science region (unbinned)
    datasec = header['DATASEC']
    xdata1, xdata2, ydata1, ydata2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()
    # grab the components...
    predata = temp[0:precol, :]
    # datasec appears to have the x value for the keywords that are zero
    # based. This is only true in the image header extensions
    # not true in the main header.  They also appear inconsistent between
    # LRISr and LRISb!
    #data     = temp[xdata1-1:xdata2-1,*]
    #data = temp[xdata1:xdata2+1, :]
    if (xdata1-1) != precol:
        msgs.error("Something wrong in LRIS datasec or precol")
    xshape = 1024 // xbin * (4//n_ext)  # Allow for single amp
    if (xshape+precol+postpix) != temp.shape[0]:
        msgs.warn("Unexpected size for LRIS detector.  We expect you did some windowing...")
        xshape = temp.shape[0] - precol - postpix
    data = temp[precol:precol+xshape,:]
    postdata = temp[nxt-postpix:nxt, :]

    # flip in X as needed...
    if x1 > x2:
        xt = x2
        x2 = x1
        x1 = xt
        data = np.flipud(data)

    # flip in Y as needed...
    if y1 > y2:
        yt = y2
        y2 = y1
        y1 = yt
        data = np.fliplr(data)
        predata = np.fliplr(predata)
        postdata = np.fliplr(postdata)

    return data, predata, postdata, x1, y1

def loadjson(filename):
    """ Load a python object saved with savejson."""
    if filename.endswith('.gz'):
        with gzip.open(filename, "rb") as f:
            obj = json.loads(f.read().decode("ascii"))
    else:
        with open(filename, 'rt') as fh:
            obj = json.load(fh)

    return obj
