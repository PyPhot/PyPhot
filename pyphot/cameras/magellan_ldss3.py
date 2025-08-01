"""
Module for Magellan LDSS3

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

class MagellanLDSS3Camera(camera.Camera):
    """
    Child to handle Magellan/LDSS3 specific code
    """
    ndet = 1
    name = 'magellan_ldss3'
    telescope = telescopes.MagellanTelescopePar()
    supported = True

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
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')

        detector_dict = dict(
                            binning         = binning,
                            det             = 1,
                            dataext         = 1,
                            specaxis        = 0,
                            specflip        = False,
                            spatflip        = False,
                            xgap            = 0.,
                            ygap            = 0.,
                            ysize           = 1.,
                            platescale      = 0.188,
                            darkcurr        = 25.0,
                            saturation      = 205000.,
                            nonlinear       = 0.85,
                            mincounts       = -1e10,
                            numamplifiers   = 2,
                            gain            = np.atleast_1d([1.65,1.47]),
                            ronoise         = np.atleast_1d([4.67,5.06]),
                            )

        # Instantiate
        return detector_dict

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

        self.meta['mjd'] = dict(ext=0, card='JD')
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')
        self.meta['amp'] = dict(ext=0, card='OPAMP')

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
        if meta_key == 'binning':
            binspatial, binspec = parse.parse_binning(headarr[1]['BINNING'])
            binning = parse.binning2string(binspec, binspatial)
            return binning
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
            if fitstbl[i]['target'] is not None:
                if 'flat' in fitstbl[i]['target'].lower():
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
        # Set group to False given the uncertain RA/DEC in fits header (i.e., the RA/DEC is not updated
        # the operator doing some manual telescope offset for getting a good guide star).
        par['postproc']['astrometry']['group'] = False
        par['postproc']['astrometry']['mosaic'] = False
        par['postproc']['astrometry']['mosaic_type'] = 'LOOSE'
        par['postproc']['astrometry']['astref_catalog'] = 'GAIA-EDR3'
        par['postproc']['astrometry']['astrefmag_limits'] = [18, 21]
        par['postproc']['astrometry']['detect_thresh'] = 10
        par['postproc']['astrometry']['analysis_thresh'] = 10
        par['postproc']['astrometry']['detect_minarea'] = 5
        par['postproc']['astrometry']['crossid_radius'] = 2

        # Set the default exposure time ranges for the frame typing
        par['calibrations']['superskyframe']['exprng'] = [10, None]
        par['calibrations']['fringeframe']['exprng'] = [10, None]
        par['scienceframe']['exprng'] = [30, None]

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

        if self.get_meta_value(scifile, 'filter') == 'u_Sloan':
            par['postproc']['photometry']['photref_catalog'] = 'SDSS'
            par['postproc']['photometry']['primary'] = 'u'
            par['postproc']['photometry']['secondary'] = 'g'
            par['postproc']['photometry']['zpt'] = 23.55
            par['postproc']['photometry']['coefficients'] = [0., 0., 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.48
        elif self.get_meta_value(scifile, 'filter') == 'g_Sloan':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'g'
            par['postproc']['photometry']['secondary'] = 'r'
            par['postproc']['photometry']['zpt'] = 27.65
            par['postproc']['photometry']['coefficients'] = [0.016, 0.160, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.18
        elif self.get_meta_value(scifile, 'filter') == 'r_Sloan':
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'r'
            par['postproc']['photometry']['secondary'] = 'i'
            par['postproc']['photometry']['zpt'] = 27.82
            par['postproc']['photometry']['coefficients'] = [0.002, 0.024, 0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.10
        elif self.get_meta_value(scifile, 'filter') == 'i_Sloan':
            # There is no need to subtract fringing for i-band and other bands
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'i'
            par['postproc']['photometry']['secondary'] = 'z'
            par['postproc']['photometry']['zpt'] = 27.81
            par['postproc']['photometry']['coefficients'] = [0.,0.058,0.]
            par['postproc']['photometry']['coeff_airmass'] = 0.04
        elif self.get_meta_value(scifile, 'filter') == 'z_Sloan':
            par['scienceframe']['process']['use_fringe'] = True # Subtract fringing if using z-band
            par['postproc']['photometry']['photref_catalog'] = 'Panstarrs'
            par['postproc']['photometry']['primary'] = 'z'
            par['postproc']['photometry']['secondary'] = 'y'
            par['postproc']['photometry']['zpt'] = 27.81
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
        # Get the empty bpm: force is always True
        bpm_img = self.empty_bpm(filename, det, shape=shape)

        # Fill in bad pixels if a master bias frame is provided
        if msbias is not None:
            return self.bpm_frombias(msbias, det, bpm_img)

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

        raw_file2 = raw_file.replace('c1.fits','c2.fits')

        # Check for file; allow for extra .gz, etc. suffix
        fil = glob.glob(raw_file + '*')
        if len(fil) != 1:
            msgs.error("Found {:d} files matching {:s}".format(len(fil)))

        # Check for file; allow for extra .gz, etc. suffix
        fil2 = glob.glob(raw_file2 + '*')
        if len(fil2) != 1:
            msgs.warn("Found {:d} files matching {:s}".format(len(fil2), raw_file))
            msgs.warn("Proceeding without the second amplifier")
            have_file2 = False
        else:
            have_file2 = True

        # detector par
        hdu = fits.open(fil[0])
        head1 = fits.getheader(fil[0], 0)
        detector_par = self.get_detector_par(hdu, det if det is None else 1)

        data1, overscan1, datasec1, biassec1, x1_1, x2_1, nxb1 = ldss3_read_amp(fil[0])
        if have_file2:
            data2, overscan2, datasec2, biassec2, x1_2, x2_2, nxb2 = ldss3_read_amp(fil2[0])
        else:
            nxb2 = nxb1
            x2_2 = x2_1

        nx, ny = x2_1 + x2_2 + nxb1 + nxb2, data1.shape[1]

        # allocate output array...
        array = np.zeros((nx, ny))
        rawdatasec_img = np.zeros_like(array, dtype=int)
        oscansec_img = np.zeros_like(array, dtype=int)

        ## For amplifier 1
        array[:nxb1,:] = overscan1
        array[nxb1:x2_1+nxb1,:] = data1
        rawdatasec_img[nxb1:x2_1+nxb1, :] = 1
        oscansec_img[:nxb1,:] = 1 # exclude the first pixel since it always has problem

        ## For amplifier 2
        if have_file2:
            array[x2_1+nxb1+x2_2:x2_1+nxb1+x2_2+nxb2,:] = np.flipud(overscan2)
            array[x2_1+nxb1:x2_1+nxb1+x2_2,:] = np.flipud(data2)
        rawdatasec_img[x2_1+nxb1:x2_1+nxb1+x2_2, :] = 2
        oscansec_img[x2_1+nxb1+x2_2:x2_1+nxb1+x2_2+nxb2,:] = 2 # exclude the first pixel since it always has problem

        # Transpose now (helps with debuggin)
        array = array.T
        rawdatasec_img = rawdatasec_img.T
        oscansec_img = oscansec_img.T

        # Need the exposure time
        exptime = hdu[self.meta['exptime']['ext']].header[self.meta['exptime']['card']]

        ## ToDo: Need to check binned data
        xbin, ybin = parse.parse_binning(detector_par['binning'])

        # Update header with an initial WCS information.
        crpix1 = nx/xbin/2
        crpix2 = ny/xbin/2
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [crpix1, crpix2]
        cdelt1 = -detector_par['platescale'] * xbin / 3600.
        cdelt2 = detector_par['platescale'] * ybin / 3600.
        w.wcs.crval = [head1['RA-D']-38.5/np.cos(head1['DEC-D']/180.* np.pi) / 3600.,
                       head1['DEC-D']-10.5/3600]

        #if (head1['DEC-D']<-29.01597) & (head1['ROTANGLE']==43.85):
        #    cdelt1 = detector_par['platescale'] * xbin / 3600.
        #    cdelt2 = detector_par['platescale'] * ybin / 3600.
        #    w.wcs.crval = [head1['RA-D']-head1['CHOFFX']/np.cos(head1['DEC-D']/180.* np.pi) / 3600.,
        #                   head1['DEC-D']+head1['CHOFFY']/3600.]
        #else:
        #    #cdelt1 = -detector_par['platescale'] * xbin / 3600.
        #    #cdelt2 = -detector_par['platescale'] * ybin / 3600.
        #    #w.wcs.crval = [head1['RA-D']+head1['CHOFFX']/np.cos(head1['DEC-D']/180.*np.pi)/3600.,
        #    #               head1['DEC-D']-head1['CHOFFY']/3600.]
        #    cdelt1 = -detector_par['platescale'] * xbin / 3600.
        #    cdelt2 = detector_par['platescale'] * ybin / 3600.
        #    w.wcs.crval = [head1['RA-D']-head1['CHOFFX']/np.cos(head1['DEC-D']/180.* np.pi) / 3600.,
        #                   head1['DEC-D']+head1['CHOFFY']/3600.]

        w.wcs.cdelt = np.array([cdelt1, cdelt2])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header_wcs = w.to_header()
        for i in range(len(header_wcs)):
            head1.append(header_wcs.cards[i])

        #
        head1.pop('DATASEC')
        head1.pop('BIASSEC')
        #hdu = fits.PrimaryHDU(header=header_wcs, data=array)
        #hdu.writeto('test.fits', overwrite=True)

        # Return, transposing array back to orient the overscan properly
        return detector_par,array, head1, exptime, rawdatasec_img, oscansec_img


def ldss3_read_amp(fil:str):
    """ Read a single amp of LDSS3 data

    Args:
        fil (str): filename

    Returns:
        tuple: data, overscan, datasec, biassec, x1, x2, nxb
    """

    msgs.info("Reading LDSS3 file: {:s}".format(fil))
    hdu = fits.open(fil)
    head1 = hdu[0].header

    # ToDo: Need to check binned data
    # get the x and y binning factors...
    binning = head1['BINNING']
    xbin, ybin = [int(ibin) for ibin in binning.split('x')]

    # First read over the header info to determine the size of the output array...
    datasec = head1['DATASEC']
    x1, x2, y1, y2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()
    biassec = head1['BIASSEC']
    b1, b2, b3, b4 = np.array(parse.load_sections(biassec, fmt_iraf=False)).flatten()
    nxb = b2 - b1 + 1

    # determine the output array size...
    nx = (x2 - x1 + 1) + nxb
    ny = y2 - y1 + 1

    # allocate output array...
    array = hdu[0].data.T[:, :ny] * 1.0
    data = array[:nx-nxb,:]
    overscan = array[nx-nxb:,:]

    return data, overscan, datasec, biassec, x1, x2, nxb

