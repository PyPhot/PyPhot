"""
Modified from PyPeIT :class:`~pypeit.spectrographs.spectrograph.Spectrograph`
class, which is the parent class for all instruments served by ``PypeIt``.

"""

from abc import ABCMeta

import numpy as np

from astropy.io import  fits

from pyphot import msgs
from pyphot import parse
from pyphot import procimg
from pyphot import meta
from pyphot.par import pyphotpar



class Camera:
    """
    Abstract base class for all instrument-specific behavior in ``PyPhot``.

    Attributes:
        rawdatasec_img (`numpy.ndarray`_):
            An image identifying the amplifier that reads each detector
            pixel.
        oscansec_img (`numpy.ndarray`_):
            An image identifying the amplifier that reads each detector
            pixel
        primary_hdrext (:obj:`int`):
            0-indexed number of the extension in the raw frames with the
            primary header data.
        meta (:obj:`dict`):
            Instrument-specific metadata model, linking header information to
            metadata elements required by ``PyPhot``.
    """
    __metaclass__ = ABCMeta

    ndet = None
    """
    Number of detectors for this instrument.
    """

    name = None
    """
    The name of the Camera. See :ref:`instruments` for the currently
    supported camera.
    """

    telescope = None
    """
    Instance of :class:`~pyphot.par.pyphotpar.TelescopePar` providing
    telescope-specific metadata.
    """

    pypeline = 'IR'
    """
    String used to select the general pipeline approach for
    """

    supported = False
    """
    Flag that ``PyPhot`` code base has been sufficiently tested with data
    from this camera that it is officially supported by the development
    team.
    """

    comment = None
    """
    A brief comment or description regarding ``PyPhot`` usage with this
    spectrograph.
    """

    meta_data_model = meta.get_meta_data_model()
    """
    Metadata model that is generic to all spectrographs.
    """

    def __init__(self):
        self.rawdatasec_img = None
        self.oscansec_img = None

        # Extension with the primary header data
        self.primary_hdrext = 0

        # Generate and check the instrument-specific metadata definition
        self.init_meta()
        self.validate_metadata()

        # TODO: Is there a better way to do this?
        # Validate the instance by checking that the class has defined the
        # number of detectors
        assert self.ndet > 0

        # TODO: Add a call to _check_telescope here?

    @classmethod
    def default_pyphot_par(cls):
        """
        Return the default parameters to use for this instrument.
        
        Returns:
            :class:`~pyphotpar.par.pyphotpar.PyPhotPar`: Parameters required by
            all of ``PyPhot`` methods.
        """
        par = pyphotpar.PyPhotPar()
        par['rdx']['camera'] = cls.name
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
        return self.__class__.default_pyphot_par() if inp_par is None else inp_par

    def _check_telescope(self):
        """Check the derived class has properly defined the telescope."""
        if self.telescope is None:
            raise ValueError('Must define the telescope used to take the observations.')
        if not isinstance(self.telescope, pyphotpar.TelescopePar):
                raise TypeError('Telescope parameters must be one of those specified in'
                                'pypeit.telescopes.')


    # TODO: This circumvents all the infrastructure we have for pulling
    # metadata from headers. Why aren't we using self.meta and
    # self.get_meta_value? See pypeit.metadata.PypeItMetaData._build()
    def parse_spec_header(self, header):
        """
        Parses an input header for key spectrograph items.

        Args:
            header (`astropy.io.fits.Header`_):
                Fits header read from a file.

        Returns:
            :obj:`dict`: Dictionary with the metadata read from ``header``.
        """
        spec_dict = {}
        #
        core_meta_keys = list(meta.define_core_meta().keys())
        core_meta_keys += ['filename']
        for key in core_meta_keys:
            if key.upper() in header.keys():
                spec_dict[key.upper()] = header[key.upper()]
        # Return
        return spec_dict

    def subheader_for_spec(self, row_fitstbl, raw_header, extra_header_cards=None,
                           allow_missing=False):
        """
        Generate a dict that will be added to the Header of spectra files
        generated by ``PypeIt`` (e.g. :class:`~pypeit.specobjs.SpecObjs`).

        Args:
            row_fitstbl (dict-like):
                Typically an `astropy.table.Row`_ or
                `astropy.io.fits.Header`_ with keys defined by
                :func:`~pypeit.core.meta.define_core_meta`.
            raw_header (`astropy.io.fits.Header`_):
                Header that defines the instrument and detector, meaning that
                the header must contain the ``INSTRUME`` and ``DETECTOR``
                header cards. If provided, this must also contain the header
                cards provided by ``extra_header_cards``.
            extra_header_cards (:obj:`list`, optional):
                Additional header cards from ``raw_header`` to include in the
                output dictionary. Can be an empty list or None.
            allow_missing (:obj:`bool`, optional):
                Ignore any keywords returned by
                :func:`~pypeit.core.meta.define_core_meta` are not present in
                ``row_fitstbl``. Otherwise, raise ``PypeItError``.

        Returns:
            :obj:`dict`: Dictionary with data to include an output fits
            header file or table downstream.
        """
        subheader = {}

        core_meta = meta.define_core_meta()
        # Core
        for key in core_meta.keys():
            try:
                subheader[key] = (row_fitstbl[key], core_meta[key]['comment'])
            except KeyError:
                if not allow_missing:
                    msgs.error("Key: {} not present in your fitstbl/Header".format(key))
        # Add a few more
        for key in ['filename']:  # For fluxing
            subheader[key] = row_fitstbl[key]

        # The following are pulled from the original header, if available
        header_cards = ['INSTRUME', 'DETECTOR']
        if extra_header_cards is not None:
            header_cards += extra_header_cards  # For specDB and more
        for card in header_cards:
             if card in raw_header.keys():
                 subheader[card] = raw_header[card]  # Self-assigned instrument name

        # Specify which pipeline created this file
        subheader['PYPELINE'] = self.pypeline
        subheader['PYP_CAMERA'] = (self.name, 'PyPhot: Camera name')

        # Observatory and Header supplied Instrument
        subheader['TELESCOP'] = (self.telescope['name'], 'Telescope')
        subheader['LON-OBS'] = (self.telescope['longitude'], 'Telescope longitude')
        subheader['LAT-OBS'] = (self.telescope['latitude'], 'Telescope latitute')
        subheader['ALT-OBS'] = (self.telescope['elevation'], 'Telescope elevation')

        # Return
        return subheader

    def empty_bpm(self, filename, det, shape=None):
        """
        Generate a generic (empty) bad-pixel mask.

        Even though they are both optional, either the precise shape for the
        image (``shape``) or an example file that can be read to get the
        shape (``filename``) *must* be provided. In the latter, the file is
        read, trimmed, and re-oriented to get the output shape. If both
        ``shape`` and ``filename`` are provided, ``shape`` is ignored.

        Args:
            filename (:obj:`str`):
                An example file to use to get the image shape. Can be None,
                but ``shape`` must be provided, if so. Note the overhead of
                this function is large if you ``filename``. You're better off
                providing ``shape``, if you know it.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (:obj:`tuple`, optional):
                Processed image shape. I.e., if the image for this instrument
                is re-oriented, trimmed, etc, this shape must be that of the
                re-oriented (trimmed, etc) image. This is required if
                ``filename`` is None, but ignored otherwise.

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set to 1
            and an unmasked value set to 0. The shape of the returned image
            should be that of the ``PypeIt`` processed image. This is the
            generic method for the base class, meaning that all pixels are
            returned as unmasked (0s).
        """
        # Load the raw frame
        if filename is None:
            _shape = shape
        else:
            detector_par, _,  _, _, rawdatasec_img, _ = self.get_rawimage(filename, det)
            # Trim + reorient
            trim = procimg.trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
            _shape = trim.shape

        # Shape must be defined at this point.
        if _shape is None:
            msgs.error('Must specify shape if filename is None.')

        # Generate
        # TODO: Why isn't this a boolean array?
        return np.zeros(_shape, dtype=np.int8)

    # TODO: This both edits and returns bpm_img. Is that the behavior we want?
    def bpm_frombias(self, msbias, det, bpm_img):
        """
        Generate a bad-pixel mask from a master bias frame.

        .. warning::
            ``bpm_img`` is edited in-place and returned

        Args:
            msbias (`numpy.ndarray`_):
                Master bias frame used to identify bad pixels.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            bpm_img (`numpy.ndarray`_):
                Bad pixel mask. **Must** be the same shape as ``msbias``.
                **This array is edited in place.**

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set to 1
            and an unmasked value set to 0. The shape of the returned image
            is the same as the provided ``msbias`` and ``bpm_img`` images.
        """
        msgs.info("Generating a BPM for det={0:d} on {1:s}".format(det, self.camera))
        medval = np.median(msbias.image)
        madval = 1.4826 * np.median(np.abs(medval - msbias.image))
        ww = np.where(np.abs(msbias.image - medval) > 10.0 * madval)
        bpm_img[ww] = 1

        # Return
        return bpm_img

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
        # Generate an empty BPM first
        bpm_img = self.empty_bpm(filename, det, shape=shape)

        # Fill in bad pixels if a master bias frame is provided
        if msbias is not None:
            bpm_img = self.bpm_frombias(msbias, det, bpm_img)

        return bpm_img

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return ['filter']

    def valid_configuration_values(self):
        """
        Return a fixed set of valid values for any/all of the configuration
        keys.

        Method is undefined for the base class.

        Returns:
            :obj:`dict`: A dictionary with any/all of the configuration keys
            and their associated discrete set of valid values. If there are
            no restrictions on configuration values, None is returned.
        """
        pass

    def config_independent_frames(self):
        """
        Define frame types that are independent of the fully defined
        instrument configuration.

        By default, bias and dark frames are considered independent of a
        configuration; however, at the moment, these frames can only be
        associated with a *single* configuration. That is, you cannot take
        afternoon biases, change the instrument configuration during the
        night, and then use the same biases for both configurations. See
        :func:`~pypeit.metadata.PypeItMetaData.set_configurations`.

        This method returns a dictionary where the keys of the dictionary are
        the list of configuration-independent frame types. The value of each
        dictionary element can be set to one or more metadata keys that can
        be used to assign each frame type to a given configuration group. See
        :func:`~pypeit.metadata.PypeItMetaData.set_configurations` and how it
        interprets the dictionary values, which can be None.

        Returns:
            :obj:`dict`: Dictionary where the keys are the frame types that
            are configuration-independent and the values are the metadata
            keywords that can be used to assign the frames to a configuration
            group.
        """
        return {'bias': None, 'dark': None}

    def pyphot_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :func:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        pyphot_keys = ['filename', 'frametype']
        # Core
        core_meta = meta.define_core_meta()
        pyphot_keys += list(core_meta.keys())  # Might wish to order these
        # Add in config_keys (if new)
        for key in self.configuration_keys():
            if key not in pyphot_keys:
                pyphot_keys.append(key)
        # Finish
        return pyphot_keys

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Method is undefined in this base class.
       
        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        pass

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}

    def get_detector_par(self, hdu, det):
        """
        Read/Set the detector metadata.

        This method is needed by some instruments that require the detector
        metadata to be interpreted from the output files. This method is
        undefined in the base class.
        """
        pass

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
        # Open
        hdu = fits.open(raw_file)

        # Grab the DetectorContainer
        detector = self.get_detector_par(hdu, det)

        # Raw image
        raw_img = hdu[detector['dataext']].data.astype(float)
        # TODO: This feels very dangerous.  Can we make this a priority?
        # TODO -- Move to FLAMINGOS2 spectrograph
        # Raw data from some spectrograph (i.e. FLAMINGOS2) have an
        # addition extention, so I add the following two lines. It's
        # easier to change here than writing another get_rawimage
        # function in the spectrograph file.
        if raw_img.ndim == 3:
            raw_img = raw_img[0]

        # Extras
        headarr = self.get_headarr(hdu)

        # Exposure time (used by ProcessRawImage)
        exptime = self.get_meta_value(headarr, 'exptime')

        # Rawdatasec, oscansec images
        binning_raw = self.get_meta_value(headarr, 'binning')

        for section in ['datasec', 'oscansec']:

            # Get the data section
            # Try using the image sections as header keywords
            # TODO -- Deal with user windowing of the CCD (e.g. Kast red)
            #  Code like the following maybe useful
            #hdr = hdu[detector[det - 1]['dataext']].header
            #image_sections = [hdr[key] for key in detector[det - 1][section]]
            # Grab from Detector
            image_sections = detector[section]
            #if not isinstance(image_sections, list):
            #    image_sections = [image_sections]
            # Always assume normal FITS header formatting
            one_indexed = True
            include_last = True

            # Initialize the image (0 means no amplifier)
            pix_img = np.zeros(raw_img.shape, dtype=int)
            if section == 'datasec':
                gainimage = np.zeros(raw_img.shape)
                rnimage = np.zeros(raw_img.shape)
            for i in range(detector['numamplifiers']):

                if image_sections is not None:  # and image_sections[i] is not None:
                    # Convert the data section from a string to a slice
                    datasec = parse.sec2slice(image_sections[i], one_indexed=one_indexed,
                                              include_end=include_last, require_dim=2,
                                              binning=binning_raw)
                    # Assign the amplifier
                    pix_img[datasec] = i+1
                    if section == 'datasec':
                        gainimage[datasec] = detector['gain'][i]
                        rnimage[datasec] = detector['ronoise'][i]

            # Finish
            if section == 'datasec':
                rawdatasec_img = pix_img.copy()
            else:
                oscansec_img = pix_img.copy()

        # Return
        return detector, raw_img, headarr, exptime,  rawdatasec_img, oscansec_img

    def get_lamps_status(self, headarr):
        """
        Return a string containing the information on the lamp status.

        Args:
            headarr (:obj:`list`):
                A list of 1 or more `astropy.io.fits.Header`_ objects.

        Returns:
            :obj:`str`: A string that uniquely represents the lamp status.
        """
        # Loop through all lamps and collect their status
        kk = 1
        lampstat = []
        while True:
            lampkey = 'lampstat{:02d}'.format(kk)
            if lampkey not in self.meta.keys():
                break
            # Pull value from header
            lampstat += self.get_meta_value(headarr, lampkey)
            kk += 1
        return "_".join(lampstat)

    def get_meta_value(self, inp, meta_key, required=False, ignore_bad_header=True,
                       usr_row=None, no_fussing=False):
        """
        Return meta data from a given file (or its array of headers).

        Args:
            inp (:obj:`str`, :obj:`list`):
                Input filename or list of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`, :obj:`list`):
                A (list of) strings with the keywords to read from the file
                header(s).
            required (:obj:`bool`, optional):
                The metadata is required and must be available. If it is not,
                the method will raise an exception.
            ignore_bad_header (:obj:`bool`, optional):
                ``PypeIt`` expects certain metadata values to have specific
                datatypes. If the keyword finds the appropriate data but it
                cannot be cast to the correct datatype, this parameter
                determines whether or not the method raises an exception. If
                True, the incorrect type is ignored. It is recommended that
                this be False unless you know for sure that ``PypeIt`` can
                proceed appropriately.
            usr_row (`astropy.table.Table`_, optional):
                A single row table with the user-supplied frametype. This is
                used to determine if the metadata value is required for each
                frametype. Must contain a columns called `frametype`;
                everything else is ignored.
            no_fussing (:obj:`bool`, optional):
                No type checking or anything. Just pass back the first value
                retrieved. This is mainly for bound pairs of meta, e.g.
                ra/dec.

        Returns:
            object: Value recovered for (each) keyword.
        """
        headarr = self.get_headarr(inp) if isinstance(inp, str) else inp

        # Loop?
        if isinstance(meta_key, list):
            return [self.get_meta_value(headarr, key, required=required) for key in meta_key]

        # Are we prepared to provide this meta data?
        if meta_key not in self.meta.keys():
            if required:
                msgs.error("Need to allow for meta_key={} in your meta data".format(meta_key))
            else:
                msgs.warn("Requested meta data for meta_key={} does not exist...".format(meta_key))
                return None

        # Check if this meta key is required
        if 'required' in self.meta[meta_key].keys():
            required = self.meta[meta_key]['required']

        # Is this not derivable?  If so, use the default
        #   or search for it as a compound method
        value = None
        if self.meta[meta_key]['card'] is None:
            if 'default' in self.meta[meta_key].keys():
                value = self.meta[meta_key]['default']
            elif 'compound' in self.meta[meta_key].keys():
                value = self.compound_meta(headarr, meta_key)
            else:
                msgs.error("Failed to load spectrograph value for meta: {}".format(meta_key))
        else:
            # Grab from the header, if we can
            try:
                value = headarr[self.meta[meta_key]['ext']][self.meta[meta_key]['card']]
            except:
                try:
                    #ToDo: This is a hack for MMIRS
                    value = headarr[0][self.meta[meta_key]['card']]
                except:
                    value = None

        # Return now?
        if no_fussing:
            return value

        # Deal with 'special' cases
        if meta_key in ['ra', 'dec'] and value is not None:
            # TODO: Can we get rid of the try/except here and instead get to the heart of the issue?
            try:
                ra, dec = meta.convert_radec(self.get_meta_value(headarr, 'ra', no_fussing=True),
                                    self.get_meta_value(headarr, 'dec', no_fussing=True))
            except:
                msgs.warn('Encounter invalid value of your coordinates. Give zeros for both RA and DEC')
                ra, dec = 0.0, 0.0
            value = ra if meta_key == 'ra' else dec

        # JFH Added this bit of code to deal with situations where the
        # header card is there but the wrong type, e.g. MJD-OBS =
        # 'null'
        try:
            if self.meta_data_model[meta_key]['dtype'] == str:
                retvalue = str(value).strip()
            elif self.meta_data_model[meta_key]['dtype'] == int:
                retvalue = int(value)
            elif self.meta_data_model[meta_key]['dtype'] == float:
                retvalue = float(value)
            elif self.meta_data_model[meta_key]['dtype'] == tuple:
                if not isinstance(value, tuple):
                    msgs.error('dtype for {0} is tuple, but value '.format(meta_key)
                               + 'provided is {0}.  Casting is not possible.'.format(type(value)))
                retvalue = value
            castable = True
        except:
            retvalue = None
            castable = False

        # JFH Added the typing to prevent a crash below when the header
        # value exists, but is the wrong type. This causes a crash
        # below when the value is cast.
        if value is None or not castable:
            # Was this required?
            if required:
                kerror = True
                if not ignore_bad_header:
                    # Is this meta required for this frame type (Spectrograph specific)
                    if ('required_ftypes' in self.meta[meta_key]) and (usr_row is not None):
                        kerror = False
                        # Is it required?
                        # TODO: Use numpy.isin ?
                        for ftype in usr_row['frametype'].split(','):
                            if ftype in self.meta[meta_key]['required_ftypes']:
                                kerror = True
                    # Bomb out?
                    if kerror:
                        msgs.warn('Required meta "{0}" did not load!'.format(meta_key)
                                   + 'You may have a corrupt header.')
                else:
                    msgs.warn('Required card {0} missing '.format(self.meta[meta_key]['card'])
                              + 'from your header.  Proceeding with risk...')
            return None

        # Return
        return retvalue

    def get_wcs(self, hdr, slits, platescale, wave0, dwv):
        """
        Construct/Read a World-Coordinate System for a frame.

        This is undefined in the base class.

        Args:
            hdr (`astropy.io.fits.Header`_):
                The header of the raw frame. The information in this
                header will be extracted and returned as a WCS.
            slits (:class:`~pypeit.slittrace.SlitTraceSet`):
                Slit traces.
            platescale (:obj:`float`): 
                The platescale of an unbinned pixel in arcsec/pixel (e.g.
                detector.platescale).
            wave0 (:obj:`float`):
                The wavelength zeropoint.
            dwv (:obj:`float`):
                Change in wavelength per spectral pixel.

        Returns:
            `astropy.wcs.wcs.WCS`_: The world-coordinate system.
        """
        msgs.warn("No WCS setup for spectrograph: {0:s}".format(self.name))
        return None


    def validate_metadata(self):
        """
        Validates the definitions of the Spectrograph metadata by making a
        series of comparisons to the metadata model defined by
        :func:`pypeit.core.meta.define_core_meta` and :attr:`meta`.
        """
        # Load up
        # TODO: Can we indicate if the metadata element is core instead
        # of having to call both of these?
        core_meta = meta.define_core_meta()
        # KBW: These should have already been defined to self
        #meta_data_model = meta.get_meta_data_model()

        # Check core
        core_keys = np.array(list(core_meta.keys()))
        indx = np.invert(np.isin(core_keys, list(self.meta.keys())))
        if np.any(indx):
            msgs.error('Required keys {0} not defined by spectrograph!'.format(core_keys[indx]))

        # Check for rtol for config keys that are type float
        config_keys = np.array(self.configuration_keys())
        indx = ['rtol' not in self.meta[key].keys() if self.meta_data_model[key]['dtype'] == float
                    else False for key in config_keys]
        if np.any(indx):
            msgs.error('rtol not set for {0} keys in spectrograph meta!'.format(config_keys[indx]))

        # Now confirm all meta are in the data model
        meta_keys = np.array(list(self.meta.keys()))
        indx = np.invert(np.isin(meta_keys, list(self.meta_data_model.keys())))
        if np.any(indx):
            msgs.error('Meta data keys {0} not in metadata model'.format(meta_keys[indx]))

    def get_headarr(self, inp, strict=True):
        """
        Read the header data from all the extensions in the file.

        Args:
            inp (:obj:`str`, `astropy.io.fits.HDUList`_):
                Name of the file to read or the previously opened HDU list.
            strict (:obj:`bool`, optional):
                Function will fault if :func:`fits.getheader` fails to read
                any of the headers. Set to False to report a warning and
                continue.

        Returns:
            :obj:`list`: A list of `astropy.io.fits.Header`_ objects with the
            extension headers.
        """
        # Faster to open the whole file and then assign the headers,
        # particularly for gzipped files (e.g., DEIMOS)
        if isinstance(inp, str):
            try:
                hdu = fits.open(inp)
            except:
                if strict:
                    msgs.error('Problem opening {0}.'.format(inp))
                else:
                    msgs.warn('Problem opening {0}.'.format(inp) + msgs.newline()
                              + 'Proceeding, but should consider removing this file!')
                    return ['None']*999 # self.numhead
        else:
            hdu = inp
        return [hdu[k].header for k in range(len(hdu))]

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

        Raises:
            NotImplementedError:
                Raised by the base class to denote that any derived class has
                not been properly defined.
        """
        raise NotImplementedError('Frame typing not defined for {0}.'.format(self.name))

    def idname(self, ftype):
        """
        Return the ``idname`` for the selected frame type for this
        instrument.

        Args:
            ftype (:obj:`str`):
                Frame type, which should be one of the keys in
                :class:`~pypeit.core.framematch.FrameTypeBitMask`.

        Returns:
            :obj:`str`: The value of ``idname`` that should be available in
            the :class:`~pypeit.metadata.PypeItMetaData` instance that
            identifies frames of this type.

        Raises:
            NotImplementedError:
                Raised by the base class to denote that any derived class has
                not been properly defined.
        """
        raise NotImplementedError('Header keyword with frame type not defined for {0}.'.format(
                                  self.name))

    def __repr__(self):
        """Return a string representation of the instance."""
        txt = '<{:s}: '.format(self.__class__.__name__)
        txt += ' camera={:s},'.format(self.name)
        txt += ' telescope={:s},'.format(self.telescope['name'])
        txt += '>'
        return txt


