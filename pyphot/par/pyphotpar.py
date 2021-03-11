# encoding: utf-8
"""
Defines parameter sets used to set the behavior for core pyphot
functionality.

For more details on the full parameter hierarchy and a tabulated
description of the keywords in each parameter set, see :ref:`pyphotpar`.

For examples of how to change the parameters for a run of pyphot using
the pyphot input file, see :ref:`pyphot_file`.

**New Parameters**:

To add a new parameter, let's call it `foo`, to any of the provided
parameter sets:

    - Add ``foo=None`` to the ``__init__`` method of the relevant
      parameter set.  E.g.::
        
        def __init__(self, existing_par=None, foo=None):

    - Add any default value (the default value is ``None`` unless you set
      it), options list, data type, and description to the body of the
      ``__init__`` method.  E.g.::

        defaults['foo'] = 'bar'
        options['foo'] = [ 'bar', 'boo', 'fighters' ]
        dtypes['foo'] = str
        descr['foo'] = 'foo? who you callin a foo!  ' \
                       'Options are: {0}'.format(', '.join(options['foo']))

    - Add the parameter to the ``from_dict`` method:
    
        - If the parameter is something that does not require
          instantiation, add the keyword to the ``parkeys`` list in the
          ``from_dict`` method.  E.g.::

            parkeys = [ 'existing_par', 'foo' ]
            kwargs = {}
            for pk in parkeys:
                kwargs[pk] = cfg[pk] if pk in k else None

        - If the parameter is another ParSet or requires instantiation,
          provide the instantiation.  For example, see how the
          :class:`ProcessImagesPar` parameter set is defined in the
          :class:`FrameGroupPar` class.  E.g.::

            pk = 'foo'
            kwargs[pk] = FooPar.from_dict(cfg[pk]) if pk in k else None

**New Parameter Sets:**

To add an entirely new parameter set, use one of the existing parameter
sets as a template, then add the parameter set to :class:`PyPhotPar`,
assuming you want it to be accessed throughout the code.

----
"""
import os
import warnings
from pkg_resources import resource_filename
import inspect
from IPython import embed
from collections import OrderedDict

import numpy

from configobj import ConfigObj

from pyphot.par.parset import ParSet
from pyphot.par import util
from pyphot.par.framematch import FrameTypeBitMask

class FrameGroupPar(ParSet):
    """
    An abstracted group of parameters that defines how specific types of
    frames should be grouped and combined.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, frametype=None, useframe=None, exprng=None, process=None):
        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set
#        defaults['frametype'] = 'bias'
        defaults['frametype'] = frametype       # This is a kludge
        options['frametype'] = FrameGroupPar.valid_frame_types()
        dtypes['frametype'] = str
        descr['frametype'] = 'Frame type.  ' \
                             'Options are: {0}'.format(', '.join(options['frametype']))

        # TODO: Add overscan parameters for each frame type?
        # TODO: JFH This is not documented. What are the options for useframe and what the  does it do?
        defaults['useframe'] = None
        dtypes['useframe'] = str
        descr['useframe'] = 'A master calibrations file to use if it exists.'

        defaults['exprng'] = [None, None]
        dtypes['exprng'] = list
        descr['exprng'] = 'Used in identifying frames of this type.  This sets the minimum ' \
                          'and maximum allowed exposure times.  There must be two items in ' \
                          'the list.  Use None to indicate no limit; i.e., to select exposures ' \
                          'with any time greater than 30 sec, use exprng = [30, None].'

        defaults['process'] = ProcessImagesPar()
        dtypes['process'] = [ ParSet, dict ]
        descr['process'] = 'Low level parameters used for basic image processing'

        # Instantiate the parameter set
        super(FrameGroupPar, self).__init__(list(pars.keys()),
                                            values=list(pars.values()),
                                            defaults=list(defaults.values()),
                                            options=list(options.values()),
                                            dtypes=list(dtypes.values()),
                                            descr=list(descr.values()))

        self.validate()

    @classmethod
    def from_dict(cls, frametype, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['useframe', 'exprng']
        # TODO: cfg can contain frametype but it is ignored...
        allkeys = parkeys + ['process', 'frametype']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for FrameGroupPar.'.format(k[badkeys]))
        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        pk = 'process'
        kwargs[pk] = ProcessImagesPar.from_dict(cfg[pk]) if pk in k else None
        return cls(frametype=frametype, **kwargs)

    @staticmethod
    def valid_frame_types():
        """
        Return the list of valid frame types.
        """
        return FrameTypeBitMask().keys()

    def validate(self):
        if len(self.data['exprng']) != 2:
            raise ValueError('exprng must be a list with two items.')


class ProcessImagesPar(ParSet):
    """
    The parameters needed to perform basic image processing.

    These parameters are primarily used by
    :class:`pyphot.processimages.ProcessImages`, the base class of many
    of the pyphot objects.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, trim=None, apply_gain=None, orient=None,
                 overscan_method=None, overscan_par=None,
                 combine=None, satpix=None,
                 mask_vig=None, minimum_vig=None,
                 mask_cr=None, clip=None,
                 n_lohi=None, replace=None, lamaxiter=None, grow=None,
                 comb_sigrej=None,
                 rmcompact=None, sigclip=None, sigfrac=None, objlim=None,
                 use_biasimage=None, use_overscan=None, use_darkimage=None,
                 use_pixelflat=None, use_illumflat=None, use_specillum=None,
                 use_pattern=None,
                 background=None, boxsize=None, filter_size=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set

        # Raw image fussing
        defaults['trim'] = True
        dtypes['trim'] = bool
        descr['trim'] = 'Trim the image to the detector supplied region'

        defaults['apply_gain'] = True
        dtypes['apply_gain'] = bool
        descr['apply_gain'] = 'Convert the ADUs to electrons using the detector gain'

        defaults['orient'] = True
        dtypes['orient'] = bool
        descr['orient'] = 'Orient the raw image into the PyPhot frame'

        # Bias, overscan, dark, pattern (i.e. detector "signal")
        defaults['use_biasimage'] = True
        dtypes['use_biasimage'] = bool
        descr['use_biasimage'] = 'Use a bias image.  If True, one or more must be supplied in the PyPhot file.'

        defaults['use_overscan'] = True
        dtypes['use_overscan'] = bool
        descr['use_overscan'] = 'Subtract off the overscan.  Detector *must* have one or code will crash.'

        defaults['overscan_method'] = 'savgol'
        options['overscan_method'] = ProcessImagesPar.valid_overscan_methods()
        dtypes['overscan_method'] = str
        descr['overscan_method'] = 'Method used to fit the overscan. ' \
                            'Options are: {0}'.format(', '.join(options['overscan_method']))
        
        defaults['overscan_par'] = [5, 65]
        dtypes['overscan_par'] = [int, list]
        descr['overscan_par'] = 'Parameters for the overscan subtraction.  For ' \
                                '\'polynomial\', set overcan_par = order, number of pixels, ' \
                                'number of repeats ; for \'savgol\', set overscan_par = ' \
                                'order, window size ; for \'median\', set overscan_par = ' \
                                'None or omit the keyword.'

        defaults['use_darkimage'] = False
        dtypes['use_darkimage'] = bool
        descr['use_darkimage'] = 'Subtract off a dark image.  If True, one or more darks must be provided.'

        defaults['use_pattern'] = False
        dtypes['use_pattern'] = bool
        descr['use_pattern'] = 'Subtract off a detector pattern. This pattern is assumed to be sinusoidal' \
                               'along one direction, with a frequency that is constant across the detector.'

        # Flats
        defaults['use_pixelflat'] = True
        dtypes['use_pixelflat'] = bool
        descr['use_pixelflat'] = 'Use the pixel flat to make pixel-level corrections.  A pixelflat image must be provied.'

        defaults['use_illumflat'] = True
        dtypes['use_illumflat'] = bool
        descr['use_illumflat'] = 'Use the illumination flat to correct for the illumination profile of each slit.'

        defaults['use_specillum'] = False
        dtypes['use_specillum'] = bool
        descr['use_specillum'] = 'Use the relative spectral illumination profiles to correct the spectral' \
                                 'illumination profile of each slit. This is primarily used for IFUs.'

        defaults['combine'] = 'weightmean'
        options['combine'] = ProcessImagesPar.valid_combine_methods()
        dtypes['combine'] = str
        descr['combine'] = 'Method used to combine multiple frames.  Options are: {0}'.format(
                                       ', '.join(options['combine']))

        defaults['clip'] = True
        dtypes['clip'] = bool
        descr['clip'] = 'Perform sigma clipping when combining.  Only used with combine=weightmean'

        defaults['comb_sigrej'] = None
        dtypes['comb_sigrej'] = float
        descr['comb_sigrej'] = 'Sigma-clipping level for when clip=True; ' \
                           'Use None for automatic limit (recommended).  '

        defaults['satpix'] = 'reject'
        options['satpix'] = ProcessImagesPar.valid_saturation_handling()
        dtypes['satpix'] = str
        descr['satpix'] = 'Handling of saturated pixels.  Options are: {0}'.format(
                                       ', '.join(options['satpix']))

        # Vignetting parameters
        defaults['mask_vig'] = False
        dtypes['mask_vig'] = bool
        descr['mask_vig'] = 'Identify Vignetting pixels and mask them'

        defaults['minimum_vig'] = 0.5
        dtypes['minimum_vig'] = [int, float]
        descr['minimum_vig'] = 'Sigma level to reject vignetted pixels'

        # CR parameters
        defaults['n_lohi'] = [0, 0]
        dtypes['n_lohi'] = list
        descr['n_lohi'] = 'Number of pixels to reject at the lowest and highest ends of the ' \
                          'distribution; i.e., n_lohi = low, high.  Use None for no limit.'

        defaults['mask_cr'] = False
        dtypes['mask_cr'] = bool
        descr['mask_cr'] = 'Identify CRs and mask them'

        defaults['lamaxiter'] = 1
        dtypes['lamaxiter'] = int
        descr['lamaxiter'] = 'Maximum number of iterations for LA cosmics routine.'

        defaults['grow'] = 1.5
        dtypes['grow'] = [int, float]
        descr['grow'] = 'Factor by which to expand regions with cosmic rays detected by the ' \
                        'LA cosmics routine.'

        defaults['rmcompact'] = True
        dtypes['rmcompact'] = bool
        descr['rmcompact'] = 'Remove compact detections in LA cosmics routine'

        defaults['sigclip'] = 4.5
        dtypes['sigclip'] = [int, float]
        descr['sigclip'] = 'Sigma level for rejection in LA cosmics routine'

        defaults['sigfrac'] = 0.3
        dtypes['sigfrac'] = [int, float]
        descr['sigfrac'] = 'Fraction for the lower clipping threshold in LA cosmics routine.'

        defaults['objlim'] = 3.0
        dtypes['objlim'] = [int, float]
        descr['objlim'] = 'Object detection limit in LA cosmics routine'

        ## bad pixel replacement methods
        defaults['replace'] = 'None'
        options['replace'] = ProcessImagesPar.valid_rejection_replacements()
        dtypes['replace'] = str
        descr['replace'] = 'If all pixels are rejected, replace them using this method.  ' \
                           'Options are: {0}'.format(', '.join(options['replace']))

        ## Background methods
        defaults['background'] = 'median'
        options['background'] = ProcessImagesPar.valid_background_methods()
        dtypes['background'] = str
        descr['background'] = 'Method used to estimate backgrounds.  Options are: {0}'.format(
                                       ', '.join(options['background']))

        defaults['boxsize'] = (50,50)
        dtypes['boxsize'] = [tuple, list]
        descr['boxsize'] = 'Boxsize for background estimation'

        defaults['filter_size'] = (3,3)
        dtypes['filter_size'] = [tuple, list]
        descr['filter_size'] = 'Filter size for background estimation'

        # Instantiate the parameter set
        super(ProcessImagesPar, self).__init__(list(pars.keys()),
                                               values=list(pars.values()),
                                               defaults=list(defaults.values()),
                                               options=list(options.values()),
                                               dtypes=list(dtypes.values()),
                                               descr=list(descr.values()))

        # Check the parameters match the method requirements
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['trim', 'apply_gain', 'orient',
                   'use_biasimage', 'use_pattern', 'use_overscan', 'overscan_method', 'overscan_par', 'use_darkimage',
                   'use_illumflat', 'use_specillum', 'use_pixelflat',
                   'combine', 'satpix', 'n_lohi', 'replace', 'mask_vig','minimum_vig',
                   'mask_cr','lamaxiter', 'grow', 'clip', 'comb_sigrej','rmcompact', 'sigclip', 'sigfrac', 'objlim',
                   'background','boxsize','filter_size']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for ProcessImagesPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_overscan_methods():
        """
        Return the valid overscan methods.
        """
        return ['polynomial', 'savgol', 'median']

    @staticmethod
    def valid_combine_methods():
        """
        Return the valid methods for combining frames.
        """
        return ['median', 'weightmean' ]

    @staticmethod
    def valid_background_methods():
        """
        Return the valid methods for combining frames.
        """
        return ['median', 'mean', 'sextractor' ]

    @staticmethod
    def valid_saturation_handling():
        """
        Return the valid approachs to handling saturated pixels.
        """
        return [ 'reject', 'force', 'nothing' ]

    @staticmethod
    def valid_rejection_replacements():
        """
        Return the valid replacement methods for rejected pixels.
        """
        return ['zero', 'min', 'max', 'mean', 'median', 'None']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """

        if self.data['n_lohi'] is not None and len(self.data['n_lohi']) != 2:
            raise ValueError('n_lohi must be a list of two numbers.')

        if not self.data['use_overscan']:
            return
        if self.data['overscan_par'] is None:
            raise ValueError('No overscan method parameters defined!')

        # Convert param to list
        if isinstance(self.data['overscan_par'], int):
            self.data['overscan_par'] = [self.data['overscan_par']]
        
        if self.data['overscan_method'] == 'polynomial' and len(self.data['overscan_par']) != 3:
            raise ValueError('For polynomial overscan method, set overscan_par = order, '
                             'number of pixels, number of repeats')

        if self.data['overscan_method'] == 'savgol' and len(self.data['overscan_par']) != 2:
            raise ValueError('For savgol overscan method, set overscan_par = order, window size')
            
        if self.data['overscan_method'] == 'median' and self.data['overscan_par'] is not None:
            warnings.warn('No parameters necessary for median overscan method.  Ignoring input.')

    def to_header(self, hdr):
        """
        Write the parameters to a header object.
        """
        hdr['OSCANMET'] = (self.data['overscan'], 'Method used for overscan subtraction')
        hdr['OSCANPAR'] = (','.join([ '{0:d}'.format(p) for p in self.data['overscan_par'] ]),
                                'Overscan method parameters')
        hdr['COMBMAT'] = ('{0}'.format(self.data['match']), 'Frame combination matching')
        hdr['COMBMETH'] = (self.data['combine'], 'Method used to combine frames')
        hdr['COMBSATP'] = (self.data['satpix'], 'Saturated pixel handling when combining frames')
        hdr['COMBSIGR'] = ('{0}'.format(self.data['sigrej']),
                                'Cosmic-ray sigma rejection when combining')
        hdr['COMBNLH'] = (','.join([ '{0}'.format(n) for n in self.data['n_lohi']]),
                                'N low and high pixels rejected when combining')
        hdr['COMBSRJ'] = (self.data['comb_sigrej'], 'Sigma rejection when combining')
        hdr['COMBREPL'] = (self.data['replace'], 'Method used to replace pixels when combining')
        hdr['LACMAXI'] = ('{0}'.format(self.data['lamaxiter']), 'Max iterations for LA cosmic')
        hdr['LACGRW'] = ('{0:.1f}'.format(self.data['grow']), 'Growth radius for LA cosmic')
        hdr['LACRMC'] = (str(self.data['rmcompact']), 'Compact objects removed by LA cosmic')
        hdr['LACSIGC'] = ('{0:.1f}'.format(self.data['sigclip']), 'Sigma clip for LA cosmic')
        hdr['LACSIGF'] = ('{0:.1f}'.format(self.data['sigfrac']),
                            'Lower clip threshold for LA cosmic')
        hdr['LACOBJL'] = ('{0:.1f}'.format(self.data['objlim']),
                            'Object detect limit for LA cosmic')

    @classmethod
    def from_header(cls, hdr):
        """
        Instantiate the object from parameters read from a fits header.
        """
        return cls(overscan=hdr['OSCANMET'],
                   overscan_par=[int(p) for p in hdr['OSCANPAR'].split(',')],
                   match=eval(hdr['COMBMAT']),
                   combine=hdr['COMBMETH'], satpix=hdr['COMBSATP'],
                   n_lohi=[int(p) for p in hdr['COMBNLH'].split(',')],
                   comb_sigrej=float(hdr['COMBSRJ']),
                   replace=hdr['COMBREPL'],
                   cr_sigrej=eval(hdr['LASIGR']),
                   lamaxiter=int(hdr['LACMAXI']), grow=float(hdr['LACGRW']),
                   rmcompact=eval(hdr['LACRMC']), sigclip=float(hdr['LACSIGC']),
                   sigfrac=float(hdr['LACSIGF']), objlim=float(hdr['LACOBJL']))


class FlatFieldPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the field
    flattening.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, method=None, pixelflat_file=None, spec_samp_fine=None,
                 spec_samp_coarse=None, spat_samp=None, tweak_slits=None, tweak_slits_thresh=None,
                 tweak_slits_maxfrac=None, rej_sticky=None, slit_trim=None, slit_illum_pad=None,
                 illum_iter=None, illum_rej=None, twod_fit_npoly=None, saturated_slits=None,
                 slit_illum_relative=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set


        # ToDO there are only two methods. The bspline method and skip, so maybe we should rename the bspline method.
        defaults['method'] = 'bspline'
        options['method'] = FlatFieldPar.valid_methods()
        dtypes['method'] = str
        descr['method'] = 'Method used to flat field the data; use skip to skip flat-fielding.  ' \
                          'Options are: None, {0}'.format(', '.join(options['method']))

        defaults['pixelflat_file'] = None
        dtypes['pixelflat_file'] = str
        descr['pixelflat_file'] = 'Filename of the image to use for pixel-level field flattening'

        defaults['spec_samp_fine'] = 1.2
        dtypes['spec_samp_fine'] = [int, float]
        descr['spec_samp_fine'] = 'bspline break point spacing in units of pixels for spectral fit to flat field blaze function.'

        defaults['spec_samp_coarse'] = 50.0
        dtypes['spec_samp_coarse'] = [int, float]
        descr['spec_samp_coarse'] = 'bspline break point spacing in units of pixels for 2-d bspline-polynomial fit to ' \
                                    'flat field image residuals. This should be a large number unless you are trying to ' \
                                    'fit a sky flat with lots of narrow spectral features.'

        defaults['spat_samp'] = 5.0
        dtypes['spat_samp'] = [int, float]
        descr['spat_samp'] = 'Spatial sampling for slit illumination function. This is the width of the median ' \
                             'filter in pixels used to determine the slit illumination function, and thus sets the ' \
                             'minimum scale on which the illumination function will have features.'

        defaults['tweak_slits'] = True
        dtypes['tweak_slits'] = bool
        descr['tweak_slits'] = 'Use the illumination flat field to tweak the slit edges. ' \
                               'This will work even if illumflatten is set to False '

        defaults['tweak_slits_thresh'] = 0.93
        dtypes['tweak_slits_thresh'] = float
        descr['tweak_slits_thresh'] = 'If tweak_slits is True, this sets the illumination function threshold used to ' \
                                      'tweak the slit boundaries based on the illumination flat. ' \
                                      'It should be a number less than 1.0'

        defaults['tweak_slits_maxfrac'] = 0.10
        dtypes['tweak_slits_maxfrac'] = float
        descr['tweak_slits_maxfrac'] = 'If tweak_slit is True, this sets the maximum fractional amount (of a slits width) ' \
                                       'allowed for trimming each (i.e. left and right) slit boundary, i.e. the default is 10% ' \
                                       'which means slits would shrink or grow by at most 20% (10% on each side)'


        defaults['rej_sticky'] = False
        dtypes['rej_sticky'] = bool
        descr['rej_sticky'] = 'Propagate the rejected pixels through the stages of the ' \
                              'flat-field fitting (i.e, from the spectral fit, to the spatial ' \
                              'fit, and finally to the 2D residual fit).  If False, pixels ' \
                              'rejected in each stage are included in each subsequent stage.'

        defaults['slit_trim'] = 3.
        dtypes['slit_trim'] = [int, float, tuple]
        descr['slit_trim'] = 'The number of pixels to trim each side of the slit when ' \
                             'selecting pixels to use for fitting the spectral response ' \
                             'function.  Single values are used for both slit edges; a ' \
                             'two-tuple can be used to trim the left and right sides differently.'

        defaults['slit_illum_pad'] = 5.
        dtypes['slit_illum_pad'] = [int, float]
        descr['slit_illum_pad'] = 'The number of pixels to pad the slit edges when constructing ' \
                                  'the slit-illumination profile. Single value applied to both ' \
                                  'edges.'

        defaults['slit_illum_relative'] = False
        dtypes['slit_illum_relative'] = [bool]
        descr['slit_illum_relative'] = 'Generate an image of the relative spectral illumination' \
                                       'for a multi-slit setup.'

        defaults['illum_iter'] = 0
        dtypes['illum_iter'] = int
        descr['illum_iter'] = 'The number of rejection iterations to perform when constructing ' \
                              'the slit-illumination profile.  No rejection iterations are ' \
                              'performed if 0.  WARNING: Functionality still being tested.'

        defaults['illum_rej'] = 5.
        dtypes['illum_rej'] = [int, float]
        descr['illum_rej'] = 'The sigma threshold used in the rejection iterations used to ' \
                             'refine the slit-illumination profile.  Rejection iterations are ' \
                             'only performed if ``illum_iter > 0``.'

        dtypes['twod_fit_npoly'] = int
        descr['twod_fit_npoly'] = 'Order of polynomial used in the 2D bspline-polynomial fit to ' \
                                  'flat-field image residuals. The code determines the order of ' \
                                  'these polynomials to each slit automatically depending on ' \
                                  'the slit width, which is why the default is None. Alter ' \
                                  'this paramter at your own risk!'

        defaults['saturated_slits'] = 'crash'
        options['saturated_slits'] = FlatFieldPar.valid_saturated_slits_methods()
        dtypes['saturated_slits'] = str
        descr['saturated_slits'] = 'Behavior when a slit is encountered with a large fraction ' \
                                   'of saturated pixels in the flat-field.  The options are: ' \
                                   '\'crash\' - Raise an error and halt the data reduction; ' \
                                   '\'mask\' - Mask the slit, meaning no science data will be ' \
                                   'extracted from the slit; \'continue\' - ignore the ' \
                                   'flat-field correction, but continue with the reduction.'

        # Instantiate the parameter set
        super(FlatFieldPar, self).__init__(list(pars.keys()),
                                           values=list(pars.values()),
                                           defaults=list(defaults.values()),
                                           options=list(options.values()),
                                           dtypes=list(dtypes.values()),
                                           descr=list(descr.values()))

        # Check the parameters match the method requirements
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['method', 'pixelflat_file', 'spec_samp_fine', 'spec_samp_coarse',
                   'spat_samp', 'tweak_slits', 'tweak_slits_thresh', 'tweak_slits_maxfrac',
                   'rej_sticky', 'slit_trim', 'slit_illum_pad', 'slit_illum_relative',
                   'illum_iter', 'illum_rej', 'twod_fit_npoly', 'saturated_slits']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for FlatFieldPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_methods():
        """
        Return the valid flat-field methods
        """
        return ['bspline', 'skip'] # [ 'PolyScan', 'bspline' ]. Same here. Not sure what PolyScan is

    @staticmethod
    def valid_saturated_slits_methods():
        """
        Return the valid options for dealing with saturated slits.
        """
        return ['crash', 'mask', 'continue']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        # Convert param to list
        #if isinstance(self.data['params'], int):
        #    self.data['params'] = [self.data['params']]
        
        # Check that there are the correct number of parameters
        #if self.data['method'] == 'PolyScan' and len(self.data['params']) != 3:
        #    raise ValueError('For PolyScan method, set params = order, number of '
        #                     'pixels, number of repeats')
        #if self.data['method'] == 'bspline' and len(self.data['params']) != 1:
        #    raise ValueError('For bspline method, set params = spacing (integer).')
        if self.data['pixelflat_file'] is None:
            return

        # Check the frame exists
        if not os.path.isfile(self.data['pixelflat_file']):
            raise ValueError('Provided frame file name does not exist: {0}'.format(
                                self.data['pixelflat_file']))

        # Check that if tweak slits is true that illumflatten is alwo true
        # TODO -- We don't need this set, do we??   See the desc of tweak_slits above
        #if self.data['tweak_slits'] and not self.data['illumflatten']:
        #    raise ValueError('In order to tweak slits illumflatten must be set to True')


class FluxCalibratePar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, extinct_correct=None, extrap_sens=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        defaults['extrap_sens'] = False
        dtypes['extrap_sens'] = bool
        descr['extrap_sens'] = "If False (default), the code will barf if one tries to use " \
                               "sensfunc at wavelengths outside its defined domain. By changing the " \
                               "par['sensfunc']['extrap_blu'] and par['sensfunc']['extrap_red'] this domain " \
                               "can be extended. If True the code will blindly extrapolate."


        defaults['extinct_correct'] = True
        dtypes['extinct_correct'] = bool
        descr['extinct_correct'] = 'If extinct_correct=True the code will use an atmospheric extinction model to ' \
                                   'extinction correct the data below 10000A. Note that this correction makes no ' \
                                   'sense if one is telluric correcting and this shold be set to False'

        # Instantiate the parameter set
        super(FluxCalibratePar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['extinct_correct', 'extrap_sens']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for FluxCalibratePar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        pass

class AstrometricPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, skip_astrometry=None, detect_thresh=None, analysis_thresh=None, detect_minarea=None,
                 crossid_radius=None, position_maxerr=None, pixscale_maxerr=None, mosaic_type=None,
                 astref_catalog=None, astref_band=None, weight_type=None, delete=None, log=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        defaults['skip_astrometry'] = False
        dtypes['skip_astrometry'] = bool
        descr['skip_astrometry'] = 'Skip the astrometry for individual detector image?'

        defaults['weight_type'] = 'MAP_WEIGHT'
        options['weight_type'] = AstrometricPar.valid_weight_type()
        dtypes['weight_type'] = str
        descr['weight_type'] = 'Background Options are: {0}'.format(', '.join(options['weight_type']))

        defaults['detect_thresh'] = 3.0
        dtypes['detect_thresh'] = [int, float]
        descr['detect_thresh'] = ' <sigmas> or <threshold>,<ZP> in mag.arcsec-2 for detection'

        defaults['analysis_thresh'] = 3.0
        dtypes['analysis_thresh'] = [int, float]
        descr['analysis_thresh'] = ' <sigmas> or <threshold>,<ZP> in mag.arcsec-2 for analysis'

        defaults['detect_minarea'] = 5
        dtypes['detect_minarea'] = [int, float]
        descr['detect_minarea'] = 'min. # of pixels above threshold'

        defaults['crossid_radius'] = 2.
        dtypes['crossid_radius'] = [int, float]
        descr['crossid_radius'] = 'Cross-id initial radius (arcsec)'

        defaults['position_maxerr'] = 0.5
        dtypes['position_maxerr'] = [int, float]
        descr['position_maxerr'] = 'Max positional uncertainty (arcmin)'

        defaults['pixscale_maxerr'] = 1.1
        dtypes['pixscale_maxerr'] = [int, float]
        descr['pixscale_maxerr'] = 'Max scale-factor uncertainty'

        defaults['mosaic_type'] = 'UNCHANGED'
        options['mosaic_type'] = AstrometricPar.valid_mosaic_methods()
        dtypes['mosaic_type'] = str
        descr['mosaic_type'] = 'Reference catalog  Options are: {0}'.format(
                                       ', '.join(options['mosaic_type']))

        defaults['astref_catalog'] = 'GAIA-DR2'
        options['astref_catalog'] = AstrometricPar.valid_catalog_methods()
        dtypes['astref_catalog'] = str
        descr['astref_catalog'] = 'Reference catalog  Options are: {0}'.format(
                                       ', '.join(options['astref_catalog']))

        defaults['astref_band'] = 'DEFAULT'
        dtypes['astref_band'] = str
        descr['astref_band'] = 'Photom. band for astr.ref.magnitudes or DEFAULT, BLUEST, or REDDEST'

        defaults['delete'] = True
        dtypes['delete'] = bool
        descr['delete'] = 'Deletec the configuration files for SExtractor, SCAMP, and SWARP?'

        defaults['log'] = False
        dtypes['log'] = bool
        descr['log'] = 'Logging for SExtractor, SCAMP, and SWARP'

        # Instantiate the parameter set
        super(AstrometricPar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['skip_astrometry', 'detect_thresh', 'analysis_thresh', 'detect_minarea', 'crossid_radius',
                   'position_maxerr', 'pixscale_maxerr', 'mosaic_type', 'astref_catalog', 'astref_band',
                   'weight_type', 'delete', 'log']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for AstrometricPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_weight_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['BACKGROUND', 'MAP_RMS', 'MAP_VARIANCE', 'MAP_WEIGHT']

    @staticmethod
    def valid_mosaic_methods():
        """
        Return the valid methods for mosaic method.
        """
        return ['UNCHANGED', 'SAME_CRVAL', 'SHARE_PROJAXIS','FIX_FOCALPLANE','LOOSE']

    @staticmethod
    def valid_catalog_methods():
        """
        Return the valid methods for reference catalog.
        """
        return ['NONE', 'FILE', 'USNO-A2','USNO-B1','GSC-2.3','TYCHO-2','UCAC-4','URAT-1','NOMAD-1','PPMX',
                'CMC-15','2MASS', 'DENIS-3', 'SDSS-R9','SDSS-R12','IGSL','GAIA-DR1','GAIA-DR2','GAIA-EDR3',
                'PANSTARRS-1','ALLWISE']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        pass

class CoaddPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, skip_coadd=None, weight_type=None, rescale_weights=None, combine_type=None,
                 clip_ampfrac=None, clip_sigma=None, blank_badpixels=None, subtract_back=None, back_type=None,
                 back_default=None, back_size=None, back_filtersize=None, back_filtthresh=None,
                 delete=None, log=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        defaults['skip_coadd'] = False
        dtypes['skip_coadd'] = bool
        descr['skip_coadd'] = 'Skip Coadding science targets?'

        defaults['weight_type'] = 'MAP_WEIGHT'
        options['weight_type'] = CoaddPar.valid_weight_type()
        dtypes['weight_type'] = str
        descr['weight_type'] = 'Background Options are: {0}'.format(', '.join(options['weight_type']))

        defaults['rescale_weights'] = False
        dtypes['rescale_weights'] = bool
        descr['rescale_weights'] = 'Rescale input weights/variances?'

        defaults['combine_type'] = 'MEDIAN'
        options['combine_type'] = CoaddPar.valid_combine_type()
        dtypes['combine_type'] = str
        descr['combine_type'] = 'Background Options are: {0}'.format(', '.join(options['combine_type']))

        defaults['clip_ampfrac'] = 0.3
        dtypes['clip_ampfrac'] = [int, float]
        descr['clip_ampfrac'] = 'Fraction of flux variation allowed with clipping'

        defaults['clip_sigma'] = 4.0
        dtypes['clip_sigma'] = [int, float]
        descr['clip_sigma'] = 'RMS error multiple variation allowed with clipping'

        defaults['blank_badpixels'] = False
        dtypes['blank_badpixels'] = bool
        descr['blank_badpixels'] = 'Set to 0 pixels having a weight of 0?'

        defaults['subtract_back'] = False
        dtypes['subtract_back'] = bool
        descr['subtract_back'] = 'Subtract skybackground with Swarp before coadding?'

        defaults['back_type'] = 'AUTO'
        options['back_type'] = CoaddPar.valid_back_type()
        dtypes['back_type'] = str
        descr['back_type'] = 'Background Options are: {0}'.format(', '.join(options['back_type']))

        defaults['back_default'] = 0.0
        dtypes['back_default'] = [int, float]
        descr['back_default'] = 'Default background value in MANUAL'

        defaults['back_size'] = 200
        dtypes['back_size'] = [int, float]
        descr['back_size'] = 'Default background value in MANUAL'

        defaults['back_filtersize'] = 3
        dtypes['back_filtersize'] = [int, float]
        descr['back_filtersize'] = 'Background map filter range (meshes)'

        defaults['back_filtthresh'] = 0.0
        dtypes['back_filtthresh'] = [int, float]
        descr['back_filtthresh'] = 'Threshold above which the background map filter operates'

        defaults['delete'] = False
        dtypes['delete'] = bool
        descr['delete'] = 'Delete the configuration files for SWARP?'

        defaults['log'] = True
        dtypes['log'] = bool
        descr['log'] = 'Logging for SWARP'

        # Instantiate the parameter set
        super(CoaddPar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['skip_coadd', 'weight_type','rescale_weights', 'combine_type', 'clip_ampfrac', 'clip_sigma',
                   'blank_badpixels','subtract_back', 'back_type', 'back_default', 'back_size','back_filtersize',
                   'back_filtthresh', 'delete', 'log']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for AstrometricPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_weight_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['BACKGROUND', 'MAP_RMS', 'MAP_VARIANCE', 'MAP_WEIGHT']

    @staticmethod
    def valid_back_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['AUTO', 'MANUAL']

    @staticmethod
    def valid_combine_type():
        """
        Return the valid methods for reference catalog.
        """
        return ['MEDIAN', 'AVERAGE', 'MIN','MAX','WEIGHTED','CLIPPED','CHI-OLD','CHI-MODE','CHI-MEAN','SUM',
                'WEIGHTED_WEIGHT','MEDIAN_WEIGHT', 'AND', 'NAND','OR','NOR']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        pass


class DetectionPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, detection_method=None, phot_apertures=None, detect_thresh=None, back_type=None,
                 back_default=None, back_size=None, back_filtersize=None, detect_minarea=None,
                 weight_type=None, backphoto_type=None, backphoto_thick=None, conv=None, nnw=None, delete=None, log=None,
                 back_rms_type=None, back_nsigma=None,back_maxiters=None,fwhm=None,nlevels=None,contrast=None,morp_filter=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        defaults['detection_method'] = 'SExtractor'
        options['detection_method'] = DetectionPar.valid_detection_method()
        dtypes['detection_method'] = str
        descr['detection_method'] = 'Background Options are: {0}'.format(', '.join(options['detection_method']))

        ## parameters for all methods
        defaults['phot_apertures'] = [1.0, 2.0, 3.0, 4.0, 5.0]
        dtypes['phot_apertures'] = [int, float, list]
        descr['phot_apertures'] = 'Photometric apertures in units of arcsec'

        defaults['detect_thresh'] = 2.0
        dtypes['detect_thresh'] = [int, float]
        descr['detect_thresh'] = ' <sigmas> or <threshold> for detection'

        defaults['back_type'] = 'AUTO'
        options['back_type'] = DetectionPar.valid_back_type()
        dtypes['back_type'] = str
        descr['back_type'] = 'Background Options are: {0}'.format(', '.join(options['back_type']))

        defaults['back_default'] = 0.0
        dtypes['back_default'] = [int, float]
        descr['back_default'] = 'Default background value in MANUAL'

        defaults['back_size'] = 200
        dtypes['back_size'] = [int, float, tuple]
        descr['back_size'] = 'Default background value in MANUAL, int for SExtractor and tuple for Others'

        defaults['back_filtersize'] = 3
        dtypes['back_filtersize'] = [int, float, tuple]
        descr['back_filtersize'] = 'Background map filter range (meshes), int for SExtractor and tuple for Others'

        ## parameters used by SExtractor or Photoutils
        defaults['detect_minarea'] = 3
        dtypes['detect_minarea'] = [int, float]
        descr['detect_minarea'] = 'min. # of pixels above threshold'

        ## parameters used by SExtractor only
        defaults['weight_type'] = 'MAP_WEIGHT'
        options['weight_type'] = DetectionPar.valid_weight_type()
        dtypes['weight_type'] = str
        descr['weight_type'] = 'Background Options are: {0}'.format(', '.join(options['weight_type']))

        defaults['backphoto_type'] = 'GLOBAL'
        options['backphoto_type'] = DetectionPar.valid_backphoto_type()
        dtypes['backphoto_type'] = str
        descr['backphoto_type'] = 'Background Options are: {0}'.format(', '.join(options['backphoto_type']))

        defaults['backphoto_thick'] = 100
        dtypes['backphoto_thick'] = [int, float]
        descr['backphoto_thick'] = 'Thickness of the background LOCAL annulus'

        defaults['conv'] = 'sex995'
        dtypes['conv'] = str
        descr['conv'] = 'Convolution matrix, either 995 or you can provide the full path of your conv file'

        defaults['nnw'] = 'sex'
        dtypes['nnw'] = str
        descr['nnw'] = 'Use SExtractor default configuration file or you can provide the full path of your nnw file'

        defaults['delete'] = True
        dtypes['delete'] = bool
        descr['delete'] = 'Deletec the configuration files for SExtractor?'

        defaults['log'] = False
        dtypes['log'] = bool
        descr['log'] = 'Logging for SExtractor?'

        ## parameters used by Photutils only
        defaults['back_rms_type'] = 'STD'
        options['back_rms_type'] = DetectionPar.valid_backrms_type()
        dtypes['back_rms_type'] = str
        descr['back_rms_type'] = 'Background Options are: {0}'.format(', '.join(options['back_type']))

        defaults['back_nsigma'] = 3
        dtypes['back_nsigma'] = [int, float]
        descr['back_nsigma'] = 'nsigma for sigma clipping background, used by Photutils only'

        defaults['back_maxiters'] = 10
        dtypes['back_maxiters'] = int
        descr['back_maxiters'] = 'maxiters for sigma clipping backgroun, used by Photutils only'

        defaults['fwhm'] = 5
        dtypes['fwhm'] = [int, float]
        descr['fwhm'] = '# of pixels of seeing, used by Photutils only'

        defaults['nlevels'] = 32
        dtypes['nlevels'] = int
        descr['nlevels'] = 'Nlevel for deblending, used by Photutils only'

        defaults['contrast'] =  0.001
        dtypes['contrast'] = float
        descr['contrast'] = 'Contrast for deblending, used by Photutils only'

        defaults['morp_filter'] = False
        dtypes['morp_filter'] = bool
        descr['morp_filter'] = 'Whether you want to use the kernel filter when measuring morphology and centroid?'\
                               'If set true, it should be similar with SExtractor. False gives a better morphology.'


        # Instantiate the parameter set
        super(DetectionPar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['detection_method', 'phot_apertures', 'detect_thresh', 'back_type', 'back_default',
                   'back_size', 'back_filtersize', 'detect_minarea', 'weight_type','backphoto_type',
                   'backphoto_thick','conv','nnw', 'delete', 'log','back_rms_type','back_nsigma','back_maxiters',
                   'fwhm','nlevels','contrast','morp_filter']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for DetectionPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_detection_method():
        """
        Return the valid methods for mosaic method.
        """
        return ['Photutils', 'SExtractor', 'DAOStar', 'IRAFStar', 'Skip']

    @staticmethod
    def valid_back_type():
        """
        Return the valid methods for mosaic method.
        AUTO and MANUAL for SExtractor while others for photutils
        """
        return ['AUTO', 'MANUAL', 'MEDIAN','MEAN','SEXTRACTOR', 'MMM', 'BIWEIGHT', 'MODE']

    @staticmethod
    def valid_weight_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['BACKGROUND', 'MAP_RMS', 'MAP_VARIANCE', 'MAP_WEIGHT']

    @staticmethod
    def valid_backrms_type():
        """
        Return the valid methods for mosaic method.
        AUTO and MANUAL for SExtractor while others for photutils
        """
        return ['STD', 'MAD', 'BIWEIGHT']

    @staticmethod
    def valid_backphoto_type():
        """
        Return the valid methods for reference catalog.
        """
        return ['GLOBAL', 'LOCAL']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        pass


class ReduxPar(ParSet):
    """
    The parameter set used to hold arguments for functionality relevant
    to the overal reduction of the the data.
    
    Critically, this parameter set defines the camera that was
    used to collect the data and the overall pipeline used in the
    reductions.
    
    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, camera=None, sextractor=None, detnum=None, sortroot=None, calwin=None, scidir=None,
                 qadir=None, coadddir=None, redux_path=None, ignore_bad_headers=None, slitspatnum=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])      # "1:" to skip 'self'

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set

        # NOTE: The validity of the camera is checked by
        # load_camera, so the specification of the viable options here is
        # not really necessary.
#        options['camera'] = ReduxPar.valid_cameras()
        dtypes['camera'] = str
        descr['camera'] = 'Spectrograph that provided the data to be reduced.  ' \
                                'See :ref:`instruments` for valid options.'
#                                'Options are: {0}'.format(', '.join(options['camera']))

        defaults['sextractor'] = 'sex'
        dtypes['sextractor'] = str
        descr['sextractor'] = "The commond for calling SExtractor. In most cases, you should use 'sex' " \
                              "For some rare cases you need to use 'sextractor', depends on how you installed it."

        dtypes['detnum'] = [int, list]
        descr['detnum'] = 'Restrict reduction to a list of detector indices.' \
                          'This cannot (and should not) be used with slitspatnum. '

        dtypes['slitspatnum'] = [str, list]
        descr['slitspatnum'] = 'Restrict reduction to a set of slit DET:SPAT values (closest slit is used). ' \
                               'Example syntax -- slitspatnum = 1:175,1:205   If you are re-running the code, ' \
                               '(i.e. modifying one slit) you *must* have the precise SPAT_ID index.' \
                               'This cannot (and should not) be used with detnum'

        dtypes['sortroot'] = str
        descr['sortroot'] = 'A filename given to output the details of the sorted files.  If ' \
                            'None, the default is the root name of the pyphot file.  If off, ' \
                            'no output is produced.'

        # TODO: Allow this to apply to each calibration frame type
        defaults['calwin'] = 0
        dtypes['calwin']   = [int, float]
        descr['calwin'] = 'The window of time in hours to search for calibration frames for a ' \
                          'science frame'

        # TODO: Explain what this actually does in the description.
        defaults['ignore_bad_headers'] = False
        dtypes['ignore_bad_headers'] = bool
        descr['ignore_bad_headers'] = 'Ignore bad headers (NOT recommended unless you know it is safe).'

        defaults['scidir'] = 'Science'
        dtypes['scidir'] = str
        descr['scidir'] = 'Directory relative to calling directory to write science files.'

        defaults['coadddir'] = 'Coadd'
        dtypes['coadddir'] = str
        descr['coadddir'] = 'Directory relative to calling directory to write science files.'

        defaults['qadir'] = 'QA'
        dtypes['qadir'] = str
        descr['qadir'] = 'Directory relative to calling directory to write quality ' \
                         'assessment files.'

        defaults['redux_path'] = os.getcwd()
        dtypes['redux_path'] = str
        descr['redux_path'] = 'Path to folder for performing reductions.  Default is the ' \
                              'current working directory.'

        # Instantiate the parameter set
        super(ReduxPar, self).__init__(list(pars.keys()),
                                        values=list(pars.values()),
                                        defaults=list(defaults.values()),
                                        options=list(options.values()),
                                        dtypes=list(dtypes.values()),
                                        descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])

        # Basic keywords
        parkeys = [ 'camera', 'sextractor', 'detnum', 'sortroot', 'calwin', 'scidir', 'qadir', 'coadddir',
                    'redux_path', 'ignore_bad_headers', 'slitspatnum']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for ReduxPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        # Check that detnum and slitspatnum are not both set
        if kwargs['detnum'] is not None and kwargs['slitspatnum'] is not None:
            raise IOError("You cannot set both detnum and slitspatnum!  Causes serious SpecObjs output challenges..")
        # Finish
        return cls(**kwargs)

#    @staticmethod
#    def valid_cameras():
#        return available_cameras

    def validate(self):
        pass


class PostProcPar(ParSet):
    """
    The parameter set used to hold arguments for sky subtraction, object
    finding and extraction in the Reduce class

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """

    def __init__(self, astrometric=None, coadd=None, detection=None, photometric=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k, values[k]) for k in args[1:]])  # "1:" to skip 'self'

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        defaults['astrometric'] = AstrometricPar()
        dtypes['astrometric'] = [ParSet, dict]
        descr['astrometric'] = 'Parameters for solving astrometric solutions.'

        defaults['coadd'] = CoaddPar()
        dtypes['coadd'] = [ParSet, dict]
        descr['coadd'] = 'Parameters for coadding science images.'

        defaults['detection'] = DetectionPar()
        dtypes['detection'] = [ParSet, dict]
        descr['detection'] = 'Parameters for solving detections.'


        # Instantiate the parameter set
        super(PostProcPar, self).__init__(list(pars.keys()),
                                             values=list(pars.values()),
                                             defaults=list(defaults.values()),
                                             options=list(options.values()),
                                             dtypes=list(dtypes.values()),
                                             descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])

        allkeys = ['astrometric', 'coadd', 'detection', 'photometric']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for ReducePar.'.format(k[badkeys]))

        kwargs = {}
        for pk in allkeys:
            kwargs[pk] = cfg[pk] if pk in k else None

        return cls(**kwargs)

    def validate(self):
        pass


class SkySubPar(ParSet):
    """
    The parameter set used to hold arguments for functionality relevant
    to sky subtraction.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """

    def __init__(self, bspline_spacing=None, sky_sigrej=None, global_sky_std=None, no_poly=None,
                 user_regions=None, joint_fit=None, load_mask=None, mask_by_boxcar=None,
                 no_local_sky=None):
        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k, values[k]) for k in args[1:]])  # "1:" to skip 'self'

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set
        defaults['bspline_spacing'] = 0.6
        dtypes['bspline_spacing'] = [int, float]
        descr['bspline_spacing'] = 'Break-point spacing for the bspline sky subtraction fits.'

        defaults['sky_sigrej'] = 3.0
        dtypes['sky_sigrej'] = float
        descr['sky_sigrej'] = 'Rejection parameter for local sky subtraction'

        defaults['global_sky_std'] = True
        dtypes['global_sky_std'] = bool
        descr['global_sky_std'] = 'Global sky subtraction will be performed on standard stars. This should be turned' \
                                  'off for example for near-IR reductions with narrow slits, since bright standards can' \
                                  'fill the slit causing global sky-subtraction to fail. In these situations we go ' \
                                  'straight to local sky-subtraction since it is designed to deal with such situations'

        defaults['no_poly'] = False
        dtypes['no_poly'] = bool
        descr['no_poly'] = 'Turn off polynomial basis (Legendre) in global sky subtraction'

        defaults['no_local_sky'] = False
        dtypes['no_local_sky'] = bool
        descr['no_local_sky'] = 'If True, turn off local sky model evaluation, but do fit object profile and perform optimal extraction'

        # Masking
        defaults['user_regions'] = None
        dtypes['user_regions'] = [str, list]
        descr['user_regions'] = 'A user-defined sky regions mask can be set using this keyword. To allow' \
                                'the code to identify the sky regions automatically, set this variable to' \
                                'an empty string. If you wish to set the sky regions, The text should be' \
                                'a comma separated list of percentages to apply to _all_ slits' \
                                ' For example: The following string   :10,35:65,80:   would select the' \
                                'first 10%, the inner 30%, and the final 20% of _all_ slits.'

        defaults['mask_by_boxcar'] = False
        dtypes['mask_by_boxcar'] = bool
        descr['mask_by_boxcar'] = 'In global sky evaluation, mask the sky region around the object by the boxcar radius (set in ExtractionPar).'

        defaults['load_mask'] = False
        dtypes['load_mask'] = bool
        descr['load_mask'] = 'Load a user-defined sky regions mask to be used for the sky regions. Note,' \
                             'if you set this to True, you must first run the pyphot_skysub_regions GUI' \
                             'to manually select and store the regions to file.'

        defaults['joint_fit'] = False
        dtypes['joint_fit'] = bool
        descr['joint_fit'] = 'Perform a simultaneous joint fit to sky regions using all available slits.'

        # Instantiate the parameter set
        super(SkySubPar, self).__init__(list(pars.keys()),
                                        values=list(pars.values()),
                                        defaults=list(defaults.values()),
                                        options=list(options.values()),
                                        dtypes=list(dtypes.values()),
                                        descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])

        # Basic keywords
        parkeys = ['bspline_spacing', 'sky_sigrej', 'global_sky_std', 'no_poly',
                   'user_regions', 'load_mask', 'joint_fit', 'mask_by_boxcar',
                   'no_local_sky']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for SkySubPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    def validate(self):
        pass


class CalibrationsPar(ParSet):
    """
    The superset of parameters used to calibrate the science data.
    
    Note that there are specific defaults for each frame group that are
    different from the defaults of the abstracted :class:`FrameGroupPar`
    class.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, master_dir=None, setup=None, bpm_usebias=None, biasframe=None,
                 darkframe=None, pixelflatframe=None, illumflatframe=None,
                 standardframe=None, flatfield=None,
                 raise_chk_error=None):


        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])      # "1:" to skip 'self'

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set
        defaults['master_dir'] = 'Masters'
        dtypes['master_dir'] = str
        descr['master_dir'] = 'If provided, it should be the name of the folder to ' \
                          'write master files. NOT A PATH. '

        dtypes['setup'] = str
        descr['setup'] = 'If masters=\'force\', this is the setup name to be used: e.g., ' \
                         'C_02_aa .  The detector number is ignored but the other information ' \
                         'must match the Master Frames in the master frame folder.'

        defaults['raise_chk_error'] = True
        dtypes['raise_chk_error'] = bool
        descr['raise_chk_error'] = 'Raise an error if the calibration check fails'

        defaults['bpm_usebias'] = False
        dtypes['bpm_usebias'] = bool
        descr['bpm_usebias'] = 'Make a bad pixel mask from bias frames? Bias frames must be provided.'

        # Calibration Frames
        defaults['biasframe'] = FrameGroupPar(frametype='bias',
                                              process=ProcessImagesPar(apply_gain=False,
                                                                       combine='median',
                                                                       use_biasimage=False,
                                                                       use_pixelflat=False,
                                                                       use_illumflat=False))
        dtypes['biasframe'] = [ ParSet, dict ]
        descr['biasframe'] = 'The frames and combination rules for the bias correction'

        defaults['darkframe'] = FrameGroupPar(frametype='dark',
                                              process=ProcessImagesPar(use_biasimage=False,
                                                                       use_overscan=False,
                                                                       apply_gain=False,
                                                                       use_pixelflat = False,
                                                                       use_illumflat = False))
        dtypes['darkframe'] = [ ParSet, dict ]
        descr['darkframe'] = 'The frames and combination rules for the dark-current correction'

        # JFH Turning off masking of saturated pixels which causes headaches becauase it was being done unintelligently
        defaults['pixelflatframe'] = FrameGroupPar(frametype='pixelflat',
                                                   process=ProcessImagesPar(satpix='nothing',
                                                                            use_pixelflat=False,
                                                                            use_illumflat=False))
        dtypes['pixelflatframe'] = [ ParSet, dict ]
        descr['pixelflatframe'] = 'The frames and combination rules for the pixel flat'

        defaults['illumflatframe'] = FrameGroupPar(frametype='illumflat',
                                                   process=ProcessImagesPar(satpix='nothing',
                                                                            use_pixelflat=False,
                                                                            use_illumflat=False))
        dtypes['illumflatframe'] = [ ParSet, dict ]
        descr['illumflatframe'] = 'The frames and combination rules for the illumination flat'


        defaults['standardframe'] = FrameGroupPar(frametype='standard',
                                                  process=ProcessImagesPar(mask_cr=True))
        dtypes['standardframe'] = [ ParSet, dict ]
        descr['standardframe'] = 'The frames and combination rules for the spectrophotometric ' \
                                 'standard observations'


        defaults['flatfield'] = FlatFieldPar()
        dtypes['flatfield'] = [ ParSet, dict ]
        descr['flatfield'] = 'Parameters used to set the flat-field procedure'


        # Instantiate the parameter set
        super(CalibrationsPar, self).__init__(list(pars.keys()),
                                              values=list(pars.values()),
                                              defaults=list(defaults.values()),
                                              options=list(options.values()),
                                              dtypes=list(dtypes.values()),
                                              descr=list(descr.values()))
        #self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])

        # Basic keywords
        parkeys = [ 'master_dir', 'setup', 'bpm_usebias', 'raise_chk_error']

        allkeys = parkeys + ['biasframe', 'darkframe', 'pixelflatframe',
                             'illumflatframe','standardframe', 'flatfield']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for CalibrationsPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None

        # Keywords that are ParSets
        pk = 'biasframe'
        kwargs[pk] = FrameGroupPar.from_dict('bias', cfg[pk]) if pk in k else None
        pk = 'darkframe'
        kwargs[pk] = FrameGroupPar.from_dict('dark', cfg[pk]) if pk in k else None
        pk = 'pixelflatframe'
        kwargs[pk] = FrameGroupPar.from_dict('pixelflat', cfg[pk]) if pk in k else None
        pk = 'illumflatframe'
        kwargs[pk] = FrameGroupPar.from_dict('illumflat', cfg[pk]) if pk in k else None
        pk = 'standardframe'
        kwargs[pk] = FrameGroupPar.from_dict('standard', cfg[pk]) if pk in k else None
        pk = 'flatfield'
        kwargs[pk] = FlatFieldPar.from_dict(cfg[pk]) if pk in k else None

        return cls(**kwargs)


class PyPhotPar(ParSet):
    """
    The superset of parameters used by PyPhot.
    
    This is a single object used as a container for all the
    user-specified arguments used by PyPhot.
    
    To get the default parameters for a given camera, e.g.::

        from pyphot.cameras.util import load_camera

        camera = load_camera('shane_kast_blue')
        par = camera.default_pyphot_par()

    If the user has a set of configuration alterations to be read from a
    pyphot file, e.g.::

        from pyphot.par.util import parse_pyphot_file
        from pyphot.cameras.util import load_camera
        from pyphot.par import PyPhotPar

        camera = load_camera('shane_kast_blue')
        spec_cfg_lines = camera.default_pyphot_par().to_config()
        user_cfg_lines = parse_pyphot_file('myrdx.pyphot')[0]
        par = PyPhotPar.from_cfg_lines(cfg_lines=spec_cfg_lines,
                                      merge_with=user_cfg_lines)

    To write the configuration of a given instance of :class:`PyPhotPar`,
    use the :func:`to_config` function::
        
        par.to_config('mypyphotpar.cfg')

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, rdx=None, calibrations=None, scienceframe=None, postproc=None,
                 flexure=None, fluxcalib=None, coadd1d=None, coadd2d=None, sensfunc=None, tellfit=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])      # "1:" to skip 'self'

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set
        defaults['rdx'] = ReduxPar()
        dtypes['rdx'] = [ ParSet, dict ]
        descr['rdx'] = 'PyPhot reduction rules.'

#        defaults['baseprocess'] = ProcessImagesPar()
#        dtypes['baseprocess'] = [ ParSet, dict ]
#        descr['baseprocess'] = 'Default-level parameters used when processing all images'

        defaults['calibrations'] = CalibrationsPar()
        dtypes['calibrations'] = [ ParSet, dict ]
        descr['calibrations'] = 'Parameters for the calibration algorithms'

        defaults['scienceframe'] = FrameGroupPar(frametype='science',
                                                 process=ProcessImagesPar(mask_cr=True))
        dtypes['scienceframe'] = [ ParSet, dict ]
        descr['scienceframe'] = 'The frames and combination rules for the science observations'

        defaults['postproc'] = PostProcPar()
        dtypes['postproc'] = [ParSet, dict]
        descr['postproc'] = 'Parameters for astrometric, coadding, and photometry.'

        # Flux calibration is turned OFF by default
        defaults['fluxcalib'] = FluxCalibratePar()
        dtypes['fluxcalib'] = [ParSet, dict]
        descr['fluxcalib'] = 'Parameters used by the flux-calibration procedure.  Flux ' \
                             'calibration is not performed by default.  To turn on, either ' \
                             'set the parameters in the \'fluxcalib\' parameter group or set ' \
                             '\'fluxcalib = True\' in the \'rdx\' parameter group to use the ' \
                             'default flux-calibration parameters.'

        # Instantiate the parameter set
        super(PyPhotPar, self).__init__(list(pars.keys()),
                                       values=list(pars.values()),
                                       defaults=list(defaults.values()),
                                       dtypes=list(dtypes.values()),
                                       descr=list(descr.values()))

        self.validate()

    @classmethod
    def from_cfg_file(cls, cfg_file=None, merge_with=None, evaluate=True):
        """
        Construct the parameter set using a configuration file.

        Note that::

            default = PyPhotPar()
            nofile = PyPhotPar.from_cfg_file()
            assert default.data == nofile.data, 'This should always pass.'

        Args:
            cfg_file (:obj:`str`, optional):
                The name of the configuration file that defines the
                default parameters.  This can be used to load a pyphot
                config file from a previous run that was constructed and
                output by pyphot.  This has to contain the full set of
                parameters, not just the subset you want to change.  For
                the latter, use `merge_with` to provide one or more
                config files to merge with the defaults to construct the
                full parameter set.
            merge_with (:obj:`str`, :obj:`list`, optional):
                One or more config files with the modifications to
                either default parameters (`cfg_file` is None) or
                the parameters provided by `cfg_file`.  The
                modifications are performed in series so the list order
                of the config files is important.
            evaluate (:obj:`bool`, optional):
                Evaluate the values in the config object before
                assigning them in the subsequent parameter sets.  The
                parameters in the config file are *always* read as
                strings, so this should almost always be true; however,
                see the warning below.
                
        .. warning::

            When `evaluate` is true, the function runs `eval()` on
            all the entries in the `ConfigObj` dictionary, done using
            :func:`_recursive_dict_evaluate`.  This has the potential to
            go haywire if the name of a parameter unintentionally
            happens to be identical to an imported or system-level
            function.  Of course, this can be useful by allowing one to
            define the function to use as a parameter, but it also means
            one has to be careful with the values that the parameters
            should be allowed to have.  The current way around this is
            to provide a list of strings that should be ignored during
            the evaluation, done using :func:`_eval_ignore`.

        .. todo::
            Allow the user to add to the ignored strings.

        Returns:
            :class:`pyphot.par.core.PyPhotPar`: The instance of the
            parameter set.
        """
        # Get the base parameters in a ConfigObj instance
        cfg = ConfigObj(PyPhotPar().to_config() if cfg_file is None else cfg_file)

        # Get the list of other configuration parameters to merge it with
        _merge_with = [] if merge_with is None else \
                        ([merge_with] if isinstance(merge_with, str) else merge_with)
        merge_cfg = ConfigObj()
        for f in _merge_with:
            merge_cfg.merge(ConfigObj(f))

        # Merge with the defaults
        cfg.merge(merge_cfg)

        # Evaluate the strings if requested
        if evaluate:
            cfg = util.recursive_dict_evaluate(cfg)
        
        # Instantiate the object based on the configuration dictionary
        return cls.from_dict(cfg)

    @classmethod
    def from_cfg_lines(cls, cfg_lines=None, merge_with=None, evaluate=True):
        """
        Construct the parameter set using the list of string lines read
        from a config file.

        Note that::

            default = PyPhotPar()
            nofile = PyPhotPar.from_cfg_lines()
            assert default.data == nofile.data, 'This should always pass.'

        Args:
            cfg_lines (:obj:`list`, optional):
                A list of strings with lines read, or made to look like
                they are, from a configuration file.  This can be used
                to load lines from a previous run of pyphot that was
                constructed and output by pyphot.  This has to contain
                the full set of parameters, not just the subset to
                change.  For the latter, leave this as the default value
                (None) and use `merge_with` to provide a set of
                lines to merge with the defaults to construct the full
                parameter set.
            merge_with (:obj:`list`, optional):
                A list of strings with lines read, or made to look like
                they are, from a configuration file that should be
                merged with the lines provided by `cfg_lines`, or the
                default parameters.
            evaluate (:obj:`bool`, optional):
                Evaluate the values in the config object before
                assigning them in the subsequent parameter sets.  The
                parameters in the config file are *always* read as
                strings, so this should almost always be true; however,
                see the warning below.
                
        .. warning::

            When `evaluate` is true, the function runs `eval()` on
            all the entries in the `ConfigObj` dictionary, done using
            :func:`_recursive_dict_evaluate`.  This has the potential to
            go haywire if the name of a parameter unintentionally
            happens to be identical to an imported or system-level
            function.  Of course, this can be useful by allowing one to
            define the function to use as a parameter, but it also means
            one has to be careful with the values that the parameters
            should be allowed to have.  The current way around this is
            to provide a list of strings that should be ignored during
            the evaluation, done using :func:`_eval_ignore`.

        .. todo::
            Allow the user to add to the ignored strings.

        Returns:
            :class:`pyphot.par.core.PyPhotPar`: The instance of the
            parameter set.
        """
        # Get the base parameters in a ConfigObj instance
        cfg = ConfigObj(PyPhotPar().to_config() if cfg_lines is None else cfg_lines)

        # Merge in additional parameters
        if merge_with is not None:
            cfg.merge(ConfigObj(merge_with))

        # Evaluate the strings if requested
        if evaluate:
            cfg = util.recursive_dict_evaluate(cfg)

        # Instantiate the object based on the configuration dictionary
        return cls.from_dict(cfg)

    @classmethod
    def from_pyphot_file(cls, ifile, evaluate=True):
        """
        Construct the parameter set using a pyphot file.
        
        Args:
            ifile (str):
                Name of the pyphot file to read.  Expects to find setup
                and data blocks in the file.  See docs.
            evaluate (:obj:`bool`, optional):
                Evaluate the values in the config object before
                assigning them in the subsequent parameter sets.  The
                parameters in the config file are *always* read as
                strings, so this should almost always be true; however,
                see the warning below.
                
        .. warning::

            When `evaluate` is true, the function runs `eval()` on
            all the entries in the `ConfigObj` dictionary, done using
            :func:`_recursive_dict_evaluate`.  This has the potential to
            go haywire if the name of a parameter unintentionally
            happens to be identical to an imported or system-level
            function.  Of course, this can be useful by allowing one to
            define the function to use as a parameter, but it also means
            one has to be careful with the values that the parameters
            should be allowed to have.  The current way around this is
            to provide a list of strings that should be ignored during
            the evaluation, done using :func:`_eval_ignore`.

        .. todo::
            Allow the user to add to the ignored strings.

        Returns:
            :class:`pyphot.par.core.PyPhotPar`: The instance of the
            parameter set.
        """
        # TODO: Need to include instrument-specific defaults somewhere...
        return cls.from_cfg_lines(merge_with=util.pyphot_config_lines(ifile), evaluate=evaluate)

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])

        allkeys = ['rdx', 'calibrations', 'scienceframe', 'postproc', 'flexure', 'fluxcalib',
                   'coadd1d', 'coadd2d', 'sensfunc', 'baseprocess', 'tellfit']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for PyPhotPar.'.format(k[badkeys]))

        kwargs = {}

        pk = 'rdx'
        kwargs[pk] = ReduxPar.from_dict(cfg[pk]) if pk in k else None

        #pk = 'calibrations'
        #kwargs[pk] = CalibrationsPar.from_dict(cfg[pk]) if pk in k else None

        pk = 'scienceframe'
        kwargs[pk] = FrameGroupPar.from_dict('science', cfg[pk]) if pk in k else None

        pk = 'postproc'
        kwargs[pk] = PostProcPar.from_dict(cfg[pk]) if pk in k else None

        # Allow flux calibration to be turned on using cfg['rdx']
        pk = 'fluxcalib'
        default = FluxCalibratePar() \
                        if pk in cfg['rdx'].keys() and cfg['rdx']['fluxcalib'] else None
        kwargs[pk] = FluxCalibratePar.from_dict(cfg[pk]) if pk in k else default


        if 'baseprocess' not in k:
            return cls(**kwargs)

        # Include any alterations to the basic processing of *all*
        # images
        self = cls(**kwargs)
        baseproc = ProcessImagesPar.from_dict(cfg['baseprocess'])
        self.sync_processing(baseproc)
        return self

    def reset_all_processimages_par(self, **kwargs):
        """
        Set all of the ProcessImagesPar objects to have the input setting

        e.g.

        par.reset_all_processimages_par(use_illumflat=False)

        Args:
            **kwargs:
        """
        # Calibrations
        for _key in self['calibrations'].keys():
            if isinstance(self['calibrations'][_key], ParSet) and 'process' in self['calibrations'][_key].keys():
                for key,value in kwargs.items():
                    self['calibrations'][_key]['process'][key] = value
        # Science frame
        for _key in self.keys():
            if isinstance(self[_key], ParSet) and 'process' in self[_key].keys():
                for key,value in kwargs.items():
                    self[_key]['process'][key] = value

    def sync_processing(self, proc_par):
        """
        Sync the processing of all the frame types based on the input
        ProcessImagesPar parameters.

        The parameters are merged in sequence starting from the
        parameter defaults, then including global adjustments provided
        by ``process``, and ending with the parameters that may have
        already been changed for each frame.

        This function can be used at anytime, but is most useful with
        the from_dict method where a ``baseprocess`` group can be
        supplied to change the processing parameters for all frames away
        from the defaults.

        Args:
            proc_par (:class:`ProcessImagesPar`):
                Effectively a new set of default image processing
                parameters for all frames.

        Raises:
            TypeError:
                Raised if the provided parameter set is not an instance
                of :class:`ProcessImagesPar`.
        """
        # Checks
        if not isinstance(proc_par, ProcessImagesPar):
            raise TypeError('Must provide an instance of ProcessImagesPar')
        
        # All the relevant ParSets are already ProcessImagesPar objects,
        # so we can work directly with the internal dictionaries.

        # Find the keys in the input that are different from the default
        default = ProcessImagesPar()
        base_diff = [ k for k in proc_par.keys() if default[k] != proc_par[k] ]

        # Calibration frames
        frames = [ f for f in self['calibrations'].keys() if 'frame' in f ]
        for f in frames:
            # Find the keys in self that are the same as the default
            frame_same = [ k for k in proc_par.keys() 
                            if self['calibrations'][f]['process'].data[k] == default[k] ]
            to_change = list(set(base_diff) & set(frame_same))
            for k in to_change:
                self['calibrations'][f]['process'].data[k] = proc_par[k]
            
        # Science frames
        frame_same = [ k for k in proc_par.keys() 
                            if self['scienceframe']['process'].data[k] == default[k] ]
        to_change = list(set(base_diff) & set(frame_same))
        for k in to_change:
            self['scienceframe']['process'].data[k] = proc_par[k]

    # TODO: Perform extensive checking that the parameters are valid for
    # a full run of PyPhot.  May not be necessary because validate will
    # be called for all the sub parameter sets, but this can do higher
    # level checks, if necessary.
    def validate(self):
        pass


class TelescopePar(ParSet):
    """
    The parameters used to define the salient properties of a telescope.

    These parameters should be *independent* of any specific use of the
    telescope.  They and are used by the :mod:`pyphot.telescopes` module
    to define the telescopes served by PyPhot, and kept as part of the
    :class:`pyphot.cameras.camera.Spectrograph` definition of
    the instruments served by PyPhot.

    To see the list of instruments served, a table with the the current
    keywords, defaults, and descriptions for the :class:`TelescopePar`
    class, and an explanation of how to define a new instrument, see
    :ref:`instruments`.
    """
    def __init__(self, name=None, longitude=None, latitude=None, elevation=None, fratio=None,
                 diameter=None):

        # Grab the parameter names and values from the function
        # arguments
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        pars = OrderedDict([(k,values[k]) for k in args[1:]])

        # Initialize the other used specifications for this parameter
        # set
        defaults = OrderedDict.fromkeys(pars.keys())
        options = OrderedDict.fromkeys(pars.keys())
        dtypes = OrderedDict.fromkeys(pars.keys())
        descr = OrderedDict.fromkeys(pars.keys())

        # Fill out parameter specifications.  Only the values that are
        # *not* None (i.e., the ones that are defined) need to be set
        defaults['name'] = 'KECK'
        options['name'] = TelescopePar.valid_telescopes()
        dtypes['name'] = str
        descr['name'] = 'Name of the telescope used to obtain the observations.  ' \
                        'Options are: {0}'.format(', '.join(options['name']))
        
        dtypes['longitude'] = [int, float]
        descr['longitude'] = 'Longitude of the telescope on Earth in degrees.'

        dtypes['latitude'] = [int, float]
        descr['latitude'] = 'Latitude of the telescope on Earth in degrees.'

        dtypes['elevation'] = [int, float]
        descr['elevation'] = 'Elevation of the telescope in m'

        dtypes['fratio'] = [int, float]
        descr['fratio'] = 'f-ratio of the telescope'

        dtypes['diameter'] = [int, float]
        descr['diameter'] = 'Diameter of the telescope in m'

        # Instantiate the parameter set
        super(TelescopePar, self).__init__(list(pars.keys()),
                                           values=list(pars.values()),
                                           defaults=list(defaults.values()),
                                           options=list(options.values()),
                                           dtypes=list(dtypes.values()),
                                           descr=list(descr.values()))

        # Check the parameters match the method requirements
        self.validate()


    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = [ 'name', 'longitude', 'latitude', 'elevation', 'fratio', 'diameter' ]

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for TelescopePar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_telescopes():
        """
        Return the valid telescopes.
        """
        return [ 'GEMINI-N','GEMINI-S', 'KECK', 'SHANE', 'WHT', 'APF', 'TNG', 'VLT', 'MAGELLAN', 'LBT', 'MMT', 'KPNO', 'NOT', 'P200']

    def validate(self):
        pass

    def platescale(self):
        r"""
        Return the platescale of the telescope in arcsec per mm.

        Calculated as

        .. math::
            p = \frac{206265}{f D},

        where :math:`f` is the f-ratio and :math:`D` is the diameter.
        If either of these is not available, the function returns
        `None`.
        """
        return None if self['fratio'] is None or self['diameter'] is None \
                else 206265/self['fratio']/self['diameter']/1e3


