# encoding: utf-8
"""
Defines parameter sets used to set the behavior for core pyphot
functionality.

Modified from PyPeIt

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
import inspect
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
                 comb_cenfunc=None, comb_stdfunc=None, clip=None, comb_maxiter=None,comb_sigrej=None,
                 satpix=None, mask_proc=None, window_size=None, maskpixvar=None, mask_negative_star=None,
                 mask_vig=None, minimum_vig=None, mask_brightstar=None, brightstar_nsigma=None,brightstar_method=None,
                 conv=None, mask_cr=None, contrast=None, use_medsky=None,
                 cr_threshold=None, neighbor_threshold=None,
                 n_lohi=None, replace=None, lamaxiter=None, grow=None,
                 rmcompact=None, sigclip=None, sigfrac=None, objlim=None,
                 mask_sat=None, sat_sig=None, sat_buf=None, sat_order=None, low_thresh=None, h_thresh=None,
                 small_edge=None, line_len=None, line_gap=None, percentile=None,
                 use_biasimage=None, use_overscan=None, use_darkimage=None,
                 use_pixelflat=None, use_illumflat=None, use_supersky=None, use_fringe=None,
                 back_type=None, back_rms_type=None, back_size=None, back_filtersize=None, back_maxiters=None):

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

        defaults['use_overscan'] = False
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

        # Flats
        defaults['use_pixelflat'] = True
        dtypes['use_pixelflat'] = bool
        descr['use_pixelflat'] = 'Use the pixel flat to make pixel-level corrections.  A pixelflat image must be provied.'

        defaults['use_illumflat'] = False
        dtypes['use_illumflat'] = bool
        descr['use_illumflat'] = 'Use the illumination flat to correct for the illumination profile.'

        defaults['use_supersky'] = True
        dtypes['use_supersky'] = bool
        descr['use_supersky'] = 'Use supersky frame to further faltten your science images.'

        defaults['use_fringe'] = False
        dtypes['use_fringe'] = bool
        descr['use_fringe'] = 'Subtract off a fringing pattern. This pattern usually appears for thin CCD at red wavelength.'

        defaults['mask_negative_star'] = False
        dtypes['mask_negative_star'] = bool
        descr['mask_negative_star'] = 'Mask negative stars? Need set to True for  dirty image like WIRCam long exposure image'

        defaults['comb_cenfunc'] = 'median'
        options['comb_cenfunc'] = ProcessImagesPar.valid_combine_methods()
        dtypes['comb_cenfunc'] = str
        descr['comb_cenfunc'] = 'Method used to combine multiple frames.  Options are: {0}'.format(
                                       ', '.join(options['comb_cenfunc']))

        defaults['comb_stdfunc'] = 'std'
        options['comb_stdfunc'] = ProcessImagesPar.valid_combine_stdfunc()
        dtypes['comb_stdfunc'] = str
        descr['comb_stdfunc'] = 'Std function used to combine multiple frames.  Options are: {0}'.format(
                                       ', '.join(options['comb_stdfunc']))

        defaults['clip'] = True
        dtypes['clip'] = bool
        descr['clip'] = 'Perform sigma clipping when combining.  Only used with combine=weightmean'

        defaults['comb_sigrej'] = 3.0
        dtypes['comb_sigrej'] = float
        descr['comb_sigrej'] = 'Sigma-clipping level for when combing images'

        defaults['comb_maxiter'] = 5
        dtypes['comb_maxiter'] = int
        descr['comb_maxiter'] = 'Maximum number of combining images.'

        defaults['satpix'] = 'reject'
        options['satpix'] = ProcessImagesPar.valid_saturation_handling()
        dtypes['satpix'] = str
        descr['satpix'] = 'Handling of saturated pixels.  Options are: {0}'.format(
                                       ', '.join(options['satpix']))

        # Detector process parameters
        defaults['mask_proc'] = True
        dtypes['mask_proc'] = bool
        descr['mask_proc'] = 'Mask bad pixels identified from detector processing'

        defaults['window_size'] = (51,51)
        dtypes['window_size'] = [tuple, list]
        descr['window_size'] = 'Box size for estimating large scale patterns, i.e. used for illuminating flat.'

        defaults['maskpixvar'] = None
        dtypes['maskpixvar'] = float
        descr['maskpixvar'] = 'The maximum allowed variance of pixelflat. The default value (0.1),'\
                              'means pixel value with >1.1 or <0.9 will be masked.'

        defaults['mask_brightstar'] = True
        dtypes['mask_brightstar'] = bool
        descr['mask_brightstar'] = 'Mask bright stars?'

        defaults['brightstar_method'] = 'sextractor'
        options['brightstar_method'] = ProcessImagesPar.valid_brightstar_methods()
        dtypes['brightstar_method'] = str
        descr['brightstar_method'] = 'If all pixels are rejected, replace them using this method.  ' \
                           'Options are: {0}'.format(', '.join(options['brightstar_method']))

        defaults['conv'] = 'sex'
        dtypes['conv'] = str
        descr['conv'] = 'Convolution matrix, either default sex, or sex995 or you can provide the full path of your conv file'

        defaults['brightstar_nsigma'] = 5
        dtypes['brightstar_nsigma'] = [int, float]
        descr['brightstar_nsigma'] = 'Sigma level to mask bright stars.'

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

        defaults['mask_cr'] = True
        dtypes['mask_cr'] = bool
        descr['mask_cr'] = 'Identify CRs and mask them'

        defaults['lamaxiter'] = 1
        dtypes['lamaxiter'] = int
        descr['lamaxiter'] = 'Maximum number of iterations for LA cosmics routine.'

        defaults['contrast'] = 2.0
        dtypes['contrast'] = [int, float]
        descr['contrast'] = 'Contrast threshold between the Laplacian image and the fine-structure image.'\
                            'If your image is critically sampled, use a value around 2.'\
                            'If your image is undersampled (e.g., HST data), a value of 4 or 5 (or more) is more appropriate.'\
                            'If your image is oversampled, use a value between 1 and 2.'

        defaults['cr_threshold'] = 5.0
        dtypes['cr_threshold'] = [int, float]
        descr['cr_threshold'] = 'The Laplacian signal-to-noise ratio threshold for cosmic-ray detection.'

        defaults['neighbor_threshold'] = 2.0
        dtypes['neighbor_threshold'] = [int, float]
        descr['neighbor_threshold'] = 'The Laplacian signal-to-noise ratio threshold for detection of cosmic rays'\
                                      'in pixels neighboring the initially-identified cosmic rays.'

        defaults['grow'] = 1.5
        dtypes['grow'] = [int, float]
        descr['grow'] = 'Factor by which to expand the masked cosmic ray, satellite, ' \
                        'negative star, and vignetting pixels'

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

        ## satellite trail parameters
        defaults['mask_sat'] = True
        dtypes['mask_sat'] = bool
        descr['mask_sat'] = 'Mask satellite trails?'

        defaults['sat_sig'] = 3.0
        dtypes['sat_sig'] = [int, float]
        descr['sat_sig'] = 'Satellite trail detection threshold'

        defaults['sat_buf'] = 20
        dtypes['sat_buf'] = int
        descr['sat_buf'] = 'Satellite trail detection buffer size. Extend to the edge if each trail close to the buffer.'

        defaults['sat_order'] = 3
        dtypes['sat_order'] = int
        descr['sat_order'] = 'Satellite trail detection interpolation order when rotating image.'

        defaults['low_thresh'] = 0.1
        dtypes['low_thresh'] = [int, float]
        descr['low_thresh'] = 'Edge detection level in the normalized image for satellite identification'

        defaults['h_thresh'] = 0.5
        dtypes['h_thresh'] = [int, float]
        descr['h_thresh'] = 'Edge detection level in the normalized image for satellite identification'

        defaults['small_edge'] = 60
        dtypes['small_edge'] = int
        descr['small_edge'] = 'min_size for edge detection, used by morph.remove_small_objects '

        defaults['line_len'] = 200
        dtypes['line_len'] = int
        descr['line_len'] = 'line_length used when performing Probabilistic Hough Transformation.'

        defaults['line_gap'] = 75
        dtypes['line_gap'] = int
        descr['line_gap'] = 'line_gap used when performing Probabilistic Hough Transformation.'

        defaults['percentile'] = (4.5,93.0)
        dtypes['percentile'] = [tuple, list]
        descr['percentile'] = 'percentile range used for scaling image used byexposure.rescale_intensity.'

        ## bad pixel replacement methods
        defaults['replace'] = 'No'
        options['replace'] = ProcessImagesPar.valid_rejection_replacements()
        dtypes['replace'] = str
        descr['replace'] = 'Replace method for cosmic ray and satellite trail hitted pixels.  ' \
                           'Options are: {0}'.format(', '.join(options['replace']))

        ## Background methods
        defaults['use_medsky'] = False
        dtypes['use_medsky'] = bool
        descr['use_medsky'] = 'Use the median sky level measured by biweight_location?'

        defaults['back_type'] = 'sextractor'
        options['back_type'] = ProcessImagesPar.valid_back_type()
        dtypes['back_type'] = str
        descr['back_type'] = 'Method used to estimate backgrounds.  Options are: {0}'.format(
                                       ', '.join(options['back_type']))

        defaults['back_rms_type'] = 'STD'
        options['back_rms_type'] = ProcessImagesPar.valid_backrms_type()
        dtypes['back_rms_type'] = str
        descr['back_rms_type'] = 'Background Options are: {0}'.format(', '.join(options['back_rms_type']))

        defaults['back_size'] = (256,256)
        dtypes['back_size'] = [tuple, list, int, float]
        descr['back_size'] = 'Box size for background estimation'

        defaults['back_filtersize'] = (3,3)
        dtypes['back_filtersize'] = [tuple, list]
        descr['back_filtersize'] = 'Filter size for background estimation'

        defaults['back_maxiters'] = 5
        dtypes['back_maxiters'] = int
        descr['back_maxiters'] = 'Maximum number of iterations for background estimation.'


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
                   'use_biasimage', 'use_overscan', 'overscan_method', 'overscan_par', 'use_darkimage',
                   'use_illumflat', 'use_pixelflat', 'use_supersky', 'use_fringe', 'mask_negative_star',
                   'comb_cenfunc', 'comb_stdfunc', 'comb_maxiter', 'satpix', 'n_lohi', 'replace', 'mask_proc', 'mask_vig','minimum_vig',
                   'window_size', 'maskpixvar', 'mask_brightstar', 'brightstar_nsigma', 'brightstar_method', 'conv',
                   'mask_cr','contrast','lamaxiter', 'grow', 'clip', 'comb_sigrej',
                   'rmcompact', 'sigclip', 'sigfrac', 'objlim','cr_threshold','neighbor_threshold',
                   'mask_sat', 'sat_sig', 'sat_buf', 'sat_order', 'low_thresh', 'h_thresh',
                   'small_edge', 'line_len', 'line_gap', 'percentile', 'use_medsky',
                   'back_type', 'back_rms_type','back_size','back_filtersize','back_maxiters']

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
        return ['median', 'weightmean', 'mean', 'sum', 'min', 'max']

    @staticmethod
    def valid_combine_stdfunc():
        """
        Return the valid methods for combining frames.
        """
        return ['std']

    @staticmethod
    def valid_brightstar_methods():
        """
        Return the valid methods for combining frames.
        """
        return ['photoutils', 'sextractor' ]

    @staticmethod
    def valid_back_type():
        """
        Return the valid methods for background estimator method.
        """
        return ['MEDIAN','MEAN','SEXTRACTOR', 'MMM', 'BIWEIGHT', 'MODE', 'GlobalMedian',
                'median','mean','sextractor', 'mmm', 'biweight', 'mode']

    @staticmethod
    def valid_backrms_type():
        """
        Return the valid methods for background rms estimator method.
        """
        return ['STD', 'MAD', 'BIWEIGHT', 'std', 'mad', 'biweight']

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
        return ['zero', 'min', 'max', 'mean', 'median', 'No']

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

class AstrometricPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, skip=None, mosaic=None, scamp_second_pass=None, detect_thresh=None, analysis_thresh=None, detect_minarea=None,
                 crossid_radius=None, position_maxerr=None, pixscale_maxerr=None, mosaic_type=None,
                 astref_catalog=None, astref_band=None, astrefmag_limits=None,
                 astrefcat_name=None, astrefcent_keys=None, astreferr_keys=None, astrefmag_key=None, astrefmagerr_key=None,
                 astrefsn_limits=None, weight_type=None, solve_photom_scamp=None, match_flipped=None,
                 posangle_maxerr=None, stability_type=None, distort_degrees=None, skip_swarp_align=None, group=None,
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

        defaults['skip'] = False
        dtypes['skip'] = bool
        descr['skip'] = 'Skip the astrometry for individual detector image?'

        defaults['mosaic'] = True
        dtypes['mosaic'] = bool
        descr['mosaic'] = 'Mosaicing multiple detectors to a MEF fits before running scamp?'

        defaults['skip_swarp_align'] = True
        dtypes['skip_swarp_align'] = bool
        descr['skip_swarp_align'] = 'Skip aligning the image before solving the astrometric solutions?'

        defaults['scamp_second_pass'] = False
        dtypes['scamp_second_pass'] = bool
        descr['scamp_second_pass'] = 'Perform second pass with SCAMP? Useful for instrument with large distortions.'

        defaults['match_flipped'] = False
        dtypes['match_flipped'] = bool
        descr['match_flipped'] = 'Allow matching with flipped axes?'

        defaults['weight_type'] = 'MAP_WEIGHT'
        options['weight_type'] = AstrometricPar.valid_weight_type()
        dtypes['weight_type'] = str
        descr['weight_type'] = 'Background Options are: {0}'.format(', '.join(options['weight_type']))

        defaults['detect_thresh'] = 5.0
        dtypes['detect_thresh'] = [int, float]
        descr['detect_thresh'] = ' <sigmas> or <threshold>,<ZP> in mag.arcsec-2 for detection'

        defaults['analysis_thresh'] = 5.0
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

        defaults['posangle_maxerr'] = 10.0
        dtypes['posangle_maxerr'] = [int, float]
        descr['posangle_maxerr'] = 'Max position-angle uncertainty (deg)'

        defaults['distort_degrees'] = 3
        dtypes['distort_degrees'] = int
        descr['distort_degrees'] = 'Polynom degree for each group'

        defaults['stability_type'] = 'INSTRUMENT'
        options['stability_type'] = AstrometricPar.valid_stability_methods()
        dtypes['stability_type'] = str
        descr['stability_type'] = 'Reference catalog  Options are: {0}'.format(
                                       ', '.join(options['stability_type']))

        defaults['mosaic_type'] = 'UNCHANGED'
        options['mosaic_type'] = AstrometricPar.valid_mosaic_methods()
        dtypes['mosaic_type'] = str
        descr['mosaic_type'] = 'Reference catalog  Options are: {0}'.format(
                                       ', '.join(options['mosaic_type']))

        defaults['astref_catalog'] = 'GAIA-EDR3'
        options['astref_catalog'] = AstrometricPar.valid_catalog_methods()
        dtypes['astref_catalog'] = str
        descr['astref_catalog'] = 'Reference catalog  Options are: {0}'.format(
                                       ', '.join(options['astref_catalog']))

        defaults['astref_band'] = 'DEFAULT'
        dtypes['astref_band'] = str
        descr['astref_band'] = 'Photom. band for astr.ref.magnitudes or DEFAULT, BLUEST, or REDDEST'

        defaults['astrefmag_limits'] = [15.0,21.0]
        dtypes['astrefmag_limits'] = [tuple, list]
        descr['astrefmag_limits'] = 'magnitude limit that will be used for astrometric calibrations'

        defaults['astrefsn_limits'] = [10.0,100.0]
        dtypes['astrefsn_limits'] = [tuple, list]
        descr['astrefsn_limits'] = 'S/N thresholds (in sigmas) for all and high-SN sample'

        defaults['astrefcat_name'] = 'NONE'
        dtypes['astrefcat_name'] = str
        descr['astrefcat_name'] = 'File names of local astrometric reference catalogues ' \
                                  '(active if ASTREF CATALOG is set to FILE), through which SCAMP ' \
                                  'will browse to find astrometric reference stars.'

        defaults['astrefcent_keys'] = 'RA, DEC'
        dtypes['astref_band'] = [str, list]
        descr['astref_band'] = 'Names of the columns, in the local astrometric reference catalogue(s), ' \
                               'that contain the centroid coordinates in degrees. Active only if ASTREF CATALOG is set to FILE.'

        defaults['astreferr_keys'] = 'RA_ERR, DEC_ERR, THETA_ERR'
        dtypes['astreferr_keys'] = [str, list]
        descr['astreferr_keys'] = 'Names of the columns, in the local astrometric reference catalogue(s), ' \
                                  'that contain the major and minor axes and position angle of the error ' \
                                  'ellipses. Active only if ASTREF CATALOG is set to FILE.'

        defaults['astrefmag_key'] = 'MAG'
        dtypes['astrefmag_key'] = str
        descr['astrefmag_key'] = 'Name of the column, in the local astrometric reference catalogue(s), ' \
                                 'that contains the catalogue magnitudes. Active only if ASTREF CATALOG is set to FILE.'

        defaults['astrefmagerr_key'] = 'MAG_ERR'
        dtypes['astrefmagerr_key'] = str
        descr['astrefmagerr_key'] = 'Name of the optional column, in the local astrometric reference catalogue(s), ' \
                                    'that contains the catalogue magnitude uncertainties. Active only if ASTREF CATALOG is set to FILE.'

        defaults['solve_photom_scamp'] = False
        dtypes['solve_photom_scamp'] = bool
        descr['solve_photom_scamp'] = 'SOLVE_PHOTOM with SCAMP? I would set it to False since PyPhot will calibrate individual chip'

        defaults['group'] = True
        dtypes['group'] = bool
        descr['group'] = 'Group all exposures when doing scamp?'

        defaults['delete'] = True
        dtypes['delete'] = bool
        descr['delete'] = 'Deletec the configuration files for SExtractor, SCAMP, and SWARP?'

        defaults['log'] = True
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
        parkeys = ['skip', 'mosaic', 'scamp_second_pass', 'detect_thresh', 'analysis_thresh', 'detect_minarea', 'crossid_radius',
                   'position_maxerr', 'pixscale_maxerr', 'mosaic_type', 'astref_catalog', 'astref_band', 'astrefmag_limits',
                   'astrefcat_name', 'astrefcent_keys', 'astreferr_keys', 'astrefmag_key', 'astrefmagerr_key', 'match_flipped',
                   'astrefsn_limits','posangle_maxerr', 'stability_type', 'distort_degrees','skip_swarp_align',
                   'weight_type', 'solve_photom_scamp', 'group', 'delete', 'log']

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
    def valid_stability_methods():
        """
        Return the valid methods for SCAMP stability method.
        """
        return ['INSTRUMENT', 'EXPOSURE', 'PRE-DISTORTED']

    @staticmethod
    def valid_catalog_methods():
        """
        Return the valid methods for reference catalog.
        """
        return ['NONE', 'FILE', 'USNO-A2','USNO-B1','GSC-2.3','TYCHO-2','UCAC-4','URAT-1','NOMAD-1','PPMX',
                'CMC-15','2MASS', 'DENIS-3', 'SDSS-R9','SDSS-R12','IGSL','GAIA-DR1','GAIA-DR2','GAIA-EDR3',
                'PANSTARRS-1','ALLWISE', 'LS-DR9','DES-DR2']

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
    def __init__(self, skip=None, weight_type=None, rescale_weights=None, combine_type=None,
                 clip_ampfrac=None, clip_sigma=None, blank_badpixels=None, subtract_back=None, back_type=None,
                 back_default=None, back_size=None, back_filtersize=None, back_filtthresh=None, resampling_type=None,
                 pixscale=None, cal_zpt=None, delete=None, log=None):

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

        defaults['skip'] = False
        dtypes['skip'] = bool
        descr['skip'] = 'Skip Coadding science targets?'

        defaults['weight_type'] = 'MAP_WEIGHT'
        options['weight_type'] = CoaddPar.valid_weight_type()
        dtypes['weight_type'] = str
        descr['weight_type'] = 'Background Options are: {0}'.format(', '.join(options['weight_type']))

        defaults['rescale_weights'] = True
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

        defaults['back_size'] = 100
        dtypes['back_size'] = [int, float, tuple, list]
        descr['back_size'] = 'Default background value in MANUAL'

        defaults['back_filtersize'] = 3
        dtypes['back_filtersize'] = [int, float]
        descr['back_filtersize'] = 'Background map filter range (meshes)'

        defaults['back_filtthresh'] = 0.0
        dtypes['back_filtthresh'] = [int, float]
        descr['back_filtthresh'] = 'Threshold above which the background map filter operates'

        defaults['resampling_type'] = 'LANCZOS3'
        options['resampling_type'] = CoaddPar.valid_resampling_type()
        dtypes['resampling_type'] = str
        descr['resampling_type'] = 'Swarp resampling type options are: {0}'.format(', '.join(options['resampling_type']))

        defaults['pixscale'] = None
        dtypes['pixscale'] = [int, float]
        descr['pixscale'] = 'pixel scale for the final coadd image'

        defaults['cal_zpt'] = True
        dtypes['cal_zpt'] = bool
        descr['cal_zpt'] = 'Calibrating the zeropoint for coadded image'

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
        parkeys = ['skip', 'weight_type','rescale_weights', 'combine_type', 'clip_ampfrac', 'clip_sigma',
                   'blank_badpixels','subtract_back', 'back_type', 'back_default', 'back_size','back_filtersize',
                   'back_filtthresh','resampling_type', 'pixscale', 'cal_zpt', 'delete', 'log']

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
          WEIGHTED: yields the best S/N
          AVERAGE: better S/N than MEDIAN but sensitive to outliers
          MEDIAN: worse S/N than AVERAGE but stable against outliers
          MIN: use the smallest pixel value in the stack
          MAX: use the highest pixel value in the stack
          CHI2: uses chi-square for the stack (all pixel values will be non-negative)
        """
        return ['MEDIAN', 'AVERAGE', 'MIN','MAX','WEIGHTED','CLIPPED','CHI-OLD','CHI-MODE','CHI-MEAN','SUM',
                'WEIGHTED_WEIGHT','MEDIAN_WEIGHT', 'AND', 'NAND','OR','NOR']

    @staticmethod
    def valid_resampling_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['NEAREST', 'BILINEAR', 'LANCZOS2', 'LANCZOS3', 'LANCZOS4', 'FLAGS']

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
    def __init__(self, skip=None, detection_method=None, phot_apertures=None, detect_thresh=None, back_type=None, analysis_thresh=None,
                 back_default=None, back_size=None, back_filtersize=None, detect_minarea=None,check_type=None,
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

        defaults['skip'] = False
        dtypes['skip'] = bool
        descr['skip'] = 'Skip detecting sources?'

        defaults['detection_method'] = 'SExtractor'
        options['detection_method'] = DetectionPar.valid_detection_method()
        dtypes['detection_method'] = str
        descr['detection_method'] = 'Background Options are: {0}'.format(', '.join(options['detection_method']))

        ## parameters for all methods
        defaults['phot_apertures'] = [1.0, 2.0, 3.0, 4.0, 5.0]
        dtypes['phot_apertures'] = [int, float, list]
        descr['phot_apertures'] = 'Photometric apertures in units of arcsec'

        defaults['detect_thresh'] = 1.5
        dtypes['detect_thresh'] = [int, float]
        descr['detect_thresh'] = ' <sigmas> or <threshold> for detection'

        defaults['analysis_thresh'] = 1.5
        dtypes['analysis_thresh'] = [int, float]
        descr['analysis_thresh'] = ' <sigmas> or <threshold>,<ZP> in mag.arcsec-2 for analysis'

        defaults['back_type'] = 'AUTO'
        options['back_type'] = DetectionPar.valid_back_type()
        dtypes['back_type'] = str
        descr['back_type'] = 'Background Options are: {0}'.format(', '.join(options['back_type']))

        defaults['back_default'] = 0.0
        dtypes['back_default'] = [int, float]
        descr['back_default'] = 'Default background value in MANUAL'

        defaults['back_size'] = 100
        dtypes['back_size'] = [int, float, tuple, list]
        descr['back_size'] = 'Default background value in MANUAL, int for SExtractor and tuple for Others'

        defaults['back_filtersize'] = 3
        dtypes['back_filtersize'] = [int, float, tuple]
        descr['back_filtersize'] = 'Background map filter range (meshes), int for SExtractor and tuple for Others'

        ## parameters used by SExtractor or Photoutils
        defaults['detect_minarea'] = 3
        dtypes['detect_minarea'] = [int, float]
        descr['detect_minarea'] = 'min. # of pixels above threshold'

        defaults['fwhm'] = 5
        dtypes['fwhm'] = [int, float]
        descr['fwhm'] = '# of pixels of seeing'

        defaults['nlevels'] = 32
        dtypes['nlevels'] = int
        descr['nlevels'] = 'Number of deblending sub-thresholds'

        defaults['contrast'] =  0.001
        dtypes['contrast'] = float
        descr['contrast'] = ' Minimum contrast parameter for deblending'

        ## parameters used by SExtractor only
        defaults['check_type'] = 'OBJECTS'
        options['check_type'] = DetectionPar.valid_check_type()
        dtypes['check_type'] = str
        descr['check_type'] = 'Background Options are: {0}'.format(', '.join(options['check_type']))

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

        defaults['conv'] = 'sex'
        dtypes['conv'] = str
        descr['conv'] = 'Convolution matrix, either default sex, or sex995 or you can provide the full path of your conv file'

        defaults['nnw'] = 'sex'
        dtypes['nnw'] = str
        descr['nnw'] = 'Use SExtractor default configuration file or you can provide the full path of your nnw file'

        defaults['delete'] = False
        dtypes['delete'] = bool
        descr['delete'] = 'Deletec the configuration files for SExtractor?'

        defaults['log'] = True
        dtypes['log'] = bool
        descr['log'] = 'Logging for SExtractor?'

        ## parameters used by Photutils only
        defaults['back_rms_type'] = 'STD'
        options['back_rms_type'] = DetectionPar.valid_backrms_type()
        dtypes['back_rms_type'] = str
        descr['back_rms_type'] = 'Background Options are: {0}'.format(', '.join(options['back_rms_type']))

        defaults['back_nsigma'] = 3
        dtypes['back_nsigma'] = [int, float]
        descr['back_nsigma'] = 'nsigma for sigma clipping background, used by Photutils only'

        defaults['back_maxiters'] = 10
        dtypes['back_maxiters'] = int
        descr['back_maxiters'] = 'maxiters for sigma clipping backgroun, used by Photutils only'

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
        parkeys = ['skip','detection_method', 'phot_apertures', 'detect_thresh', 'back_type', 'back_default', 'analysis_thresh',
                   'back_size', 'back_filtersize', 'detect_minarea', 'check_type','weight_type','backphoto_type',
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
        return ['Photutils', 'SExtractor', 'photutils', 'sextractor', 'DAOStar', 'IRAFStar']

    @staticmethod
    def valid_check_type():
        """
        Return the valid methods for mosaic method.
        """
        return ['NONE', 'BACKGROUND', 'BACKGROUND_RMS','MINIBACKGROUND','SEXTRACTOR', 'MINIBACK_RMS', ' -BACKGROUND']

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

class PhotometryPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, skip=None, photref_catalog=None, zpt=None, external_flag=None,
                 primary=None, secondary=None, coefficients=None, coeff_airmass=None, nstar_min=None):

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

        defaults['skip'] = False
        dtypes['skip'] = bool
        descr['skip'] = 'Skip detecting sources?'

        defaults['external_flag'] = True
        dtypes['external_flag'] = bool
        descr['external_flag'] = 'Apply external flag cut when calibrating zeropoint? '

        defaults['photref_catalog'] = 'Panstarrs'
        options['photref_catalog'] = PhotometryPar.valid_catalog_methods()
        dtypes['photref_catalog'] = str
        descr['photref_catalog'] = 'Background Options are: {0}'.format(', '.join(options['photref_catalog']))

        defaults['zpt'] = 0.
        dtypes['zpt'] = [int, float]
        descr['zpt'] = 'Zero point'

        defaults['nstar_min'] = 10.
        dtypes['nstar_min'] = [int, float]
        descr['nstar_min'] = 'The minimum number of stars used for photometric calibrations'

        defaults['primary'] = 'r'
        dtypes['primary'] = str
        descr['primary'] = 'Primary calibration filter'

        defaults['secondary'] = 'i'
        dtypes['secondary'] = str
        descr['secondary'] = 'Secondary calibration filter'

        defaults['coefficients'] = [0.,0.,0.]
        dtypes['coefficients'] = [tuple, list]
        descr['coefficients'] = 'Color-term coefficients, i.e. mag = primary+c0+c1*(primary-secondary)+c1*(primary-secondary)**2'

        defaults['coeff_airmass'] = 0.
        dtypes['coeff_airmass'] = [int, float]
        descr['coeff_airmass'] = 'Extinction-term coefficient, i.e. mag_real=mag_obs-coeff_airmass*airmass'


        # Instantiate the parameter set
        super(PhotometryPar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['skip', 'photref_catalog', 'zpt', 'external_flag', 'primary', 'secondary',
                   'coefficients', 'coeff_airmass', 'nstar_min']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for DetectionPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

    @staticmethod
    def valid_catalog_methods():
        """
        Return the valid methods for reference catalog.
        """
        return ['Twomass', 'SDSS', 'Gaia', 'Panstarrs', 'LEGACY', 'ALLWISE']

    def validate(self):
        """
        Check the parameters are valid for the provided method.
        """
        pass

class QAPar(ParSet):
    """
    A parameter set holding the arguments for how to perform the flux
    calibration.

    For a table with the current keywords, defaults, and descriptions,
    see :ref:`pyphotpar`.
    """
    def __init__(self, skip=None, vmin = None, vmax = None, interval_method=None, stretch_method=None, cmap=None,
                 plot_wcs=None, show=None):

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

        defaults['skip'] = False
        dtypes['skip'] = bool
        descr['skip'] = 'Skip producing QA plots?'



        defaults['plot_wcs'] = True
        dtypes['plot_wcs'] = bool
        descr['plot_wcs'] = 'Using WCS information?'

        defaults['vmin'] = None
        dtypes['vmin'] = [int, float]
        descr['vmin'] = 'vmin used for the plot'

        defaults['vmax'] = None
        dtypes['vmax'] = [int, float]
        descr['vmax'] = 'vmax used for the plot'

        defaults['show'] = False
        dtypes['show'] = bool
        descr['show'] = 'Show the QA plot?'

        defaults['interval_method'] = 'zscale'
        dtypes['interval_method'] = str
        descr['interval_method'] = 'interval method when showing image'

        defaults['stretch_method'] = 'linear'
        dtypes['stretch_method'] = str
        descr['stretch_method'] = 'stretching method when showing image'

        defaults['cmap'] = 'gist_yarg_r'
        dtypes['cmap'] = str
        descr['cmap'] = 'color map used for showing image'

        # Instantiate the parameter set
        super(QAPar, self).__init__(list(pars.keys()),
                                                 values=list(pars.values()),
                                                 defaults=list(defaults.values()),
                                                 dtypes=list(dtypes.values()),
                                                 descr=list(descr.values()))
        self.validate()

    @classmethod
    def from_dict(cls, cfg):
        k = numpy.array([*cfg.keys()])
        parkeys = ['skip','vmin', 'vmax', 'interval_method', 'stretch_method', 'cmap', 'plot_wcs', 'show']
        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for DetectionPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None
        return cls(**kwargs)

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
                 qadir=None, coadddir=None, redux_path=None, ignore_bad_headers=None,
                 skip_master=None, skip_detproc=None, skip_sciproc=None, skip_astrometry=None,
                 skip_chipcal=None, skip_img_qa=None, skip_coadd=None, skip_detection=None,
                 n_process=None):

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
        descr['detnum'] = 'Restrict reduction to a list of detector indices.'

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

        defaults['skip_master'] = False
        dtypes['skip_master'] = bool
        descr['skip_master'] = 'Skip building all the master calibrations?'

        defaults['skip_detproc'] = False
        dtypes['skip_detproc'] = bool
        descr['skip_detproc'] = 'Skip detproc for all science chips?'

        defaults['skip_sciproc'] = False
        dtypes['skip_sciproc'] = bool
        descr['skip_sciproc'] = 'Skip sciproc for all science chips?'

        defaults['skip_astrometry'] = False
        dtypes['skip_astrometry'] = bool
        descr['skip_astrometry'] = 'Skip astrometry for all science chips?'

        defaults['skip_chipcal'] = False
        dtypes['skip_chipcal'] = bool
        descr['skip_chipcal'] = 'Skip zeropoint calibrations for individual science chips?'

        defaults['skip_coadd'] = False
        dtypes['skip_coadd'] = bool
        descr['skip_coadd'] = 'Skip coadding and mosaiking?'

        defaults['skip_detection'] = False
        dtypes['skip_detection'] = bool
        descr['skip_detection'] = 'Skip extracting photometric catalog from coadded images'

        defaults['skip_img_qa'] = False
        dtypes['skip_img_qa'] = bool
        descr['skip_img_qa'] = 'Skip producing QA for resampled images?'

        defaults['n_process'] = 4
        dtypes['n_process'] = int
        descr['n_process'] = 'Number of process for the parallel processing. Several core functions are paralleled.'\
                             'Including detproc, sciproc, astrometric, cal_chips, and show_images'

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
                    'redux_path', 'ignore_bad_headers',
                    'skip_master','skip_detproc','skip_sciproc','skip_astrometry', 'skip_chipcal', 'skip_img_qa',
                    'skip_coadd', 'skip_detection','n_process']

        badkeys = numpy.array([pk not in parkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for ReduxPar.'.format(k[badkeys]))

        kwargs = {}
        for pk in parkeys:
            kwargs[pk] = cfg[pk] if pk in k else None

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

    def __init__(self, astrometry=None, coadd=None, detection=None, photometry=None, qa=None):

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

        defaults['astrometry'] = AstrometricPar()
        dtypes['astrometry'] = [ParSet, dict]
        descr['astrometry'] = 'Parameters for solving astrometric solutions.'

        defaults['coadd'] = CoaddPar()
        dtypes['coadd'] = [ParSet, dict]
        descr['coadd'] = 'Parameters for coadding science images.'

        defaults['detection'] = DetectionPar()
        dtypes['detection'] = [ParSet, dict]
        descr['detection'] = 'Parameters for solving detections.'

        defaults['photometry'] = PhotometryPar()
        dtypes['photometry'] = [ParSet, dict]
        descr['photometry'] = 'Parameters for solving photometry.'

        defaults['qa'] = QAPar()
        dtypes['qa'] = [ParSet, dict]
        descr['qa'] = 'Parameters for solving photometry.'

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

        allkeys = ['astrometry', 'coadd', 'detection', 'photometry', 'qa']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for ReducePar.'.format(k[badkeys]))

        kwargs = {}
        pk = 'astrometry'
        kwargs[pk] = AstrometricPar.from_dict(cfg[pk]) if pk in k else None
        pk = 'coadd'
        kwargs[pk] = CoaddPar.from_dict(cfg[pk]) if pk in k else None
        pk = 'detection'
        kwargs[pk] = DetectionPar.from_dict(cfg[pk]) if pk in k else None
        pk = 'photometry'
        kwargs[pk] = PhotometryPar.from_dict(cfg[pk]) if pk in k else None
        pk = 'qa'
        kwargs[pk] = QAPar.from_dict(cfg[pk]) if pk in k else None

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
                 superskyframe=None, fringeframe=None,
                 standardframe=None, raise_chk_error=None):


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
                                                                       comb_cenfunc='median',
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

        defaults['superskyframe'] = FrameGroupPar(frametype='supersky',
                                                  process=ProcessImagesPar(satpix='nothing',
                                                                           use_pixelflat=False))
        dtypes['superskyframe'] = [ ParSet, dict ]
        descr['superskyframe'] = 'The frames and combination rules for the illumination flat'


        defaults['fringeframe'] = FrameGroupPar(frametype='fringe',
                                                process=ProcessImagesPar(satpix='nothing',
                                                                         use_pixelflat=False))
        dtypes['fringeframe'] = [ ParSet, dict ]
        descr['fringeframe'] = 'The frames and combination rules for the illumination flat'


        defaults['standardframe'] = FrameGroupPar(frametype='standard',
                                                  process=ProcessImagesPar(mask_cr=True))
        dtypes['standardframe'] = [ ParSet, dict ]
        descr['standardframe'] = 'The frames and combination rules for the spectrophotometric ' \
                                 'standard observations'

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

        allkeys = parkeys + ['biasframe', 'darkframe', 'pixelflatframe','illumflatframe',
                             'superskyframe','fringeframe','standardframe']
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
        pk = 'superskyframe'
        kwargs[pk] = FrameGroupPar.from_dict('supersky', cfg[pk]) if pk in k else None
        pk = 'fringeframe'
        kwargs[pk] = FrameGroupPar.from_dict('fringe', cfg[pk]) if pk in k else None
        pk = 'standardframe'
        kwargs[pk] = FrameGroupPar.from_dict('standard', cfg[pk]) if pk in k else None

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
    def __init__(self, rdx=None, calibrations=None, scienceframe=None, postproc=None):

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
        descr['postproc'] = 'Parameters for astrometry, coadding, and photometry.'

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

        allkeys = ['rdx', 'calibrations', 'scienceframe', 'postproc', 'baseprocess']
        badkeys = numpy.array([pk not in allkeys for pk in k])
        if numpy.any(badkeys):
            raise ValueError('{0} not recognized key(s) for PyPhotPar.'.format(k[badkeys]))

        kwargs = {}

        pk = 'rdx'
        kwargs[pk] = ReduxPar.from_dict(cfg[pk]) if pk in k else None

        pk = 'calibrations'
        kwargs[pk] = CalibrationsPar.from_dict(cfg[pk]) if pk in k else None

        pk = 'scienceframe'
        kwargs[pk] = FrameGroupPar.from_dict('science', cfg[pk]) if pk in k else None

        pk = 'postproc'
        kwargs[pk] = PostProcPar.from_dict(cfg[pk]) if pk in k else None

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
        return [ 'CFHT','GEMINI-N','GEMINI-S', 'KECK', 'SHANE', 'WHT', 'APF', 'TNG', 'VLT', 'MAGELLAN', 'LBT', 'MMT', 'KPNO', 'NOT', 'P200']

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


