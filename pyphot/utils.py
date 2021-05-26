import itertools

import numpy as np
from scipy.optimize import curve_fit

from collections import deque
from bisect import insort, bisect_left

from astropy.stats import sigma_clip

from pyphot import msgs

def inverse(array):
    """

    Calculate and return the inverse of the input array, enforcing
    positivity and setting values <= 0 to zero.  The input array should
    be a quantity expected to always be positive, like a variance or an
    inverse variance. The quantity::

        out = (array > 0.0)/(np.abs(array) + (array == 0.0))

    is returned.

    Args:
        a (np.ndarray):

    Returns:
        np.ndarray:

    """
    return (array > 0.0)/(np.abs(array) + (array == 0.0))


def smooth(x, window_len, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that edge effects are minimize at the beginning and end part of the signal.

     This code taken from this cookbook and slightly modified: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    .. todo::
        the window parameter could be the window itself if an array instead of a string

    Args:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing., default is 'flat'

    Returns:
        the smoothed signal, same shape as x

    Examples:

        >>> t=linspace(-2,2,0.1)
        >>> x=sin(t)+randn(len(t))*0.1
        >>> y=smooth(x)

    Notes:

        - See also: numpy.hanning, numpy.hamming, numpy.bartlett,
          numpy.blackman, numpy.convolve scipy.signal.lfilter

        - length(output) != length(input), to correct this, return
          y[(window_len/2-1):-(window_len/2)] instead of just y.

    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')

    return y[(window_len-1):(y.size-(window_len-1))]


def fast_running_median(seq, window_size):
    """

    Compute the median of sequence of numbers with a running window. The
    boundary conditions are identical to the scipy 'reflect' boundary
    codition:

    'reflect' (`d c b a | a b c d | d c b a`)

    The input is extended by reflecting about the edge of the last pixel.

    This code has been confirmed to produce identical results to
    scipy.ndimage.filters.median_filter with the reflect boundary
    condition, but is ~ 100 times faster.

    Args:
        seq (list or 1-d numpy array of numbers):
        window_size (int): size of running window.

    Returns:
        ndarray: median filtered values

    Code contributed by Peter Otten, made to be consistent with
    scipy.ndimage.filters.median_filter by Joe Hennawi.

    See discussion at:
    http://groups.google.com/group/comp.lang.python/browse_thread/thread/d0e011c87174c2d0
    """
    # Enforce that the window_size needs to be smaller than the sequence, otherwise we get arrays of the wrong size
    # upon return (very bad). Added by JFH. Should we print out an error here?

    if (window_size > (len(seq)-1)):
        msgs.warn('window_size > len(seq)-1. Truncating window_size to len(seq)-1, but something is probably wrong....')
    if (window_size < 0):
        msgs.warn('window_size is negative. This does not make sense something is probably wrong. Setting window size to 1')

    window_size = int(np.fmax(np.fmin(int(window_size), len(seq)-1),1))
    # pad the array for the reflection
    seq_pad = np.concatenate((seq[0:window_size][::-1],seq,seq[-1:(-1-window_size):-1]))

    seq_pad = iter(seq_pad)
    d = deque()
    s = []
    result = []
    for item in itertools.islice(seq_pad, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq_pad:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])

    # This takes care of the offset produced by the original code deducec by trial and error comparison with
    # scipy.ndimage.filters.medfilt

    result = np.roll(result, -window_size//2 + 1)
    return result[window_size:-window_size]

def subsample(frame):
    """
    Used by LACosmic

    Args:
        frame (ndarray):

    Returns:
        ndarray: Sliced image

    """
    newshape = (2*frame.shape[0], 2*frame.shape[1])
    slices = [slice(0, old, float(old)/new) for old, new in zip(frame.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')
    return frame[tuple(indices)]

def rebin(a, newshape):
    """

    Rebin an array to a new shape using slicing. This routine is taken
    from: https://scipy-cookbook.readthedocs.io/items/Rebinning.html.
    The image shapes need not be integer multiples of each other, but in
    this regime the transformation will not be reversible, i.e. if
    a_orig = rebin(rebin(a,newshape), a.shape) then a_orig will not be
    everywhere equal to a (but it will be equal in most places).

    Args:
        a (ndarray, any dtype):
            Image of any dimensionality and data type
        newshape (tuple):
            Shape of the new image desired. Dimensionality must be the
            same as a.

    Returns:
        ndarray: same dtype as input Image with same values as a
        rebinning to shape newshape
    """
    if not len(a.shape) == len(newshape):
        msgs.error('Dimension of a image does not match dimension of new requested image shape')

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  # choose the biggest smaller integer index
    return a[tuple(indices)]

# TODO This function is only used by procimg.lacosmic. Can it be replaced by above?
def rebin_evlist(frame, newshape):
    # This appears to be from
    # https://scipy-cookbook.readthedocs.io/items/Rebinning.html
    shape = frame.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(newshape)
    evList = ['frame.reshape('] + \
             ['int(newshape[%d]),int(factor[%d]),'% (i, i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)' % (i+1) for i in range(lenShape)] + \
             ['/factor[%d]' % i for i in range(lenShape)]
    return eval(''.join(evList))


def robust_curve_fit(func, xx, yy, niters=5, sigclip=3, maxiters_sigclip=5, cenfunc='median', stdfunc='std',
                     p0=None, sigma=None, absolute_sigma=None, bounds=None, method=None, jac=None, **kwargs):

    if bounds is None:
        bounds = (-np.inf, np.inf)
    ii=0
    xx_mask, yy_mask = xx.copy(), yy.copy()
    while ii < niters:
        popt, pcov = curve_fit(func, xx_mask, yy_mask, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,
                               bounds=bounds, method=method, jac=jac, **kwargs)
        diff = abs(func(xx_mask,popt[0],popt[1]) - yy_mask)

        masked_data = sigma_clip(diff, sigma=sigclip, maxiters=maxiters_sigclip, masked=True,
                                 cenfunc=cenfunc, stdfunc=stdfunc, copy=True)

        xx_mask, yy_mask = xx_mask[np.invert(masked_data.mask)], yy_mask[np.invert(masked_data.mask)]
        ii +=1

    return popt, pcov