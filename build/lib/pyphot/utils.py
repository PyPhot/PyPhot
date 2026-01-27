import itertools

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt

from collections import deque
from bisect import insort, bisect_left

import multiprocessing
from multiprocessing import Process, Queue

from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.stats import biweight_location, biweight_midvariance
from astropy.visualization import MinMaxInterval, ManualInterval, ZScaleInterval
from astropy.visualization import AsinhStretch, HistEqStretch, LinearStretch, LogStretch
from astropy.visualization import PowerDistStretch, PowerStretch, SinhStretch, SqrtStretch
from astropy.visualization import ImageNormalize


from pyphot import msgs, io

def pyplot_rcparams():
    """
    params for pretty matplotlib plots

    Returns:

    """
    # set some plotting parameters
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.direction"] = 'in'
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.handletextpad"] = 1
    plt.rcParams["legend.handlelength"] = 1.1
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.default"] = "regular"

def pyplot_rcparams_default():
    """
    restore default rcparams

    Returns:

    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


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

def gain_correct(data, datasec_img, gain):
    '''
    Convert data from ADU to e-
    Parameters
    ----------
    data
    datasec_img
    gain

    Returns
    -------

    '''

    numamplifiers = np.size(gain)
    for iamp in range(numamplifiers):
        this_amp = datasec_img == iamp + 1
        data[this_amp] *= gain[iamp]

    return data

def pixel_stats(pixels, bpm=None, sigclip=3, n_clip=10, min_pix=50):
    """
    Calculate image statistics to determine median sky level and RMS noise. Uses
    biweight as "robust" estimator of these quantities.

    :param pixels: Array to calculate statistics for
    :param sigclip: Sigma value at which to clip outliers
    :param n_clip: Number of clipping iterations
    :param min_pix: Minimum number of retained pixels
    :return: 2-tuple of distribution mode, scale
    """
    clip_iter = 0
    sky, rms = 0, 1
    if bpm is None:
        bpm = np.ones(pixels.shape, dtype=bool)
    gpm = np.invert(bpm)
    while True:
        sky = biweight_location(pixels[gpm], ignore_nan=True)
        rms = np.sqrt(biweight_midvariance(pixels[gpm]))
        gpm &= np.abs(pixels - sky) < sigclip * rms
        clip_iter += 1
        if np.sum(gpm) < min_pix or clip_iter >= n_clip:
            break
    return sky, rms

def gauss1D(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean)**2 / (2*stddev**2))) + offset


def gauss2D(xy_tuple, amplitude, xo, yo, sigma, theta, offset):

    (x, y)= xy_tuple
    if np.size(sigma) > 1:
        sigma_x, sigma_y = sigma[0], sigma[1]
    else:
        sigma_x, sigma_y = sigma, sigma
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo) ** 2)))
    return g.ravel()


def robust_curve_fit(func, xx, yy, niters=5, sigclip=3, maxiters_sigclip=5, cenfunc='median', stdfunc='std',
                     p0=None, sigma=None, absolute_sigma=None, bounds=None, method=None, jac=None, **kwargs):

    if bounds is None:
        bounds = (-np.inf, np.inf)
    ii=0
    xx_mask, yy_mask = xx.copy(), yy.copy()
    if sigma is None:
        sigma_mask = None
    else:
        sigma_mask = sigma.copy()

    while ii < niters:
        popt, pcov = curve_fit(func, xx_mask, yy_mask, p0=p0, sigma=sigma_mask, absolute_sigma=absolute_sigma,
                               bounds=bounds, method=method, jac=jac, **kwargs)
        diff = abs(func(xx_mask,*popt) - yy_mask)

        masked_data = sigma_clip(diff, sigma=sigclip, maxiters=maxiters_sigclip, masked=True,
                                 cenfunc=cenfunc, stdfunc=stdfunc, copy=True)

        if len(xx.shape) == len(yy.shape):
            xx_mask, yy_mask = xx_mask[np.invert(masked_data.mask)], yy_mask[np.invert(masked_data.mask)]
        elif len(xx.shape) == 2*len(yy.shape):
            xx_mask, yy_mask = xx_mask[:,np.invert(masked_data.mask)], yy_mask[np.invert(masked_data.mask)]
        else:
            msgs.error('Only 1D and 2D models are acceptted.')

        if sigma_mask is not None:
            sigma_mask = sigma_mask[np.invert(masked_data.mask)]

        ii +=1

    return popt, pcov

def showimage(image, header=None, outroot=None, interval_method='zscale', vmin=None, vmax=None,
              stretch_method='linear', cmap='gist_yarg_r', plot_wcs=True, show=False, verbose=False):

    plt.rcParams["ytick.direction"] = 'in'
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["font.family"] = "Times New Roman"

    # if only one image, set it to list
    if isinstance(image, str):
        msgs.info('Plotting image {:}'.format(image))
        header, data, flag = io.load_fits(image)
    else:
        data = image

    ny, nx = data.shape

    # Create interval object
    if interval_method.lower() == 'zscale':
        interval = ZScaleInterval()
    elif interval_method.lower() == 'minmax':
        interval = MinMaxInterval()
    else:
        interval = ManualInterval(vmin=vmin, vmax=vmax)

    vmin, vmax = interval.get_limits(data)
    if verbose:
        msgs.info('Using vmin={:} and vmax={:} for the plot'.format(vmin, vmax))

    if stretch_method.lower()=='asinh':
        stretch = AsinhStretch()
    elif stretch_method.lower()=='histeq':
        stretch = HistEqStretch()
    elif stretch_method.lower()=='linear':
        stretch = LinearStretch()
    elif stretch_method.lower()=='log':
        stretch = LogStretch()
    elif stretch_method.lower()=='powerdist':
        stretch = PowerDistStretch()
    elif stretch_method.lower()=='power':
        stretch = PowerStretch()
    elif stretch_method.lower()=='sinh':
        stretch = SinhStretch()
    elif stretch_method.lower()=='sqrt':
        stretch = SqrtStretch()
    else:
        msgs.error('Please use one of the following stretch: asinh, histeq, linear, log, powerdist, power, sinh, sqrt.')

    if verbose:
        msgs.info('Using {:} stretch for the plot'.format(stretch_method))

    # making the plot
    f = plt.figure(figsize=(4, 3.5*ny/nx))
    f.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.98, wspace=0, hspace=0)
    if plot_wcs and header is not None:
        this_wcs = WCS(header)
        plt.subplot(projection=this_wcs)
        plt.xlabel('RA', fontsize=14)
        plt.ylabel('DEC', fontsize=14)
    else:
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)

    # Create an ImageNormalize object using a SqrtStretch object
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)

    plt.imshow(data, origin='lower', norm=norm, cmap=cmap)

    if outroot is not None:
        if verbose:
            msgs.info('Saving the QA to {:}.pdf'.format(outroot))
        plt.savefig('{:}.pdf'.format(outroot))
    if show:
        plt.show()
    plt.close()

def showimages(fitsimages, n_process=4, outroots=None, interval_method='zscale', vmin=None, vmax=None,
               stretch_method='linear', cmap='gist_yarg_r', plot_wcs=True, show=False, verbose=True):

    n_file = len(fitsimages)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    if outroots is not None:
        if len(fitsimages) != len(outroots):
            msgs.error('The length of outroots should be the same with the number of fitsimages.')
    else:
        outroots = [None] * n_file

    if n_process == 1:
        for ii, scifile in enumerate(fitsimages):
            showimage(scifile, outroot=outroots[ii], interval_method=interval_method,
                      vmin=vmin, vmax=vmax, stretch_method=stretch_method, cmap=cmap,
                      plot_wcs=plot_wcs, show=show, verbose=verbose)
    else:
        msgs.info('Start parallel processing with n_process={:}'.format(n_process))
        work_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put((fitsimages[ii], outroots[ii]))

        # creating processes
        for w in range(n_process):
            p = Process(target=_showimage_worker, args=(work_queue,), kwargs={
                'interval_method': interval_method, 'vmin': vmin, 'vmax': vmax, 'stretch_method': stretch_method,
                'cmap': cmap, 'plot_wcs': plot_wcs, 'show': False, 'verbose':False})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

def _showimage_worker(work_queue, interval_method='zscale', vmin=None, vmax=None,
              stretch_method='linear', cmap='gist_yarg_r', plot_wcs=True, show=False, verbose=False):

    """Multiprocessing worker for sciproc."""
    while not work_queue.empty():
        image, outroot = work_queue.get()
        showimage(image, outroot=outroot, header=None, interval_method=interval_method,
                  vmin=vmin, vmax=vmax, stretch_method=stretch_method, cmap=cmap,
                  plot_wcs=plot_wcs, show=show, verbose=False)