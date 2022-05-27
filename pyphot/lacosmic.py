# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to remove cosmic rays from an astronomical image using the
L.A.Cosmic algorithm (van Dokkum 2001; PASP 113, 1420).
This code was downloaded from https://github.com/larrybradley/lacosmic and with some modification.
"""
import gc
import numpy as np
from scipy import ndimage

from pyphot import msgs, utils

def lacosmic(data, contrast, cr_threshold, neighbor_threshold,
             error=None, mask=None, background=None, effective_gain=None,
             readnoise=None, maxiter=4, border_mode='mirror', verbose=True):
    """
    Remove cosmic rays from an astronomical image using the L.A.Cosmic
    algorithm.

    The `L.A.Cosmic algorithm
    <http://www.astro.yale.edu/dokkum/lacosmic/>`_ is based on Laplacian
    edge detection and is described in `van Dokkum (2001; PASP 113,
    1420)
    <https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract>`_.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    contrast : float
        Contrast threshold between the Laplacian image and the
        fine-structure image.  If your image is critically sampled, use
        a value around 2.  If your image is undersampled (e.g., HST
        data), a value of 4 or 5 (or more) is more appropriate.  If your
        image is oversampled, use a value between 1 and 2.  For details,
        please see `PASP 113, 1420 (2001)
        <https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract>`_,
        which calls this parameter :math:`f_{\\mbox{lim}}`.  In
        particular, Figure 4 shows the approximate relationship between
        the ``contrast`` parameter and the full-width half-maximum (in
        pixels) of stars in your image.

    cr_threshold : float
        The Laplacian signal-to-noise ratio threshold for cosmic-ray
        detection.

    neighbor_threshold : float
        The Laplacian signal-to-noise ratio threshold for detection of
        cosmic rays in pixels neighboring the initially-identified
        cosmic rays.

    error : array_like, optional
        The 1-sigma errors of the input ``data``.  If ``error`` is not
        input, then ``effective_gain`` and ``readnoise`` will be used to
        construct an approximate model of the ``error``.  If ``error``
        is input, it will override the ``effective_gain`` and
        ``readnoise`` parameters.  ``error`` must have the same shape as
        ``data``.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when identifying cosmic rays.  It is
        highly recommended that saturated stars be included in ``mask``.

    background : float or array_like, optional
        The background level previously subtracted from the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  If the input
        ``data`` has not been background subtracted, then set
        ``background=None`` (default).

    effective_gain : float, array-like, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data``.  For example, if your input ``data`` are in units of
        ADU, then ``effective_gain`` should represent electrons/ADU.  If
        your input ``data`` are in units of electrons/s then
        ``effective_gain`` should be the exposure time (or an exposure
        time map).  ``effective_gain`` and ``readnoise`` must be
        specified if ``error`` is not input.

    readnoise : float, optional
        The read noise (in electrons) in the input ``data``.
        ``effective_gain`` and ``readnoise`` must be specified if
        ``error`` is not input.

    maxiter : float, optional
        The maximum number of iterations.  The default is 4.  The
        routine will automatically exit if no additional cosmic rays are
        identified in an interation.  If the routine is still
        identifying cosmic rays after four iterations, then you are
        likely digging into sources (e.g., saturated stars) and/or the
        noise.  In that case, try inputing a ``mask`` or increasing the
        value of ``cr_threshold``.

    border_mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode in which the array borders are handled during
        convolution and median filtering.  For 'constant', the fill
        value is 0.  The default is 'mirror', which matches the original
        L.A.Cosmic algorithm.

    Returns
    -------
    cleaned_image : `~numpy.ndarray`
        The cosmic-ray cleaned image.

    crmask : `~numpy.ndarray` (bool)
        A mask image of the identified cosmic rays.  Cosmic-ray pixels
        have a value of `True`.
    """

    block_size = 2.0
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

    clean_data = data.copy()
    if background is not None:
        clean_data += background
    final_crmask = np.zeros(data.shape, dtype=bool)

    if error is not None:
        if data.shape != error.shape:
            msgs.error('error and data must have the same shape')
    clean_error_image = error

    ncosmics, ncosmics_tot = 0, 0
    for iteration in range(maxiter):
        sampled_img = block_replicate(clean_data, block_size)
        convolved_img = ndimage.convolve(sampled_img, kernel,
                                         mode=border_mode).clip(min=0.0)
        laplacian_img = block_reduce(convolved_img, block_size)

        if clean_error_image is None:
            if effective_gain is None or readnoise is None:
                msgs.error('effective_gain and readnoise must be input if error is not input')
            med5_img = ndimage.median_filter(clean_data, size=5,
                                             mode=border_mode).clip(min=1.e-5)
            error_image = (np.sqrt(effective_gain*med5_img + readnoise**2) /
                           effective_gain)
        else:
            error_image = clean_error_image

        snr_img = laplacian_img * utils.inverse(block_size * error_image)
        # this is used to remove extended structures (larger than ~5x5)
        snr_img -= ndimage.median_filter(snr_img, size=5, mode=border_mode)

        # used to remove compact bright objects
        med3_img = ndimage.median_filter(clean_data, size=3, mode=border_mode)
        med7_img = ndimage.median_filter(med3_img, size=7, mode=border_mode)
        finestruct_img = ((med3_img - med7_img) * utils.inverse(error_image)).clip(min=0.01)

        cr_mask1 = snr_img > cr_threshold
        # NOTE: to follow the paper exactly, this condition should be
        # "> contrast * block_size".  "lacos_im.cl" uses simply "> contrast"
        cr_mask2 = (snr_img * utils.inverse(finestruct_img)) > contrast
        cr_mask = cr_mask1 * cr_mask2
        if mask is not None:
            cr_mask = np.logical_and(cr_mask, np.invert(mask))

        # grow cosmic rays by one pixel and check in snr_img
        selem = np.ones((3, 3))
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = cr_mask1 * neigh_mask
        # now grow one more pixel and lower the detection threshold
        neigh_mask = ndimage.binary_dilation(cr_mask, selem)
        cr_mask = (snr_img > neighbor_threshold) * neigh_mask

        # previously unknown cosmic rays found in this iteration
        crmask_new = np.logical_and(~final_crmask, cr_mask)
        ncosmics = np.count_nonzero(crmask_new)

        final_crmask = np.logical_or(final_crmask, cr_mask)
        ncosmics_tot += ncosmics
        if verbose:
            msgs.info('Iteration {0}: Found {1} cosmic-ray pixels, '
                     'Total: {2}'.format(iteration + 1, ncosmics, ncosmics_tot))

    del clean_data
    gc.collect()

    return  final_crmask


def _clean_masked_pixels(data, mask, size=5, exclude_mask=None, verbose=True):
    """
    Clean masked pixels in an image.  Each masked pixel is replaced by
    the median of unmasked pixels in a 2D window of ``size`` centered on
    it.  If all pixels in the window are masked, then the window is
    increased in size until unmasked pixels are found.

    Pixels in ``exclude_mask`` are not cleaned, but they are excluded
    when calculating the local median.
    """

    if (size % 2) != 1:  # pragma: no cover
        msgs.error('size must be an odd integer')
    if data.shape != mask.shape:  # pragma: no cover
        msgs.error('mask must have the same shape as data')
    ny, nx = data.shape
    mask_coords = np.argwhere(mask)
    if exclude_mask is not None:
        if data.shape != exclude_mask.shape:  # pragma: no cover
            msgs.error('exclude_mask must have the same shape as data')
        maskall = np.logical_or(mask, exclude_mask)
    else:
        maskall = mask
    mask_idx = maskall.nonzero()
    data_nanmask = data.copy()
    data_nanmask[mask_idx] = np.nan

    nexpanded = 0
    for coord in mask_coords:
        y, x = coord
        median_val, expanded = _local_median(data_nanmask, x, y, nx, ny,
                                             size=size)
        data[y, x] = median_val
        if expanded:
            nexpanded += 1
    if nexpanded > 0 and verbose:
        msgs.info('    Found {0} {1}x{1} masked regions while '
                 'cleaning.'.format(nexpanded, size))
    del data_nanmask
    gc.collect()

    return data


def _local_median(data_nanmask, x, y, nx, ny, size=5, expanded=False):
    """Compute the local median in a 2D window, excluding NaN."""

    hy, hx = size // 2, size // 2
    x0, x1 = np.array([x - hx, x + hx + 1]).clip(0, nx)
    y0, y1 = np.array([y - hy, y + hy + 1]).clip(0, ny)
    region = data_nanmask[y0:y1, x0:x1].ravel()
    goodpixels = region[np.isfinite(region)]
    if len(goodpixels) > 0:
        median_val = np.median(goodpixels)
    else:
        newsize = size + 2  # keep size odd
        median_val, expanded = _local_median(data_nanmask, x, y, nx, ny,
                                             size=newsize, expanded=True)
    return median_val, expanded

def _process_block_inputs(data, block_size):
    data = np.asanyarray(data)
    block_size = np.atleast_1d(block_size)

    if np.any(block_size <= 0):
        msgs.error('block_size elements must be strictly positive')

    if data.ndim > 1 and len(block_size) == 1:
        block_size = np.repeat(block_size, data.ndim)

    if len(block_size) != data.ndim:
        msgs.error('block_size must be a scalar or have the same '
                         'length as the number of data dimensions')

    block_size_int = block_size.astype(int)
    if np.any(block_size_int != block_size):  # e.g., 2.0 is OK, 2.1 is not
        msgs.error('block_size elements must be integers')

    return data, block_size_int


def reshape_as_blocks(data, block_size):
    """
    Reshape a data array into blocks.

    This is useful to efficiently apply functions on block subsets of
    the data instead of using loops.  The reshaped array is a view of
    the input data array.

    .. versionadded:: 4.1

    Parameters
    ----------
    data : `~numpy.ndarray`
        The input data array.

    block_size : int or array_like (int)
        The integer block size along each axis.  If ``block_size`` is a
        scalar and ``data`` has more than one dimension, then
        ``block_size`` will be used for for every axis.  Each dimension
        of ``block_size`` must divide evenly into the corresponding
        dimension of ``data``.

    Returns
    -------
    output : `~numpy.ndarray`
        The reshaped array as a view of the input ``data`` array.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.nddata import reshape_as_blocks
    >>> data = np.arange(16).reshape(4, 4)
    >>> data
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> reshape_as_blocks(data, (2, 2))
    array([[[[ 0,  1],
             [ 4,  5]],
            [[ 2,  3],
             [ 6,  7]]],
           [[[ 8,  9],
             [12, 13]],
            [[10, 11],
             [14, 15]]]])
    """

    data, block_size = _process_block_inputs(data, block_size)

    if np.any(np.mod(data.shape, block_size) != 0):
        msgs.error('Each dimension of block_size must divide evenly '
                         'into the corresponding dimension of data')

    nblocks = np.array(data.shape) // block_size
    new_shape = tuple(k for ij in zip(nblocks, block_size) for k in ij)
    nblocks_idx = tuple(range(0, len(new_shape), 2))  # even indices
    block_idx = tuple(range(1, len(new_shape), 2))  # odd indices

    return data.reshape(new_shape).transpose(nblocks_idx + block_idx)


def block_reduce(data, block_size, func=np.sum):
    """
    Downsample a data array by applying a function to local blocks.

    If ``data`` is not perfectly divisible by ``block_size`` along a
    given axis then the data will be trimmed (from the end) along that
    axis.

    Parameters
    ----------
    data : array_like
        The data to be resampled.

    block_size : int or array_like (int)
        The integer block size along each axis.  If ``block_size`` is a
        scalar and ``data`` has more than one dimension, then
        ``block_size`` will be used for for every axis.

    func : callable, optional
        The method to use to downsample the data.  Must be a callable
        that takes in a `~numpy.ndarray` along with an ``axis`` keyword,
        which defines the axis or axes along which the function is
        applied.  The ``axis`` keyword must accept multiple axes as a
        tuple.  The default is `~numpy.sum`, which provides block
        summation (and conserves the data sum).

    Returns
    -------
    output : array_like
        The resampled data.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.nddata import block_reduce
    >>> data = np.arange(16).reshape(4, 4)
    >>> block_reduce(data, 2)  # doctest: +FLOAT_CMP
    array([[10, 18],
           [42, 50]])

    >>> block_reduce(data, 2, func=np.mean)  # doctest: +FLOAT_CMP
    array([[  2.5,   4.5],
           [ 10.5,  12.5]])
    """

    data, block_size = _process_block_inputs(data, block_size)
    nblocks = np.array(data.shape) // block_size
    size_init = nblocks * block_size  # evenly-divisible size

    # trim data if necessary
    for axis in range(data.ndim):
        if data.shape[axis] != size_init[axis]:
            data = data.swapaxes(0, axis)
            data = data[:size_init[axis]]
            data = data.swapaxes(0, axis)

    reshaped = reshape_as_blocks(data, block_size)
    axis = tuple(range(data.ndim, reshaped.ndim))

    return func(reshaped, axis=axis)


def block_replicate(data, block_size, conserve_sum=True):
    """
    Upsample a data array by block replication.

    Parameters
    ----------
    data : array_like
        The data to be block replicated.

    block_size : int or array_like (int)
        The integer block size along each axis.  If ``block_size`` is a
        scalar and ``data`` has more than one dimension, then
        ``block_size`` will be used for for every axis.

    conserve_sum : bool, optional
        If `True` (the default) then the sum of the output
        block-replicated data will equal the sum of the input ``data``.

    Returns
    -------
    output : array_like
        The block-replicated data.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.nddata import block_replicate
    >>> data = np.array([[0., 1.], [2., 3.]])
    >>> block_replicate(data, 2)  # doctest: +FLOAT_CMP
    array([[0.  , 0.  , 0.25, 0.25],
           [0.  , 0.  , 0.25, 0.25],
           [0.5 , 0.5 , 0.75, 0.75],
           [0.5 , 0.5 , 0.75, 0.75]])

    >>> block_replicate(data, 2, conserve_sum=False)  # doctest: +FLOAT_CMP
    array([[0., 0., 1., 1.],
           [0., 0., 1., 1.],
           [2., 2., 3., 3.],
           [2., 2., 3., 3.]])
    """

    data, block_size = _process_block_inputs(data, block_size)
    for i in range(data.ndim):
        data = np.repeat(data, block_size[i], axis=i)

    if conserve_sum:
        data = data / float(np.prod(block_size))

    return data
