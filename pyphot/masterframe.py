"""
Implements the master frame base class.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import numpy as np
from scipy.ndimage import gaussian_filter

from astropy import stats
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder

from pyphot import io

def biasframe(biasfiles, camera, det, masterbias_name,cenfunc='median', stdfunc='std',
              sigma=3, maxiters=5):

    images = []
    for ifile in biasfiles:
        detector_par, array, header, exptime,_,_ = camera.get_rawimage(ifile, det)
        images.append(array)

    images = np.array(images)
    gpms = np.ones_like(images, dtype=bool)

    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, np.invert(gpms), sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)

    if cenfunc == 'medina':
        stack = median
    else:
        stack = mean

    header['OLDTIME'] = (header['EXPTIME'], 'Original exposure time')
    header['EXPTIME'] = 0.0
    io.save_fits(masterbias_name, stack, header, 'MasterBias', overwrite=True)


def darkframe(darkfiles, camera, det, masterdark_name, masterbiasimg=None, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=5):

    images = []
    for ifile in darkfiles:
        detector_par, array, header, exptime,_,_ = camera.get_rawimage(ifile, det)
        if masterbiasimg is not None:
            array -= masterbiasimg
        images.append(array/exptime)

    images = np.array(images)
    gpms = np.ones_like(images, dtype=bool)

    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, np.invert(gpms), sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)

    if cenfunc == 'medina':
        stack = median
    else:
        stack = mean

    header['OLDTIME'] = (header['EXPTIME'], 'Original exposure time')
    header['EXPTIME'] = 1.0
    io.save_fits(masterdark_name, stack, header, 'MasterDark', overwrite=True)

def combineflat(flatfiles, camera, det, masterbiasimg=None, masterdarkimg=None, cenfunc='median',
                   stdfunc='std', sigma=5, maxiters=5):

    images = []
    masks = []
    for ifile in flatfiles:
        detector_par, array, header, exptime,_,_ = camera.get_rawimage(ifile, det)
        if masterbiasimg is not None:
            array -= masterbiasimg
        if masterdarkimg is not None:
            array -= masterdarkimg*exptime
        mean, median, std = sigma_clipped_stats(array, sigma=sigma)
        ## ToDo: mask bright stars with photoutils
        try:
            starmask = np.zeros_like(array, dtype=bool)
            daofind = DAOStarFinder(fwhm=5.0, threshold=20*std)
            sources = daofind(array - median)
            for iobj in range(len(sources)):
                xx,yy = sources['xcentroid'][iobj].astype(int), sources['ycentroid'][iobj].astype(int)
                # ToDo: this is a hack, using the shape from daofind to mask, seems x and y are reversed
                array[np.fmax(yy-10,0):np.fmin(yy+10,array.shape[1]), np.fmax(xx-10,0):np.fmin(xx+10,array.shape[0])] = median
                starmask[np.fmax(yy-10,0):np.fmin(yy+10,array.shape[1]), np.fmax(xx-10,0):np.fmin(xx+10,array.shape[0])] = 1
        except:
            starmask = np.zeros_like(array, dtype=bool)
        # further mask hot pixels
        hotmask = array > median+5*std
        array[hotmask] = median
        # further mask zero pixels
        zeromask = array == 0.
        array[zeromask] = median

        images.append(array/median)
        masks.append(starmask|hotmask|zeromask)

    images = np.array(images)
    masks = np.array(masks)
    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, masks, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)

    if cenfunc == 'median':
        stack = median / np.nanmedian(median)
    else:
        stack = mean / np.nanmedian(median)

    stack[np.isnan(stack)] = np.nanmedian(stack)

    return stack, header

def pixelflatframe(flatfiles, camera, det, masterpixflat_name, masterbiasimg=None, masterdarkimg=None, cenfunc='median',
                   stdfunc='std', sigma=3, maxiters=5):

    stack, header = combineflat(flatfiles, camera, det, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg, cenfunc=cenfunc,
                   stdfunc=stdfunc, sigma=sigma, maxiters=maxiters)

    io.save_fits(masterpixflat_name, stack, header, 'MasterPixelFlat', overwrite=True)

def illumflatframe(flatfiles, camera, det, masterillumflat_name, masterbiasimg=None, masterdarkimg=None, masterpixflatimg=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=5, window_size=50):

    stack, header = combineflat(flatfiles, camera, det, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg, cenfunc=cenfunc,
                   stdfunc=stdfunc, sigma=sigma, maxiters=maxiters)

    if masterpixflatimg is None:
        masterpixflatimg = np.ones_like(stack)

    flat = gaussian_filter(stack/masterpixflatimg, sigma=window_size)

    io.save_fits(masterillumflat_name, flat, header, 'MasterIllumFlat', overwrite=True)
