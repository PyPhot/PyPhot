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
    bpm = camera.bpm(biasfiles[0], det, shape=None, msbias=None).astype('bool')
    bpms = np.repeat(bpm[np.newaxis, :, :], images.shape[0], axis=0)

    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, bpms, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)
    ## ToDo: identify bad pixels from masterbias

    if cenfunc == 'median':
        stack = median
    else:
        stack = mean

    header['OLDTIME'] = (exptime, 'Original exposure time')
    header['EXPTIME'] = 0.0

    io.save_fits(masterbias_name, stack, header, 'MasterBias', mask=bpm.astype('int16'), overwrite=True)


def darkframe(darkfiles, camera, det, masterdark_name, masterbiasimg=None, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=5):

    images = []
    for ifile in darkfiles:
        detector_par, array, header, exptime,_,_ = camera.get_rawimage(ifile, det)
        if masterbiasimg is not None:
            array -= masterbiasimg
        images.append(array/exptime)

    images = np.array(images)

    bpm = camera.bpm(darkfiles[0], det, shape=None, msbias=None).astype('bool')
    bpms = np.repeat(bpm[np.newaxis, :, :], images.shape[0], axis=0)

    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, bpms, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)
    ## ToDo: identify bad pixels from masterdark

    if cenfunc == 'median':
        stack = median
    else:
        stack = mean

    header['OLDTIME'] = (exptime, 'Original exposure time')
    header['EXPTIME'] = 1.0
    io.save_fits(masterdark_name, stack, header, 'MasterDark', mask=bpm.astype('int16'), overwrite=True)


def combineflat(flatfiles, camera, det, masterbiasimg=None, masterdarkimg=None, cenfunc='median',
                   stdfunc='std', sigma=5, maxiters=5, window_size=50, maskillum=0.3):

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

    ## mask bad pixels (i.e. pixel variance greater than XX% using maskbad)
    illum = gaussian_filter(stack, sigma=window_size)
    bpm_illum = abs(1-stack/illum)>maskillum

    ## add bpm and masks into the flat
    bpm = camera.bpm(flatfiles[0], det, shape=None, msbias=None).astype('bool')
    stack_bpm = (bpm_illum | bpm | (np.isnan(stack))).astype('int16')

    return header, stack, stack_bpm

def illumflatframe(flatfiles, camera, det, masterillumflat_name, masterbiasimg=None, masterdarkimg=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=5, window_size=51, maskillum=0.3):

    header, stack, bpm = combineflat(flatfiles, camera, det, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg, cenfunc=cenfunc,
                         stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size, maskillum=maskillum)

    flat = gaussian_filter(stack, sigma=window_size)
    io.save_fits(masterillumflat_name, flat, header, 'MasterIllumFlat', mask=bpm, overwrite=True)

def pixelflatframe(flatfiles, camera, det, masterpixflat_name, masterbiasimg=None, masterdarkimg=None, masterillumflatimg=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=5, window_size=51, maskillum=0.3):

    header, stack, bpm = combineflat(flatfiles, camera, det, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg, cenfunc=cenfunc,
                         stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size, maskillum=maskillum)

    if masterillumflatimg is None:
        masterillumflatimg = np.ones_like(stack)

    flat = stack / masterillumflatimg

    io.save_fits(masterpixflat_name, flat, header, 'MasterPixelFlat', mask=bpm, overwrite=True)

def fringeframe(fringefiles, masterfringe_name, mask=None, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=5):

    data0, header = io.load_fits(fringefiles[0])
    nx, ny, nz = data0.shape[0], data0.shape[1], len(fringefiles)
    data3D = np.zeros((nx, ny, nz))
    for iimg in range(nz):
        this_data, this_header = io.load_fits(fringefiles[iimg])
        data3D[:, :, iimg] = this_data / this_header['EXPTIME']

    mean, median, stddev = stats.sigma_clipped_stats(data3D, mask=mask, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=2)

    if cenfunc == 'median':
        stack = median
    else:
        stack = mean


    bpm = (np.isnan(stack)).astype('int16')

    header['OLDTIME'] = (header['EXPTIME'], 'Original exposure time')
    header['EXPTIME'] = 1.0

    io.save_fits(masterfringe_name, stack, header, 'MasterFringe', mask=bpm, overwrite=True)
