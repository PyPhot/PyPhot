"""
Implements the master frame base class.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import gc
import numpy as np
from scipy.ndimage import gaussian_filter,median_filter

from astropy import stats
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from photutils import Background2D, MedianBackground
from photutils import detect_sources

from pyphot import io,msgs
from pyphot import postproc


def biasframe(biasfiles, camera, det, masterbias_name,cenfunc='median', stdfunc='std',
              sigma=3, maxiters=3):

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
    del images, bpm, bpms
    gc.collect()


def darkframe(darkfiles, camera, det, masterdark_name, masterbiasimg=None, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=3):

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
    del images, bpm, bpms
    gc.collect()


def combineflat(flatfiles, maskfiles=None, camera=None, det=None, masterbiasimg=None, masterdarkimg=None, cenfunc='median',
                stdfunc='std', sigma=5, maxiters=3, window_size=(51,51), maskpixvar=None,
                maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', sextractor_task='sex'):

    images = []
    masks = []

    for ii, ifile in enumerate(flatfiles):
        if camera is not None:
            msgs.info('Reading raw flat image {:}'.format(ifile))
            detector_par, array, header, exptime,_,_ = camera.get_rawimage(ifile, det)
            if masterbiasimg is not None:
                array -= masterbiasimg
            if masterdarkimg is not None:
                array -= masterdarkimg*exptime
            bpm = camera.bpm(ifile, det, shape=None, msbias=None).astype('bool')
        else:
            msgs.info('Reading ccd proccessed flat image {:}'.format(ifile))
            header, array, mask_image = io.load_fits(ifile)
            if maskfiles is not None:
                _, flag_image, _ = io.load_fits(maskfiles[ii])
                bpm = flag_image.astype('bool')
            else:
                bpm = mask_image.astype('bool')

        ## Mask bright stars
        if maskbrightstar:
            starmask = postproc.mask_bright_star(array, mask=bpm, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigma,
                                                 back_maxiters=maxiters, method=maskbrightstar_method, task=sextractor_task)
        else:
            starmask = np.zeros_like(array, dtype=bool)

        # Sigma_clipping statistics for the image
        mean, median, std = sigma_clipped_stats(array, mask=np.logical_or(bpm, starmask) , sigma=sigma,
                                                maxiters=maxiters, cenfunc=cenfunc, stdfunc=stdfunc)

        ## Mask hot pixels
        hotmask = array > median+5*std
        array[hotmask] = median

        ## Mask zero pixels
        zeromask = array == 0.
        array[zeromask] = median

        images.append(array/median)
        masks.append(bpm | starmask | hotmask | zeromask)

    msgs.info('Combing flat images')
    images = np.array(images)
    masks = np.array(masks)
    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, masks, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)

    if cenfunc == 'median':
        stack = median / np.nanmedian(median)
    else:
        stack = mean / np.nanmedian(median)

    stack[np.isnan(stack)] = 1.0 # replace bad pixels with 1, so not flat fielding that pixel

    if maskpixvar is not None:
        # mask bad pixels based on pixelflat (i.e. pixel variance greater than XX% using maskbad)
        # Only used for pixel flat
        #illum = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
        sigma_clip = SigmaClip(sigma=sigma)
        bkg = Background2D(stack.copy(), window_size, mask=np.isnan(stack), filter_size=(3,3), sigma_clip=sigma_clip,
                           bkg_estimator=MedianBackground())
        illum = bkg.background
        bpm_pixvar = abs(1-stack/illum)>maskpixvar
    else:
        bpm_pixvar = np.zeros_like(array, dtype=bool)

    # bpm for the flat
    stack_bpm = bpm_pixvar | (np.isnan(stack))

    del images, masks
    gc.collect()

    return header, stack, stack_bpm

def illumflatframe(flatfiles, camera, det, masterillumflat_name, masterbiasimg=None, masterdarkimg=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51),
                   maskbrightstar=False, brightstar_nsigma=5, maskbrightstar_method='sextractor', sextractor_task='sex'):

    msgs.info('Building illuminating flat')
    header, stack, bpm = combineflat(flatfiles, camera=camera, det=det, masterbiasimg=masterbiasimg,
                                     masterdarkimg=masterdarkimg, cenfunc=cenfunc, stdfunc=stdfunc, sigma=sigma,
                                     maxiters=maxiters, window_size=window_size, maskpixvar=None,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                                     maskbrightstar_method=maskbrightstar_method, sextractor_task=sextractor_task)

    ## ToDo: currently I am using photoutils for the illuminating flat. Need to get a better combineflat
    sigma_clip = SigmaClip(sigma=sigma)
    bkg = Background2D(stack.copy(), window_size, mask=bpm, filter_size=(3,3), sigma_clip=sigma_clip,
                       bkg_estimator=MedianBackground())
    flat = bkg.background
    # scipy gaussian_filter seems not ideal, could produce some problem at the edge.
    #flat = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
    io.save_fits(masterillumflat_name, flat, header, 'MasterIllumFlat', mask=bpm, overwrite=True)

def pixelflatframe(flatfiles, camera, det, masterpixflat_name, masterbiasimg=None, masterdarkimg=None, masterillumflatimg=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51), maskpixvar=0.1,
                   maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', sextractor_task='sex'):

    msgs.info('Building pixel flat')
    header, stack, bpm = combineflat(flatfiles, camera=camera, det=det, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg, cenfunc=cenfunc,
                                     stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size, maskpixvar=maskpixvar,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                                     maskbrightstar_method=maskbrightstar_method, sextractor_task=sextractor_task)

    if masterillumflatimg is None:
        masterillumflatimg = np.ones_like(stack)

    flat = stack / masterillumflatimg
    io.save_fits(masterpixflat_name, flat, header, 'MasterPixelFlat', mask=bpm, overwrite=True)

def superskyframe(superskyfiles, mastersupersky_name, maskfiles=None,
                  cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51),
                  maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', sextractor_task='sex'):

    msgs.info('Building super sky flat')
    header, stack, bpm = combineflat(superskyfiles, maskfiles=maskfiles, cenfunc=cenfunc, maskpixvar=None,
                                     stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                                     maskbrightstar_method=maskbrightstar_method, sextractor_task=sextractor_task)

    ## ToDo: currently I am using photoutils for the supersky. Need to get a better combineflat
    sigma_clip = SigmaClip(sigma=sigma)
    bkg = Background2D(stack.copy(), window_size, mask=bpm, filter_size=(3,3), sigma_clip=sigma_clip,
                       bkg_estimator=MedianBackground())
    flat = bkg.background
    #flat = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
    #flat = median_filter(stack, size=window_size[0], mode='mirror')
    #io.save_fits(mastersupersky_name.replace('.fits','1.fits'), flat, header, 'MasterSuperSky', mask=bpm, overwrite=True)
    io.save_fits(mastersupersky_name, flat, header, 'MasterSuperSky', mask=bpm, overwrite=True)

def fringeframe(fringefiles, masterfringe_name, fringemaskfiles=None, mastersuperskyimg=None, cenfunc='median', stdfunc='std',
                sigma=3, maxiters=3, maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor',
                sextractor_task='sex'):

    header, data0, mask0 = io.load_fits(fringefiles[0])
    nx, ny, nz = data0.shape[0], data0.shape[1], len(fringefiles)
    data3D = np.zeros((nx, ny, nz))
    mask3D = np.zeros((nx, ny, nz),dtype='bool')
    for iimg in range(nz):
        this_header, this_data, this_mask_image = io.load_fits(fringefiles[iimg])
        if mastersuperskyimg is not None:
            this_data = this_data / mastersuperskyimg
        if fringemaskfiles is not None:
            _, this_mask_image, _ = io.load_fits(fringemaskfiles[iimg])
        this_mask = this_mask_image.astype('bool')

        # Mask very bright stars
        if maskbrightstar:
            #from photutils import detect_sources
            #mean, median, stddev = stats.sigma_clipped_stats(this_data, mask=this_mask, sigma=sigma, maxiters=maxiters,
            #                                                 cenfunc=cenfunc, stdfunc=stdfunc)
            #segm = detect_sources(this_data, brightstar_nsigma*stddev, npixels=5)
            #starmask = segm.data.astype('bool')
            starmask = postproc.mask_bright_star(this_data, mask=this_mask, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigma,
                                                 back_maxiters=maxiters, method=maskbrightstar_method, task=sextractor_task)
            this_mask = np.logical_or(this_mask, starmask)

        data3D[:, :, iimg] = this_data / this_header['EXPTIME']
        mask3D[:, :, iimg] = this_mask.astype('bool')

    ## constructing the master fringe frame
    mean, median, stddev = stats.sigma_clipped_stats(data3D, mask=mask3D, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=2)

    if cenfunc == 'median':
        stack = median
    else:
        stack = mean

    bpm = (np.isnan(stack)).astype('int16')
    stack[np.isnan(stack)] = 0.
    header['OLDTIME'] = (header['EXPTIME'], 'Original exposure time')
    header['EXPTIME'] = 1.0
    # save master fringe frame
    io.save_fits(masterfringe_name, stack, header, 'MasterFringe', mask=bpm, overwrite=True)
    del data3D, mask3D
    gc.collect()
