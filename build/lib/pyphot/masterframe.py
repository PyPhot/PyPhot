"""
Implements the master frame base class.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst

"""
import gc, os
import numpy as np
from scipy.ndimage import gaussian_filter,median_filter

import multiprocessing
from multiprocessing import Process, Queue

from astropy import stats
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel

from pyphot import io,msgs, utils
from pyphot import procimg
from pyphot.photometry import BKG2D, mask_bright_star


def biasframe(biasfiles, camera, det, masterbias_name, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=3):

    images = []
    for ifile in biasfiles:
        detector_par, raw, header, exptime, rawdatasec_img, oscansec_img = camera.get_rawimage(ifile, det)
        array = procimg.trim_frame(raw, rawdatasec_img < 0.1)
        datasec_img = procimg.trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
        bias_image = utils.gain_correct(array, datasec_img, detector_par['gain'])
        images.append(bias_image)

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
    stack[np.isnan(stack)] = 0.0 # replace bad pixels with 0

    header['OLDTIME'] = (exptime, 'Original exposure time')
    header['EXPTIME'] = 0.0
    header['UNITS'] = ('e-', 'Data units')

    io.save_fits(masterbias_name, stack, header, 'MasterBias', mask=bpm.astype('int16'), overwrite=True)
    del images, bpm, bpms
    gc.collect()


def darkframe(darkfiles, camera, det, masterdark_name, masterbias=None, cenfunc='median', stdfunc='std',
              sigma=3, maxiters=3):

    if masterbias is not None:
        _, masterbiasimg, maskbiasimg = io.load_fits(masterbias)
    else:
        masterbiasimg = None

    images = []
    for ifile in darkfiles:
        detector_par, raw, header, exptime, rawdatasec_img, oscansec_img = camera.get_rawimage(ifile, det)
        array = procimg.trim_frame(raw, rawdatasec_img < 0.1)
        datasec_img = procimg.trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
        dark_image = utils.gain_correct(array, datasec_img, detector_par['gain'])
        if masterbiasimg is not None:
            dark_image -= masterbiasimg
        images.append(dark_image*utils.inverse(exptime))

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
    stack[np.isnan(stack)] = 0.0 # replace bad pixels with 0

    header['OLDTIME'] = (exptime, 'Original exposure time')
    header['EXPTIME'] = 1.0
    header['UNITS'] = ('e-/s', 'Data units')
    io.save_fits(masterdark_name, stack, header, 'MasterDark', mask=bpm.astype('int16'), overwrite=True)
    del images, bpm, bpms
    gc.collect()


def combineflat(flatfiles, maskfiles=None, camera=None, det=None, masterbias=None, masterdark=None, cenfunc='median',
                stdfunc='std', sigma=5, maxiters=3, window_size=(51,51), maskpixvar=None, minimum_vig=None,
                maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', conv='sex',
                sextractor_task='sex'):

    if masterbias is not None:
        _, masterbiasimg, maskbiasimg = io.load_fits(masterbias)
    else:
        masterbiasimg = None
    if masterdark is not None:
        _, masterdarkimg, maskdarkimg = io.load_fits(masterdark)
    else:
        masterdarkimg = None

    images = []
    masks = []
    masks_vig = []
    norm = []

    if camera is not None:
        msgs.info('Get bpm image for flatfielding')
        bpm = camera.bpm(flatfiles[0], det, shape=None, msbias=None).astype('bool')

    for ii, ifile in enumerate(flatfiles):

        if camera is not None:
            detector_par, raw, header, exptime, rawdatasec_img, oscansec_img = camera.get_rawimage(ifile, det)
            star_fits_file = ifile.replace('.fits', '_starmask.fits')
            array = procimg.trim_frame(raw, rawdatasec_img < 0.1)
            datasec_img = procimg.trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
            flat_image = utils.gain_correct(array, datasec_img, detector_par['gain'])
            #numamplifiers = detector_par['numamplifiers']
            #gain = detector_par['gain']
            if masterbiasimg is not None:
                flat_image -= masterbiasimg
            if masterdarkimg is not None:
                flat_image -= masterdarkimg*exptime
        else:
            # This is mainly used for supersky flat
            msgs.info('Reading detector proccessed flat image {:}'.format(ifile))
            star_fits_file = ifile.replace('_proc.fits', '_starmask.fits')
            header, flat_image, mask_image = io.load_fits(ifile)
            if maskfiles is not None:
                _, flag_image, _ = io.load_fits(maskfiles[ii])
                bpm = flag_image.astype('bool')
            else:
                bpm = mask_image.astype('bool')

        # Get Vignetting mask
        if minimum_vig is not None:
            # ToDo: This is specific for LRIS, but it should work for IMACS as well
            extra_bpm = flat_image < np.percentile(flat_image, 95) * (1-minimum_vig) #np.percentile(this_array, minimum_vig*100)
            # this_bpm |= extra_bpm
        else:
            extra_bpm = np.zeros_like(flat_image, dtype='bool')

        new_bpm = np.logical_or(bpm, extra_bpm) # a new bpm for statistics
        ## Mask bright stars
        if maskbrightstar:
            if os.path.exists(star_fits_file):
                msgs.info('Loading star mask from existing star mask image')
                _, star_image, _ = io.load_fits(star_fits_file)
                this_starmask = star_image>0
            else:
                this_starmask = mask_bright_star(flat_image, mask=new_bpm, brightstar_nsigma=brightstar_nsigma,
                                                 back_nsigma=sigma, back_maxiters=maxiters,
                                                 method=maskbrightstar_method, conv=conv,
                                                 task=sextractor_task)
        else:
            this_starmask = np.zeros_like(flat_image, dtype=bool)

        # Sigma_clipping statistics for the image
        this_mean, this_median, this_std = sigma_clipped_stats(flat_image, mask=np.logical_or(new_bpm, this_starmask),
                                                sigma=sigma, maxiters=maxiters, cenfunc=cenfunc, stdfunc=stdfunc)

        ## Mask hot pixels
        this_hotmask = flat_image > this_median + 5 * this_std
        flat_image[this_hotmask] = this_median

        ## Mask zero pixels
        this_zeromask = flat_image == 0.
        flat_image[this_zeromask] = this_median

        ## Append the data
        #images.append(array)
        #images.append(flat_image)
        images.append(flat_image * utils.inverse(this_median))
        masks.append(bpm | this_starmask | this_hotmask | this_zeromask)
        masks_vig.append(new_bpm | this_starmask | this_hotmask | this_zeromask)
        norm.append(this_median)

    msgs.info('Combing flat images')
    images = np.array(images)
    masks = np.array(masks)
    ## ToDo: Add weighted mean
    mean, median, stddev = stats.sigma_clipped_stats(images, masks, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=0)
    #mean_stat, median_stat, stddev_stat = stats.sigma_clipped_stats(images, masks_vig, sigma=sigma, maxiters=maxiters,
    #                                                 cenfunc=cenfunc, stdfunc=stdfunc, axis=0)
    #norm = np.nanmedian(median_stat)
    header['FNorm'] = (np.median(norm), 'Flat normalization')
    header['UNITS'] = ('e-/e-', 'Data units')
    if cenfunc == 'median':
        stack = median #* utils.inverse(norm)
    else:
        stack = mean #* utils.inverse(norm)

    bpm_nan = np.isnan(stack) | (stack==0.)
    stack[bpm_nan] = 0. # replace bad pixels with 0

    if maskpixvar is not None:
        # mask bad pixels based on pixelflat (i.e. pixel variance greater than XX% using maskbad)
        # Only used for pixel flat
        #illum = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
        illum, _ = BKG2D(stack, window_size, mask=bpm_nan, filter_size=(3,3),
                         sigclip=sigma, back_type='sextractor', back_rms_type='std',
                         back_maxiters=maxiters, sextractor_task=sextractor_task)

        bpm_pixvar = abs(1-stack*utils.inverse(illum))>maskpixvar
    else:
        bpm_pixvar = np.zeros_like(flat_image, dtype=bool)

    # bpm for the flat
    stack_bpm = bpm_pixvar | bpm_nan

    del images, masks, masks_vig
    gc.collect()

    return header, stack, stack_bpm

def illumflatframe(flatfiles, camera, det, masterillumflat_name, masterbias=None, masterdark=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51), minimum_vig=None,
                   maskbrightstar=False, brightstar_nsigma=5, maskbrightstar_method='sextractor',
                   conv='sex', sextractor_task='sex'):

    msgs.info('Building illuminating flat')
    header, stack, bpm = combineflat(flatfiles, camera=camera, det=det, masterbias=masterbias,
                                     masterdark=masterdark, cenfunc=cenfunc, stdfunc=stdfunc, sigma=sigma,
                                     maxiters=maxiters, window_size=window_size, maskpixvar=None, minimum_vig=minimum_vig,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma, conv=conv,
                                     maskbrightstar_method=maskbrightstar_method, sextractor_task=sextractor_task)

    ## ToDo: currently I am using sextractor for the illuminating flat. Need to get a better combineflat
    #from astropy.stats import SigmaClip
    #from photutils import Background2D, MedianBackground
    #sigma_clip = SigmaClip(sigma=sigma)
    #bkg = Background2D(stack.copy(), window_size, mask=bpm, filter_size=(3,3), sigma_clip=sigma_clip,
    #                   bkg_estimator=MedianBackground())
    #flat = bkg.background
    flat, _ = BKG2D(stack, window_size, mask=bpm, filter_size=(3, 3),
                     sigclip=sigma, back_type='sextractor', back_rms_type='std',
                     back_maxiters=maxiters, sextractor_task=sextractor_task)
    flat[bpm] = 0.
    # scipy gaussian_filter seems not ideal, could produce some problem at the edge.
    #flat = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
    io.save_fits(masterillumflat_name, flat, header, 'MasterIllumFlat', mask=bpm, overwrite=True)

def pixelflatframe(flatfiles, camera, det, masterpixflat_name, masterbias=None, masterdark=None, masterillumflat=None,
                   cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51), maskpixvar=0.1, minimum_vig=None,
                   maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', conv='sex',
                   sextractor_task='sex'):

    msgs.info('Building pixel flat')
    header, stack, bpm = combineflat(flatfiles, camera=camera, det=det, masterbias=masterbias, masterdark=masterdark, cenfunc=cenfunc,
                                     stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size, maskpixvar=maskpixvar,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma, minimum_vig=minimum_vig,
                                     maskbrightstar_method=maskbrightstar_method, conv=conv, sextractor_task=sextractor_task)

    if masterillumflat is None:
        masterillumflatimg = np.ones_like(stack)
    else:
        _, masterillumflatimg, maskillumflatimg = io.load_fits(masterillumflat)

    flat = stack * utils.inverse(masterillumflatimg)
    io.save_fits(masterpixflat_name, flat, header, 'MasterPixelFlat', mask=bpm, overwrite=True)

def superskyframe(superskyfiles, mastersupersky_name, maskfiles=None,
                  cenfunc='median', stdfunc='std', sigma=3, maxiters=3, window_size=(51,51),
                  maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor', conv='sex',
                  sextractor_task='sex'):

    msgs.info('Building super sky flat')
    header, stack, bpm = combineflat(superskyfiles, maskfiles=maskfiles, cenfunc=cenfunc, maskpixvar=None,
                                     stdfunc=stdfunc, sigma=sigma, maxiters=maxiters, window_size=window_size,
                                     maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                                     maskbrightstar_method=maskbrightstar_method, conv=conv,
                                     sextractor_task=sextractor_task)

    ## ToDo: currently I am using sextractor for the supersky. Need to get a better combineflat
    flat, _ = BKG2D(stack, window_size, mask=bpm, filter_size=(3, 3),
                     sigclip=sigma, back_type='sextractor', back_rms_type='std',
                     back_maxiters=maxiters, sextractor_task=sextractor_task)
    flat[bpm] = 0.
    #flat = gaussian_filter(stack, sigma=window_size[0], mode='mirror')
    #flat = median_filter(stack, size=window_size[0], mode='mirror')
    #io.save_fits(mastersupersky_name.replace('.fits','1.fits'), flat, header, 'MasterSuperSky', mask=bpm, overwrite=True)
    io.save_fits(mastersupersky_name, flat, header, 'MasterSuperSky', mask=bpm, overwrite=True)

def fringeframe(fringefiles, masterfringe_name, fringemaskfiles=None, mastersuperskyimg=None, cenfunc='median', stdfunc='std',
                sigma=3, maxiters=3, maskbrightstar=True, brightstar_nsigma=5, maskbrightstar_method='sextractor',
                conv='sex',sextractor_task='sex'):

    header, data0, mask0 = io.load_fits(fringefiles[0])
    nx, ny, nz = data0.shape[0], data0.shape[1], len(fringefiles)
    data3D = np.zeros((nx, ny, nz))
    mask3D = np.zeros((nx, ny, nz),dtype='bool')
    for iimg in range(nz):
        this_header, this_data, this_mask_image = io.load_fits(fringefiles[iimg])
        if mastersuperskyimg is not None:
            this_data = this_data * utils.inverse(mastersuperskyimg)
        if fringemaskfiles is not None:
            _, this_mask_image, _ = io.load_fits(fringemaskfiles[iimg])
        this_mask = this_mask_image.astype('bool')

        # Mask bright stars
        if maskbrightstar:
            star_fits_file = fringefiles[iimg].replace('_sci.fits', '_starmask.fits')
            if os.path.exists(star_fits_file):
                msgs.info('Loading star mask from existing star mask image')
                _, star_image, _ = io.load_fits(star_fits_file)
                starmask = star_image>0
            else:
                # from photutils import detect_sources
                # mean, median, stddev = stats.sigma_clipped_stats(this_data, mask=this_mask, sigma=sigma, maxiters=maxiters,
                #                                                 cenfunc=cenfunc, stdfunc=stdfunc)
                # segm = detect_sources(this_data, brightstar_nsigma*stddev, npixels=5)
                # starmask = segm.data.astype('bool')
                starmask = mask_bright_star(this_data, mask=this_mask, brightstar_nsigma=brightstar_nsigma,
                                            back_nsigma=sigma, back_maxiters=maxiters,
                                             method=maskbrightstar_method, conv=conv, task=sextractor_task)
        else:
            starmask = np.zeros_like(this_data, dtype=bool)

        this_mask = np.logical_or(this_mask, starmask)

        data3D[:, :, iimg] = this_data * utils.inverse(this_header['EXPTIME'])
        mask3D[:, :, iimg] = this_mask.astype('bool')

    ## constructing the master fringe frame
    mean, median, stddev = stats.sigma_clipped_stats(data3D, mask=mask3D, sigma=sigma, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc=stdfunc, axis=2)

    if cenfunc == 'median':
        stack = median
    else:
        stack = mean

    bpm = (np.isnan(stack)).astype('int32')
    stack[np.isnan(stack)] = 0.
    header['OLDTIME'] = (header['EXPTIME'], 'Original exposure time')
    header['EXPTIME'] = 1.0
    # save master fringe frame
    io.save_fits(masterfringe_name, stack, header, 'MasterFringe', mask=bpm, overwrite=True)
    del data3D, mask3D
    gc.collect()

class MasterFrames():
    """
    Build master frames
    """

    def __init__(self, par, camera, det, master_key, raw_shape, reuse_masters=True):

        self.par = par
        self.camera = camera
        self.det = det
        self.master_key = master_key
        self.raw_shape = raw_shape
        self.master_dir = self.par['calibrations']['master_dir']
        self.use_bias = self.par['scienceframe']['process']['use_biasimage']
        self.use_dark = self.par['scienceframe']['process']['use_darkimage']
        self.use_illum = self.par['scienceframe']['process']['use_illumflat']
        self.use_pixel = self.par['scienceframe']['process']['use_pixelflat']
        self.reuse_masters = reuse_masters

        if self.use_bias:
            self.masterbias_name = os.path.join(self.master_dir, 'MasterBias_{:}'.format(self.master_key))
        else:
            self.masterbias_name = None
        if self.use_dark:
            self.masterdark_name = os.path.join(self.master_dir, 'MasterDark_{:}'.format(self.master_key))
        else:
            self.masterdark_name = None
        if self.use_illum:
            self.masterillumflat_name = os.path.join(self.master_dir, 'MasterIllumFlat_{:}'.format(self.master_key))
        else:
            self.masterillumflat_name = None
        if self.use_pixel:
            self.masterpixflat_name = os.path.join(self.master_dir, 'MasterPixelFlat_{:}'.format(self.master_key))
        else:
            self.masterpixflat_name = None

    def build(self, biasfiles=None, darkfiles=None, illumflatfiles=None, pixflatfiles=None):

        ## Load parameters from Calibration Par
        # Bias
        b_cenfunc = self.par['calibrations']['biasframe']['process']['comb_cenfunc']
        b_stdfunc = self.par['calibrations']['biasframe']['process']['comb_stdfunc']
        b_sigrej = self.par['calibrations']['biasframe']['process']['comb_sigrej']
        b_maxiter = self.par['calibrations']['biasframe']['process']['comb_maxiter']
        # Dark
        d_cenfunc = self.par['calibrations']['darkframe']['process']['comb_cenfunc']
        d_stdfunc = self.par['calibrations']['darkframe']['process']['comb_stdfunc']
        d_sigrej = self.par['calibrations']['darkframe']['process']['comb_sigrej']
        d_maxiter = self.par['calibrations']['darkframe']['process']['comb_maxiter']
        # IllumFlat
        i_cenfunc = self.par['calibrations']['illumflatframe']['process']['comb_cenfunc']
        i_stdfunc = self.par['calibrations']['illumflatframe']['process']['comb_stdfunc']
        i_sigrej = self.par['calibrations']['illumflatframe']['process']['comb_sigrej']
        i_maxiter = self.par['calibrations']['illumflatframe']['process']['comb_maxiter']
        i_window = self.par['calibrations']['illumflatframe']['process']['window_size']
        i_maskbrigtstar = self.par['calibrations']['illumflatframe']['process']['mask_brightstar']
        i_brightstar_nsigma = self.par['calibrations']['illumflatframe']['process']['brightstar_nsigma']
        i_maskbrightstar_method = self.par['calibrations']['illumflatframe']['process']['brightstar_method']
        i_conv = self.par['calibrations']['illumflatframe']['process']['conv']
        # PixelFlag
        p_cenfunc = self.par['calibrations']['pixelflatframe']['process']['comb_cenfunc']
        p_stdfunc = self.par['calibrations']['pixelflatframe']['process']['comb_stdfunc']
        p_sigrej = self.par['calibrations']['pixelflatframe']['process']['comb_sigrej']
        p_maxiter = self.par['calibrations']['pixelflatframe']['process']['comb_maxiter']
        p_window = self.par['calibrations']['pixelflatframe']['process']['window_size']
        p_maskbrigtstar = self.par['calibrations']['pixelflatframe']['process']['mask_brightstar']
        p_brightstar_nsigma = self.par['calibrations']['pixelflatframe']['process']['brightstar_nsigma']
        p_maskbrightstar_method = self.par['calibrations']['pixelflatframe']['process']['brightstar_method']
        p_conv = self.par['calibrations']['pixelflatframe']['process']['conv']
        maskpixvar = self.par['calibrations']['pixelflatframe']['process']['maskpixvar']
        # all
        minimum_vig = self.par['scienceframe']['process']['minimum_vig']
        sextractor_task = self.par['rdx']['sextractor']

        # Build Bias
        if self.use_bias:
            if os.path.exists(self.masterbias_name) and self.reuse_masters:
                msgs.info('Using existing master file {:}'.format(self.masterbias_name))
            else:
                msgs.info('Building master Bias Frame {:}'.format(self.masterbias_name))
                biasframe(biasfiles, self.camera, self.det, self.masterbias_name,
                          cenfunc=b_cenfunc, stdfunc=b_stdfunc, sigma=b_sigrej, maxiters=b_maxiter)

        # Build Dark
        if self.use_dark:
            if os.path.exists(self.masterdark_name) and self.reuse_masters:
                msgs.info('Using existing master Dark frame {:}'.format(self.masterdark_name))
            else:
                msgs.info('Building master file {:}'.format(self.masterdark_name))
                darkframe(darkfiles, self.camera, self.det, self.masterdark_name,
                          masterbias=self.masterbias_name,
                          cenfunc=d_cenfunc, stdfunc=d_stdfunc, sigma=d_sigrej, maxiters=d_maxiter)

        # Build Illumination Flat
        if self.use_illum:
            if os.path.exists(self.masterillumflat_name) and self.reuse_masters:
                msgs.info('Using existing master file {:}'.format(self.masterillumflat_name))
            else:
                msgs.info('Building master IllumFlat frame {:}'.format(self.masterillumflat_name))
                illumflatframe(illumflatfiles, self.camera, self.det, self.masterillumflat_name,
                               masterbias=self.masterbias_name, masterdark=self.masterdark_name,
                               cenfunc=i_cenfunc,stdfunc=i_stdfunc,sigma=i_sigrej,maxiters=i_maxiter,
                               window_size=i_window,minimum_vig=minimum_vig,
                               maskbrightstar=i_maskbrigtstar,brightstar_nsigma=i_brightstar_nsigma,
                               maskbrightstar_method=i_maskbrightstar_method,
                               conv=i_conv, sextractor_task=sextractor_task)

        # Build Pixel Flat
        if self.use_pixel:
            if os.path.exists(self.masterpixflat_name) and self.reuse_masters:
                msgs.info('Using existing master file {:}'.format(self.masterpixflat_name))
            else:
                msgs.info('Building master PixelFlat frame {:}'.format(self.masterpixflat_name))
                pixelflatframe(pixflatfiles, self.camera, self.det, self.masterpixflat_name,
                               masterbias=self.masterbias_name, masterdark=self.masterdark_name,
                               masterillumflat=self.masterillumflat_name,
                               cenfunc=p_cenfunc,stdfunc=p_stdfunc,sigma=p_sigrej,maxiters=p_maxiter,
                               window_size=p_window,minimum_vig=minimum_vig,maskpixvar=maskpixvar,
                               maskbrightstar=p_maskbrigtstar,brightstar_nsigma=p_brightstar_nsigma,
                               maskbrightstar_method=p_maskbrightstar_method,
                               conv=p_conv, sextractor_task=sextractor_task)

    def load(self):

        if self.use_bias:
            if os.path.exists(self.masterbias_name):
                _, masterbiasimg, maskbiasimg = io.load_fits(self.masterbias_name)
            else:
                msgs.error('Please build master files first!')
        else:
            masterbiasimg = np.zeros(self.raw_shape)
            maskbiasimg = np.zeros(self.raw_shape, dtype='int')

        if self.use_dark:
            if os.path.exists(self.masterdark_name):
                _, masterdarkimg, maskdarkimg = io.load_fits(self.masterdark_name)
            else:
                msgs.error('Please build master files first!')
        else:
            masterdarkimg = np.zeros(self.raw_shape)
            maskdarkimg = np.zeros(self.raw_shape, dtype='int')

        if self.use_illum:
            if os.path.exists(self.masterillumflat_name):
                headerillum, masterillumflatimg, maskillumflatimg = io.load_fits(self.masterillumflat_name)
                norm_illum = headerillum['FNorm']
            else:
                msgs.error('Please build master files first!')
        else:
            masterillumflatimg = np.ones(self.raw_shape)
            maskillumflatimg = np.zeros(self.raw_shape, dtype='int')
            norm_illum = 1.

        if self.use_pixel:
            if os.path.exists(self.masterpixflat_name):
                headerpixel, masterpixflatimg, maskpixflatimg = io.load_fits(self.masterpixflat_name)
                norm_pixel= headerpixel['FNorm']
            else:
                msgs.error('Please build master files first!')
        else:
            masterpixflatimg = np.ones(self.raw_shape)
            maskpixflatimg = np.zeros(self.raw_shape, dtype='int')
            norm_pixel = 1.

        if self.par['scienceframe']['process']['mask_proc']:
            bpm_sum = maskbiasimg + maskdarkimg + maskillumflatimg + maskpixflatimg
            bpm_proc = bpm_sum.astype('bool')
        else:
            bpm_proc = np.zeros(self.raw_shape, dtype='bool')

        return masterbiasimg, masterdarkimg, masterillumflatimg, masterpixflatimg, bpm_proc, norm_illum, norm_pixel

## Parallel functions for calling MasterFrames
def build_masters(detectors, master_keys, raw_shapes, camera=None, par=None, biasfiles=None,
                  darkfiles=None, illumflatfiles=None, pixflatfiles=None, reuse_masters=True):
    '''
    Calling MasterFrames in parallel.
    Parameters
    ----------
    detectors
    master_keys
    raw_shapes
    camera
    par
    biasfiles
    darkfiles
    illumflatfiles
    pixflatfiles
    reuse_masters

    Returns
    -------

    '''

    n_process = par['rdx']['n_process']
    n_det = len(detectors)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process > n_det:
        n_process = n_det

    if n_process == 1:
        for ii in range(n_det):
            Master = MasterFrames(par, camera, detectors[ii], master_keys[ii], raw_shapes[ii],
                                  reuse_masters=reuse_masters)
            # Build MasterFrames
            Master.build(biasfiles=biasfiles, darkfiles=darkfiles,
                         illumflatfiles=illumflatfiles, pixflatfiles=pixflatfiles)
    else:
        msgs.info('Build master files with n_process={:}'.format(n_process))
        work_queue = Queue()
        processes = []
        for ii in range(n_det):
            work_queue.put((detectors[ii], master_keys[ii], raw_shapes[ii]))
        # creating processes
        for w in range(n_process):
            p = Process(target=_build_masters_worker, args=(work_queue,), kwargs={
                'camera': camera, 'par': par, 'biasfiles': biasfiles, 'darkfiles': darkfiles,
                'illumflatfiles': illumflatfiles, 'pixflatfiles': pixflatfiles,
                'reuse_masters': reuse_masters})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

def _build_masters(idet, master_key, raw_shape, camera=None, par=None, biasfiles=None, darkfiles=None,
                   illumflatfiles=None, pixflatfiles=None, reuse_masters=True):
    # Initialize
    Master = MasterFrames(par, camera, idet, master_key, raw_shape, reuse_masters=reuse_masters)
    # Build MasterFrames
    Master.build(biasfiles=biasfiles, darkfiles=darkfiles,
                 illumflatfiles=illumflatfiles, pixflatfiles=pixflatfiles)

def _build_masters_worker(work_queue, camera=None, par=None, biasfiles=None, darkfiles=None,
                          illumflatfiles=None, pixflatfiles=None, reuse_masters=True):
    while not work_queue.empty():
        idet, master_key, raw_shape = work_queue.get()
        _build_masters(idet, master_key, raw_shape, camera=camera, par=par, biasfiles=biasfiles,
                      darkfiles=darkfiles, illumflatfiles=illumflatfiles, pixflatfiles=pixflatfiles,
                      reuse_masters=reuse_masters)

def rescale_flat(camera, par, detectors, master_keys, raw_shapes):
    '''
    The Masterflats were build on the detector basis (i.e., normlized by different values for different detectors).
    This function aims to rescale the normalization to ensure all detectors are normalized by the same number.
    Parameters
    ----------
    camera
    par
    detectors
    master_keys
    raw_shapes
    reuse_masters

    Returns
    -------

    '''
    norm_illum_list = []
    norm_pixel_list = []
    for ii, idet in enumerate(detectors):
        Master = MasterFrames(par, camera, idet, master_keys[ii], raw_shapes[ii])
        # Load master frames
        _, _, masterillumflatimg, masterpixflatimg, _, norm_illum, norm_pixel = Master.load()
        norm_illum_list.append(norm_illum)
        norm_pixel_list.append(norm_pixel)

    median_illum_scale = np.median(norm_illum_list)
    median_pixel_scale = np.median(norm_pixel_list)

    for ii, idet in enumerate(detectors):
        Master = MasterFrames(par, camera, idet, master_keys[ii], raw_shapes[ii])
        if Master.masterpixflat_name is not None:
            headerpixel, masterpixflatimg, maskpixflatimg = io.load_fits(Master.masterpixflat_name)
            if 'FNormS' not in headerpixel:
                headerpixel['FNormS'] = (median_pixel_scale, 'Re-scaled flat normalization')
                msgs.info('Rescale MasterPixelFlat with FNorm={:}'.format(median_pixel_scale))
                io.save_fits(Master.masterpixflat_name, masterpixflatimg * norm_pixel_list[ii] / median_pixel_scale,
                             headerpixel, 'MasterPixelFlat', mask=maskpixflatimg, overwrite=True)
        if Master.masterillumflat_name is not None:
            headerillum, masterillumflatimg, maskillumflatimg = io.load_fits(Master.masterillumflat_name)
            if 'FNormS' not in headerillum:
                headerillum['FNormS'] = (median_pixel_scale, 'Re-scaled flat normalization')
                msgs.info('Rescale MasterIllumFlat with FNorm={:}'.format(median_pixel_scale))
                io.save_fits(Master.masterillumflat_name, masterillumflatimg * norm_illum_list[ii] / median_illum_scale,
                             headerillum, 'MasterIllumFlat', mask=maskillumflatimg, overwrite=True)
