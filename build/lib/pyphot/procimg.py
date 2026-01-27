""" Module for image processing core methods

"""
import gc
import os
import numpy as np

import multiprocessing
from multiprocessing import Process, Queue

from astropy import stats

from pyphot import msgs
from pyphot import utils
from pyphot import io
from pyphot import masterframe, postproc
from pyphot.lacosmic import lacosmic
from pyphot.satdet import satdet
from pyphot.photometry import BKG2D, mask_bright_star

def detproc(scifiles, camera, det, n_process=4, science_path=None, masterbiasimg=None, masterdarkimg=None, masterpixflatimg=None,
            masterillumflatimg=None, bpm_proc=None, mask_vig=False, minimum_vig=0.5, apply_gain=False, grow=1.5,
            maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex',
            sextractor_task='sex', verbose=True, overwrite=True):

    n_file = len(scifiles)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file
    sci_fits_list = []
    flag_fits_list = []

    if n_process == 1:
        for ii, scifile in enumerate(scifiles):
            sci_fits_file, flag_fits_file = _detproc_one(scifile, camera, det,
                        science_path=science_path, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg,
                        masterpixflatimg=masterpixflatimg, masterillumflatimg=masterillumflatimg, bpm_proc=bpm_proc,
                        mask_vig=mask_vig, minimum_vig=minimum_vig, apply_gain=apply_gain, grow=grow,
                        maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                        maskbrightstar_method=maskbrightstar_method, conv=conv,
                        sextractor_task=sextractor_task, verbose=verbose, overwrite=overwrite)
            sci_fits_list.append(sci_fits_file)
            flag_fits_list.append(flag_fits_file)
    else:
        msgs.info('Start parallel processing with n_process={:}'.format(n_process))
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put((scifiles[ii], camera, det))

        # creating processes
        for w in range(n_process):
            p = Process(target=_detproc_worker, args=(work_queue, done_queue), kwargs={
                'science_path': science_path, 'masterbiasimg': masterbiasimg,
                'masterdarkimg': masterdarkimg, 'masterpixflatimg': masterpixflatimg, 'masterillumflatimg': masterillumflatimg,
                'bpm_proc': bpm_proc, 'mask_vig': mask_vig, 'minimum_vig': minimum_vig, 'apply_gain': apply_gain,
                'maskbrightstar':maskbrightstar, 'brightstar_nsigma':brightstar_nsigma, 'maskbrightstar_method':maskbrightstar_method,
                'conv':conv, 'grow': grow, 'sextractor_task': sextractor_task, 'verbose':False, 'overwrite':overwrite})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

        # print the output
        while not done_queue.empty():
            sci_fits_file, flag_fits_file = done_queue.get()
            sci_fits_list.append(sci_fits_file)
            flag_fits_list.append(flag_fits_file)

    return sci_fits_list, flag_fits_list

def _detproc_one(scifile, camera, det, science_path=None, masterbiasimg=None, masterdarkimg=None, masterpixflatimg=None,
                 masterillumflatimg=None, bpm_proc=None, mask_vig=False, minimum_vig=0.5, apply_gain=False, grow=1.5,
                 maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex',
                 sextractor_task='sex', verbose=True, overwrite=True):

    rootname = scifile.split('/')[-1]
    if science_path is not None:
        rootname = os.path.join(science_path,rootname)
        if '.gz' in rootname:
            rootname = rootname.replace('.gz','')
        elif '.fz' in rootname:
            rootname = rootname.replace('.fz','')

    # prepare output file names
    sci_fits_file = rootname.replace('.fits','_det{:02d}_proc.fits'.format(det))
    ivar_fits_file = rootname.replace('.fits','_det{:02d}_proc.ivar.fits'.format(det))
    wht_fits_file = rootname.replace('.fits','_det{:02d}_proc.weight.fits'.format(det))
    flag_fits_file = rootname.replace('.fits','_det{:02d}_detmask.fits'.format(det))
    star_fits_file = rootname.replace('.fits','_det{:02d}_starmask.fits'.format(det))

    if os.path.exists(sci_fits_file) and not overwrite:
        msgs.info('The Science product {:} exists, skipping...'.format(sci_fits_file))
    else:
        msgs.info('Processing {:}'.format(scifile))
        detector_par, raw, header, exptime, rawdatasec_img, rawoscansec_img = camera.get_rawimage(scifile, det)
        saturation, nonlinear = detector_par['saturation'], detector_par['nonlinear']
        raw_image = trim_frame(raw, rawdatasec_img < 0.1)
        datasec_img = trim_frame(rawdatasec_img, rawdatasec_img < 0.1)
        oscansec_img = trim_frame(rawoscansec_img, rawdatasec_img < 0.1)

        # detector bad pixel mask
        bpm = camera.bpm(scifile, det, shape=None, msbias=None)>0.

        # Saturated pixel maskx
        bpm_sat = raw_image > saturation*nonlinear

        # Zero pixel mask
        bpm_zero = raw_image == 0.

        # mask nan values
        bpm_nan = np.isnan(raw_image) | np.isinf(raw_image)
        raw_image[bpm_nan] = 0.

        # mask generated from the processing of master files
        if bpm_proc is None:
            bpm_proc = np.zeros_like(raw_image, dtype='bool')

        ## Always set apply_gain to be True for reduction, otherwise the noise model would be incorrect.
        ## This parameter can be set to False when people want to measure the gain for your detectors.
        if apply_gain:
            sci_image = utils.gain_correct(raw_image, datasec_img, detector_par['gain'])
            header['GAIN'] = (1.0, 'Effective gain')
            header['UNITS'] = ('e-', 'Data units')
        else:
            sci_image = np.copy(raw_image)
            header['UNITS'] = ('ADU', 'Data units')

        # Detector Processing
        if masterbiasimg is not None:
            sci_image -= masterbiasimg # should I assume masterbiasimg has zero error?
        if masterdarkimg is not None:
            sci_image -= masterdarkimg*exptime # should I assume masterdarkimg has zero error?
        if masterpixflatimg is not None:
            sci_image *= utils.inverse(masterpixflatimg)
        if masterillumflatimg is not None:
            sci_image *= utils.inverse(masterillumflatimg)

        # Mask Vignetting pixels, should be done after gain correction and detector processing!
        if mask_vig:
            msgs.info('Masking significantly vignetting (>{:}%) pixels'.format(minimum_vig*100))
            if (masterillumflatimg is not None) and (np.sum(masterillumflatimg != 1) > 0):
                flat_for_vig = masterillumflatimg.copy()
            elif (masterpixflatimg is not None) and (np.sum(masterpixflatimg != 1) > 0):
                flat_for_vig = masterpixflatimg.copy()
            else:
                flat_for_vig = np.ones_like(sci_image)
            bpm_vig_1 = flat_for_vig < 1-minimum_vig
            bad_vig_cameras = ['magellan_imacsf2', 'keck_lris_blue', 'keck_lris_red']
            if camera.name in bad_vig_cameras:
                # ToDo: This is a hack. Not sure whether the following is safe or not.
                #  Basically, we are using the sky background for vignetting.
                #  IMACS need this since the guider moves around the detector.
                #  Keck LIRS also need this given its weid illumination at the edge.
                bpm_for_vig = bpm | bpm_sat | bpm_zero | bpm_proc | bpm_vig_1
                #starmask = mask_bright_star(sci_image, mask=bpm_for_vig, brightstar_nsigma=5., back_nsigma=3.,
                #                            back_maxiters=5, method='sextractor', conv=conv,
                #                            task=sextractor_task, verbose=verbose)
                bkg_for_vig, _ = BKG2D(sci_image, (50,50), mask=bpm_for_vig, filter_size=(3,3),
                                       sigclip=5., back_type='sextractor', verbose=verbose)
                bpm_vig_2 = sci_image < (1-minimum_vig) * np.median(bkg_for_vig[np.invert(bpm_for_vig)])
                bpm_vig_3 = bkg_for_vig < (1-minimum_vig) * np.median(bkg_for_vig[np.invert(bpm_for_vig)])
                bpm_vig_all = bpm_vig_1 | bpm_vig_2 | bpm_vig_3
                bpm_vig = grow_masked(bpm_vig_all, grow, verbose=verbose)
            else:
                bpm_vig = bpm_vig_1
        else:
            bpm_vig = np.zeros_like(sci_image, dtype=bool)

        # Get a total mask up to this stage
        bpm_for_wht = bpm | bpm_sat | bpm_zero | bpm_nan | bpm_proc | bpm_vig

        if maskbrightstar:
            msgs.info('Masking bright stars before calculating median sky background')
            starmask = mask_bright_star(sci_image, mask=bpm_for_wht, brightstar_nsigma=brightstar_nsigma,
                                        method=maskbrightstar_method, conv=conv, task=sextractor_task,
                                        verbose=verbose)
        else:
            starmask = np.zeros_like(sci_image, dtype=bool)

        # Calculate median sky background
        med_sky, med_rms = utils.pixel_stats(sci_image, bpm=bpm_for_wht | starmask, sigclip=3, n_clip=10,
                                             min_pix=int(np.size(sci_image)*0.2))
        msgs.info('Median sky value is {:0.2f} e-'.format(med_sky))
        # Use flat and sky background to generate a weight map
        msgs.info('Generating weight map using flat and median sky background')
        wht_image = utils.inverse(np.ones_like(sci_image) * med_sky)

        # Inverse variance map
        msgs.info('Generating inverse variance map')
        numamplifiers = detector_par['numamplifiers']
        ivar_image = np.zeros_like(raw_image)
        for iamp in range(numamplifiers):
            this_amp = datasec_img == iamp + 1
            ivar_image[this_amp] = utils.inverse(sci_image[this_amp] + detector_par['ronoise'][iamp]**2)

        ## Propagate errors
        #if masterbiasimg is not None:
        #    sci_image -= masterbiasimg # should I assume masterbiasimg has zero error?
        #if masterdarkimg is not None:
        #    sci_image -= masterdarkimg*exptime # should I assume masterdarkimg has zero error?
        if masterpixflatimg is not None:
            ivar_image *= masterpixflatimg**2
            wht_image *= masterpixflatimg**2
        if masterillumflatimg is not None:
            ivar_image *= masterillumflatimg**2
            wht_image *= masterillumflatimg**2

        ## Set viginetting pixel to be zero
        sci_image[bpm_vig] = 0.

        ## Update header
        header['MEDSKY'] = (med_sky, 'Median Sky Value, units e-')
        header['DETPROC'] = ('TRUE', 'DETPROC is done?')

        # save image
        io.save_fits(sci_fits_file, sci_image, header, 'SCI', overwrite=True)
        msgs.info('Science image {:} saved'.format(sci_fits_file))
        # save inverse variance map
        io.save_fits(ivar_fits_file, ivar_image, header, 'IVAR', overwrite=True)
        msgs.info('Inverse variance image {:} saved'.format(ivar_fits_file))
        # save weight map
        io.save_fits(wht_fits_file, wht_image, header, 'WEIGHT', overwrite=True)
        msgs.info('Weight image {:} saved'.format(wht_fits_file))
        # save flag image
        flag_image = bpm*np.int(2**0) + bpm_proc*np.int(2**1) + bpm_sat*np.int(2**2) + \
                     bpm_zero * np.int(2**3) + bpm_nan*np.int(2**4) + bpm_vig*np.int(2**5)
        io.save_fits(flag_fits_file, flag_image.astype('int32'), header, 'FLAG', overwrite=True)
        msgs.info('Flag image {:} saved'.format(flag_fits_file))
        # I save star mask here to avoid running maskbrightstar again in the future
        io.save_fits(star_fits_file, starmask.astype('int32'), header, 'FLAG', overwrite=True)
        msgs.info('Bright star mask image {:} saved'.format(star_fits_file))

        del(rawdatasec_img, rawoscansec_img, raw_image, datasec_img, oscansec_img)
        del(sci_image, ivar_image, wht_image, flag_image)
        del(bpm, bpm_proc, bpm_sat, bpm_zero, bpm_nan, bpm_vig)
        gc.collect()

    return sci_fits_file, flag_fits_file

def _detproc_worker(work_queue, done_queue, science_path=None, masterbiasimg=None, masterdarkimg=None, masterpixflatimg=None,
                    masterillumflatimg=None, bpm_proc=None, mask_vig=False, minimum_vig=0.5, apply_gain=False, grow=1.5,
                    maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex',
                    sextractor_task='sex', verbose=True, overwrite=True):

    """Multiprocessing worker for sciproc."""
    while not work_queue.empty():
        scifile, camera, det = work_queue.get()
        sci_fits_file, flag_fits_file = _detproc_one(scifile, camera, det,
                        science_path=science_path, masterbiasimg=masterbiasimg, masterdarkimg=masterdarkimg,
                        masterpixflatimg=masterpixflatimg, masterillumflatimg=masterillumflatimg,
                        bpm_proc=bpm_proc, mask_vig=mask_vig, minimum_vig=minimum_vig,
                        apply_gain=apply_gain, grow=grow,
                        maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma, conv=conv,
                        maskbrightstar_method=maskbrightstar_method,
                        sextractor_task=sextractor_task, verbose=verbose, overwrite=overwrite)

        done_queue.put((sci_fits_file, flag_fits_file))

def sciproc(scifiles, flagfiles, n_process=4, airmass=None, coeff_airmass=0., mastersuperskyimg=None, use_medsky=False,
            back_type='median', back_rms_type='std', back_size=(200,200), back_filtersize=(3, 3), back_maxiters=5, grow=1.5,
            maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex', sextractor_task='sex',
            mask_cr=True, contrast=2, lamaxiter=1, sigclip=5.0, cr_threshold=5.0, neighbor_threshold=2.0,
            mask_sat=True, sat_sig=3.0, sat_buf=20, sat_order=3, low_thresh=0.1, h_thresh=0.5,
            small_edge=60, line_len=200, line_gap=75, percentile=(4.5, 93.0),
            mask_negative_star=False, replace=None, verbose=True, overwrite=True):

    n_file = len(scifiles)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file
    sci_fits_list = []
    wht_fits_list = []
    flag_fits_list = []

    if airmass is not None:
        if len(airmass) != len(scifiles):
            msgs.error('The length of airmass table should be the same with the number of exposures.')
    else:
        airmass = [None]*n_file
    if n_process == 1:
        for ii, scifile in enumerate(scifiles):
            sci_fits_file, wht_fits_file, flag_fits_file = _sciproc_one(scifile, flagfiles[ii],
                airmass=airmass[ii], coeff_airmass=coeff_airmass, mastersuperskyimg=mastersuperskyimg, use_medsky=use_medsky,
                back_type=back_type, back_rms_type=back_rms_type, back_size=back_size, back_filtersize=back_filtersize,
                back_maxiters=back_maxiters, grow=grow, maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
                maskbrightstar_method=maskbrightstar_method, conv=conv, sextractor_task=sextractor_task,
                mask_cr=mask_cr, contrast=contrast, lamaxiter=lamaxiter, sigclip=sigclip, cr_threshold=cr_threshold,
                neighbor_threshold=neighbor_threshold, mask_sat=mask_sat, sat_sig=sat_sig, sat_buf=sat_buf,
                sat_order=sat_order, low_thresh=low_thresh, h_thresh=h_thresh,
                small_edge=small_edge, line_len=line_len, line_gap=line_gap, percentile=percentile,
                mask_negative_star=mask_negative_star, replace=replace, verbose=verbose, overwrite=overwrite)
            sci_fits_list.append(sci_fits_file)
            wht_fits_list.append(wht_fits_file)
            flag_fits_list.append(flag_fits_file)
    else:
        msgs.info('Start parallel processing with n_process={:}'.format(n_process))
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put((scifiles[ii], flagfiles[ii], airmass[ii]))

        # creating processes
        for w in range(n_process):
            p = Process(target=_sciproc_worker, args=(work_queue, done_queue), kwargs={
                'mastersuperskyimg': mastersuperskyimg, 'coeff_airmass': coeff_airmass, 'use_medsky':use_medsky,
                'back_type': back_type, 'back_rms_type': back_rms_type, 'back_size': back_size, 'back_filtersize': back_filtersize,
                'back_maxiters': back_maxiters, 'grow': grow, 'maskbrightstar': maskbrightstar, 'brightstar_nsigma': brightstar_nsigma,
                'maskbrightstar_method': maskbrightstar_method, 'conv':conv, 'sextractor_task': sextractor_task,
                'mask_cr': mask_cr, 'contrast': contrast, 'lamaxiter': lamaxiter, 'sigclip': sigclip, 'cr_threshold': cr_threshold,
                'neighbor_threshold': neighbor_threshold, 'mask_sat': mask_sat, 'sat_sig': sat_sig, 'sat_buf': sat_buf,
                'sat_order': sat_order, 'low_thresh': low_thresh, 'h_thresh': h_thresh,
                'small_edge': small_edge, 'line_len': line_len, 'line_gap': line_gap, 'percentile': percentile,
                'mask_negative_star': mask_negative_star, 'replace': replace, 'verbose':False, 'overwrite':overwrite})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

        # print the output
        while not done_queue.empty():
            sci_fits_file, wht_fits_file, flag_fits_file = done_queue.get()
            sci_fits_list.append(sci_fits_file)
            wht_fits_list.append(wht_fits_file)
            flag_fits_list.append(flag_fits_file)

    return sci_fits_list, wht_fits_list, flag_fits_list

def _sciproc_one(scifile, flagfile, airmass, coeff_airmass=0., mastersuperskyimg=None, use_medsky=False,
                 back_type='median', back_rms_type='std', back_size=(200,200), back_filtersize=(3, 3), back_maxiters=5, grow=1.5,
                 maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex', sextractor_task='sex',
                 mask_cr=True, contrast=2, lamaxiter=1, sigclip=5.0, cr_threshold=5.0, neighbor_threshold=2.0,
                 mask_sat=True, sat_sig=3.0, sat_buf=20, sat_order=3, low_thresh=0.1, h_thresh=0.5,
                 small_edge=60, line_len=200, line_gap=75, percentile=(4.5, 93.0),
                 mask_negative_star=False, replace=None, verbose=True, overwrite=True):

    # prepare output names
    sci_fits_file = scifile.replace('_proc.fits','_sci.fits')
    ivar_fits_file = scifile.replace('_proc.fits','_sci.ivar.fits')
    wht_fits_file = scifile.replace('_proc.fits','_sci.weight.fits')
    flag_fits_file = scifile.replace('_proc.fits','_flag.fits')
    star_fits_file = scifile.replace('_proc.fits','_starmask.fits')

    if os.path.exists(sci_fits_file) and not overwrite:
        msgs.info('The Science product {:} exists, skipping...'.format(sci_fits_file))
    else:
        msgs.info('Processing {:}'.format(scifile))
        header, data, _ = io.load_fits(scifile)
        _, ivar_image, _ = io.load_fits(scifile.replace('_proc.fits','_proc.ivar.fits'))
        _, wht_image, _ = io.load_fits(scifile.replace('_proc.fits','_proc.weight.fits'))
        _, flag_image, _ = io.load_fits(flagfile)
        bpm = flag_image>0
        bpm_zero = (data == 0.)
        bpm_saturation = (flag_image & 2**2)>0
        bpm_vig = (flag_image & 2**5)>0
        bpm_vig_only = np.logical_and(bpm_vig, np.invert(bpm_zero)) # pixels identified by vig but not zeros
        ## super flattening your images
        if mastersuperskyimg is not None:
            data *= utils.inverse(mastersuperskyimg)
            ivar_image *= mastersuperskyimg**2
            wht_image *= mastersuperskyimg**2

        # do the extinction correction.
        if airmass is not None:
            mag_ext = coeff_airmass * (airmass-1)
            data *= 10**(0.4*mag_ext)
            ivar_image *= 10**(-0.8*mag_ext)
            wht_image *= 10**(-0.8*mag_ext)

        # mask bright stars before estimating the background
        if maskbrightstar:
            if os.path.exists(star_fits_file):
                msgs.info('Loading star mask from existing star mask image')
                _, star_image, _ = io.load_fits(star_fits_file)
                starmask = star_image>0
            else:
                # do not mask viginetting pixels when estimating the background to reduce edge effect
                bpm_for_star = np.logical_and(bpm, np.invert(bpm_vig_only))
                starmask = mask_bright_star(data, mask=bpm_for_star, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigclip,
                                            back_maxiters=back_maxiters, method=maskbrightstar_method, task=sextractor_task,
                                            conv=conv, verbose=verbose)
                io.save_fits(star_fits_file, starmask.astype('int32'), header, 'FLAG', overwrite=True)
                msgs.info('Stark mask image {:} saved'.format(star_fits_file))
        else:
            starmask = np.zeros_like(data, dtype=bool)

        # estimate the 2D background with all masks
        bpm_for_bkg = np.logical_or(bpm, starmask)
        med_sky, med_rms = utils.pixel_stats(data, bpm=bpm_for_bkg, sigclip=sigclip, n_clip=back_maxiters,
                                             min_pix=int(np.size(data)*0.2))
        # ToDo: Scale the wht_image?
        wht_image *= header['MEDSKY']/med_sky
        # Update the medium sky value in fits header
        msgs.info('Median sky value is {:0.2f} e-'.format(med_sky))

        rms_image = utils.inverse(np.sqrt(ivar_image))

        if use_medsky:
            msgs.info('Using median sky counts for sky subtraction.')
            background_array = np.zeros_like(data) + med_sky
        else:
            background_array, _ = BKG2D(data, back_size, mask=bpm_for_bkg, filter_size=back_filtersize,
                                                     sigclip=sigclip, back_type=back_type, back_rms_type=back_rms_type,
                                                     back_maxiters=back_maxiters, sextractor_task=sextractor_task,
                                                     verbose=verbose)
        # set the background for zero pixels to zero
        background_array[bpm_zero] = 0.

        # subtract the background
        if verbose:
            msgs.info('Subtracting 2D background')
        sci_image = data-background_array

        # CR mask
        # do not trade saturation as bad pixel when searching for CR and satellite trail
        bpm_for_cr = np.logical_and(bpm, np.invert(bpm_saturation))
        if mask_cr:
            if verbose:
                msgs.info('Identifying cosmic rays using the L.A.Cosmic algorithm')
            bpm_cr_tmp = lacosmic(sci_image, contrast, cr_threshold, neighbor_threshold,
                                  error=rms_image, mask=bpm_for_cr, background=background_array, effective_gain=None,
                                  readnoise=None, maxiter=lamaxiter, border_mode='mirror', verbose=verbose)
            bpm_cr = grow_masked(bpm_cr_tmp, grow, verbose=verbose)
            del bpm_cr_tmp
        else:
            if verbose:
                msgs.warn('Skipped cosmic ray rejection process!')
            bpm_cr = np.zeros_like(sci_image,dtype=bool)

        # satellite trail mask
        if mask_sat:
            if verbose:
                msgs.info('Identifying satellite trails using the Canny algorithm following ACSTOOLS.')
            bpm_sat_tmp = satdet(sci_image, bpm=bpm_for_cr, sigma=sat_sig, buf=sat_buf, order=sat_order,
                                 low_thresh=low_thresh, h_thresh=h_thresh, small_edge=small_edge,
                                 line_len=line_len, line_gap=line_gap, percentile=percentile, verbose=verbose)
            bpm_sat = grow_masked(bpm_sat_tmp, grow, verbose=verbose)
            del bpm_sat_tmp
        else:
            bpm_sat = np.zeros_like(sci_image,dtype=bool)

        # negative star mask
        if mask_negative_star:
            if verbose:
                msgs.info('Masking negative stars with {:}'.format(maskbrightstar_method))
            bpm_negative_tmp = mask_bright_star(0-sci_image, mask=bpm, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigclip,
                                                back_maxiters=back_maxiters, method=maskbrightstar_method, task=sextractor_task,
                                                conv=conv, verbose=verbose)
            bpm_negative = grow_masked(bpm_negative_tmp, grow, verbose=verbose)
            del bpm_negative_tmp
        else:
            bpm_negative = np.zeros_like(sci_image, dtype=bool)

        # add the cosmic ray and satellite trail flag to the flag images
        flag_image_new = flag_image + bpm_cr.astype('int32')*np.int(2**6) + bpm_sat.astype('int32')*np.int(2**7)
        flag_image_new += bpm_negative.astype('int32')*np.int(2**8)

        ## replace bad pixels but not saturated pixels to make the image nicer
        #flag_image = bpm*np.int(2**0) + bpm_proc*np.int(2**1) + bpm_sat*np.int(2**2) + \
        #             bpm_zero * np.int(2**3) + bpm_nan*np.int(2**4) + bpm_vig*np.int(2**5)
        # ToDo: explore the replacement algorithm, replace a bad pixel using the median of a box
        bpm_all = flag_image_new>0
        bpm_replace = np.logical_and(bpm_all, np.invert(bpm_saturation))
        if replace == 'zero':
            sci_image[bpm_replace] = 0
        elif replace == 'median':
            _,sci_image[bpm_replace],_ = stats.sigma_clipped_stats(sci_image, bpm_all, sigma=sigclip, maxiters=5)
        elif replace == 'mean':
            sci_image[bpm_replace],_,_ = stats.sigma_clipped_stats(sci_image, bpm_all, sigma=sigclip, maxiters=5)
        elif replace == 'min':
            sci_image[bpm_replace] = np.min(sci_image[np.invert(bpm_all)])
        elif replace == 'max':
            sci_image[bpm_replace] = np.max(sci_image[np.invert(bpm_all)])
        else:
            if verbose:
                msgs.info('Not replacing bad pixel values')

        ## Generate weight map used for SExtractor and SWarp (WEIGHT_TYPE = MAP_WEIGHT)
        #wht_image = utils.inverse(background_array)
        # Set bad pixel's weight to be zero
        wht_image[flag_image_new>0] = 0

        # Update header
        header['MEDSKY'] = (med_sky, 'Median Sky Value, units e-')
        header['SCIPROC'] = ('TRUE', 'SCIPROC is done?')

        # save images
        io.save_fits(sci_fits_file, sci_image, header, 'SCI', overwrite=True)
        msgs.info('Science image {:} saved'.format(sci_fits_file))
        io.save_fits(ivar_fits_file, ivar_image, header, 'IVAR', overwrite=True)
        msgs.info('Inverse variance image {:} saved'.format(ivar_fits_file))
        io.save_fits(wht_fits_file, wht_image, header, 'WEIGHT', overwrite=True)
        msgs.info('Weight image {:} saved'.format(wht_fits_file))
        io.save_fits(flag_fits_file, flag_image_new.astype('int32'), header, 'FLAG', overwrite=True)
        msgs.info('Flag image {:} saved'.format(flag_fits_file))

        del(data, sci_image, ivar_image, wht_image, flag_image, flag_image_new, rms_image, background_array)
        del(bpm_all, bpm, bpm_replace, bpm_saturation, bpm_zero, bpm_vig_only, bpm_vig)
        del(bpm_for_bkg, bpm_negative, bpm_sat, bpm_for_cr, bpm_cr)
        gc.collect()

    return sci_fits_file, wht_fits_file, flag_fits_file

def _sciproc_worker(work_queue, done_queue, coeff_airmass=0., mastersuperskyimg=None, use_medsky=False,
                    back_type='median', back_rms_type='std', back_size=(200,200), back_filtersize=(3, 3), back_maxiters=5, grow=1.5,
                    maskbrightstar=True, brightstar_nsigma=3, maskbrightstar_method='sextractor', conv='sex', sextractor_task='sex',
                    mask_cr=True, contrast=2, lamaxiter=1, sigclip=5.0, cr_threshold=5.0, neighbor_threshold=2.0,
                    mask_sat=True, sat_sig=3.0, sat_buf=20, sat_order=3, low_thresh=0.1, h_thresh=0.5,
                    small_edge=60, line_len=200, line_gap=75, percentile=(4.5, 93.0),
                    mask_negative_star=False, replace=None, verbose=False, overwrite=True):

    """Multiprocessing worker for sciproc."""
    while not work_queue.empty():
        scifile, flagfile, airmass = work_queue.get()
        sci_fits_file, wht_fits_file, flag_fits_file = _sciproc_one(scifile, flagfile, airmass,
            coeff_airmass=coeff_airmass, mastersuperskyimg=mastersuperskyimg, use_medsky=use_medsky,
            back_type=back_type, back_rms_type=back_rms_type, back_size=back_size, back_filtersize=back_filtersize,
            back_maxiters=back_maxiters, grow=grow, maskbrightstar=maskbrightstar, brightstar_nsigma=brightstar_nsigma,
            maskbrightstar_method=maskbrightstar_method, conv=conv, sextractor_task=sextractor_task,
            mask_cr=mask_cr, contrast=contrast, lamaxiter=lamaxiter, sigclip=sigclip, cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold, mask_sat=mask_sat, sat_sig=sat_sig, sat_buf=sat_buf,
            sat_order=sat_order, low_thresh=low_thresh, h_thresh=h_thresh,
            small_edge=small_edge, line_len=line_len, line_gap=line_gap, percentile=percentile,
            mask_negative_star=mask_negative_star, replace=replace, verbose=verbose, overwrite=overwrite)

        done_queue.put((sci_fits_file, wht_fits_file, flag_fits_file))

def defringing(sci_fits_list, masterfringeimg):

    ## ToDo: matching the amplitude of friging rather than scale with exposure time.
    for i in range(len(sci_fits_list)):
        #ToDo: parallel this
        header, data, _ = io.load_fits(sci_fits_list[i])
        mask_zero = data == 0.
        if 'DEFRING' in header.keys():
            msgs.info('The De-fringed image {:} exists, skipping...'.format(sci_fits_list[i]))
        else:
            data -= masterfringeimg * header['EXPTIME']
            data[mask_zero] = 0
            header['DEFRING'] = ('TRUE', 'De-Fringing is done?')
            io.save_fits(sci_fits_list[i], data, header, 'ScienceImage', overwrite=True)
            msgs.info('De-fringed science image {:} saved'.format(sci_fits_list[i]))

def trim_frame(frame, mask):
    """
    Trim the masked regions from a frame.

    Args:
        frame (:obj:`numpy.ndarray`):
            Image to be trimmed
        mask (:obj:`numpy.ndarray`):
            Boolean image set to True for values that should be trimmed
            and False for values to be returned in the output trimmed
            image.

    Return:
        :obj:`numpy.ndarray`:
            Trimmed image

    Raises:
        PypPhotError:
            Error raised if the trimmed image includes masked values
            because the shape of the valid region is odd.
    """
    if np.any(mask[np.invert(np.all(mask,axis=1)),:][:,np.invert(np.all(mask,axis=0))]):
        msgs.error('Data section is oddly shaped.  Trimming does not exclude all '
                   'pixels outside the data sections.')
    return frame[np.invert(np.all(mask,axis=1)),:][:,np.invert(np.all(mask,axis=0))]

def grow_masked(img, grow, verbose=True):

    img = img.astype(float)
    growval =1.0
    if verbose:
        msgs.info('Growing mask by {:}'.format(grow))
    if not np.any(img == growval):
        return img.astype(bool)

    _img = img.copy()
    sz_x, sz_y = img.shape
    d = int(1+grow)
    rsqr = grow*grow

    # Grow any masked values by the specified amount
    for x in range(sz_x):
        for y in range(sz_y):
            if img[x,y] != growval:
                continue

            mnx = 0 if x-d < 0 else x-d
            mxx = x+d+1 if x+d+1 < sz_x else sz_x
            mny = 0 if y-d < 0 else y-d
            mxy = y+d+1 if y+d+1 < sz_y else sz_y

            for i in range(mnx,mxx):
                for j in range(mny, mxy):
                    if (i-x)*(i-x)+(j-y)*(j-y) <= rsqr:
                        _img[i,j] = growval
    return _img.astype(bool)

class ImageProc():
    """
    Class for handling imaging processing
    """

    def __init__(self, par, camera, det, science_path, master_key, raw_shape, reuse_masters=True, overwrite=True):

        self.par = par
        self.camera = camera
        self.det = det
        self.science_path = science_path
        self.master_key = master_key
        self.reuse_masters = reuse_masters
        self.overwrite = overwrite
        self.mastersuperskyimg = np.ones(raw_shape)
        self.masksuperskyimg = np.zeros(raw_shape, dtype='int32')
        self.masterfringeimg = np.ones(raw_shape)
        self.maskfringeimg = np.zeros(raw_shape, dtype='int32')

        self.sextask = self.par['rdx']['sextractor']
        self.n_process=self.par['rdx']['n_process']

        self.maskbrightstar = self.par['scienceframe']['process']['mask_brightstar']
        self.brightstar_nsigma = self.par['scienceframe']['process']['brightstar_nsigma']
        self.maskbrightstar_method = self.par['scienceframe']['process']['brightstar_method']
        self.conv= self.par['scienceframe']['process']['conv']

    def run_detproc(self, procfiles, masterbiasimg, masterdarkimg, masterpixflatimg, masterillumflatimg, bpm_proc):
        '''
        Bias, dark subtraction and flat fielding, support parallel processing
        Parameters
        ----------
        procfiles
        masterbiasimg
        masterdarkimg
        masterpixflatimg
        masterillumflatimg
        bpm_proc

        Returns
        -------

        '''

        proc_fits_list, detmask_fits_list = detproc(procfiles, self.camera, self.det,
                                                    science_path=self.science_path,
                                                    masterbiasimg=masterbiasimg,
                                                    masterdarkimg=masterdarkimg,
                                                    masterpixflatimg=masterpixflatimg,
                                                    masterillumflatimg=masterillumflatimg,
                                                    bpm_proc=bpm_proc,
                                                    maskbrightstar=self.maskbrightstar,
                                                    brightstar_nsigma=self.brightstar_nsigma,
                                                    maskbrightstar_method=self.maskbrightstar_method,
                                                    conv=self.conv,
                                                    apply_gain=self.par['scienceframe']['process']['apply_gain'],
                                                    mask_vig=self.par['scienceframe']['process']['mask_vig'],
                                                    minimum_vig=self.par['scienceframe']['process']['minimum_vig'],
                                                    grow=self.par['scienceframe']['process']['grow'],
                                                    sextractor_task=self.sextask,
                                                    n_process=self.n_process,
                                                    overwrite=self.overwrite)

    def build_supersky(self, superskyrawfiles):
        '''
        Build Supersky MasterFrame
        Parameters
        ----------
        superskyrawfiles

        Returns
        -------

        '''

        mastersupersky_name = os.path.join(self.par['calibrations']['master_dir'],
                                           'MasterSuperSky_{:}'.format(self.master_key))
        if os.path.exists(mastersupersky_name) and self.reuse_masters:
            msgs.info('Using existing master file {:}'.format(mastersupersky_name))
        else:
            if np.size(superskyrawfiles) < 3:
                msgs.warn('The number of SuperSky frames should be generally >=3.')
            superskyfiles = []
            superskymaskfiles = []
            for ifile in superskyrawfiles:
                rootname = os.path.join(self.science_path, ifile.split('/')[-1])
                if '.gz' in rootname:
                    rootname = rootname.replace('.gz', '')
                elif '.fz' in rootname:
                    rootname = rootname.replace('.fz', '')
                # prepare input file names
                superskyfile = rootname.replace('.fits', '_det{:02d}_proc.fits'.format(self.det))
                superskyfiles.append(superskyfile)
                superskymaskfile = rootname.replace('.fits', '_det{:02d}_detmask.fits'.format(self.det))
                superskymaskfiles.append(superskymaskfile)

            masterframe.superskyframe(superskyfiles, mastersupersky_name,
                                      maskfiles=superskymaskfiles,
                                      cenfunc=self.par['calibrations']['superskyframe']['process']['comb_cenfunc'],
                                      stdfunc=self.par['calibrations']['superskyframe']['process']['comb_stdfunc'],
                                      sigma=self.par['calibrations']['superskyframe']['process']['comb_sigrej'],
                                      maxiters=self.par['calibrations']['superskyframe']['process']['comb_maxiter'],
                                      window_size=self.par['calibrations']['superskyframe']['process']['window_size'],
                                      maskbrightstar=self.maskbrightstar,
                                      brightstar_nsigma=self.brightstar_nsigma,
                                      maskbrightstar_method=self.maskbrightstar_method,
                                      conv=self.conv,
                                      sextractor_task=self.sextask)
        _, self.mastersuperskyimg, self.masksuperskyimg = io.load_fits(mastersupersky_name)

    def run_sciproc(self, sciprocfiles, sciproc_airmass):
        '''
        Run sciproc
        Parameters
        ----------
        sciprocfiles
        sciproc_airmass

        Returns
        -------

        '''

        # Prepare lists for sciproc
        sciproc_fits_list = []
        scimask_fits_list = []
        for ifile in sciprocfiles:
            rootname = os.path.join(self.science_path, os.path.basename(ifile))
            sci_fits_file = rootname.replace('.fits', '_det{:02d}_proc.fits'.format(self.det))
            flag_fits_file = rootname.replace('.fits', '_det{:02d}_detmask.fits'.format(self.det))
            sciproc_fits_list.append(sci_fits_file)
            scimask_fits_list.append(flag_fits_file)

        ## Do the sciproc
        sci_fits_list, wht_fits_list, flag_fits_list = sciproc(sciproc_fits_list, scimask_fits_list,
                                                    mastersuperskyimg=self.mastersuperskyimg,
                                                    airmass=sciproc_airmass,
                                                    coeff_airmass=self.par['postproc']['photometry']['coeff_airmass'],
                                                    use_medsky=self.par['scienceframe']['process']['use_medsky'],
                                                    back_type=self.par['scienceframe']['process']['back_type'],
                                                    back_rms_type=self.par['scienceframe']['process']['back_rms_type'],
                                                    back_size=self.par['scienceframe']['process']['back_size'],
                                                    back_filtersize=self.par['scienceframe']['process']['back_filtersize'],
                                                    maskbrightstar=self.maskbrightstar,
                                                    brightstar_nsigma=self.brightstar_nsigma,
                                                    maskbrightstar_method=self.maskbrightstar_method,
                                                    conv=self.conv,
                                                    sigclip=self.par['scienceframe']['process']['sigclip'],
                                                    mask_cr=self.par['scienceframe']['process']['mask_cr'],
                                                    lamaxiter=self.par['scienceframe']['process']['lamaxiter'],
                                                    cr_threshold=self.par['scienceframe']['process']['cr_threshold'],
                                                    neighbor_threshold=self.par['scienceframe']['process']['neighbor_threshold'],
                                                    contrast=self.par['scienceframe']['process']['contrast'],
                                                    grow=self.par['scienceframe']['process']['grow'],
                                                    # sigfrac=self.par['scienceframe']['process']['sigfrac'],
                                                    # objlim=self.par['scienceframe']['process']['objlim'],
                                                    mask_sat=self.par['scienceframe']['process']['mask_sat'],
                                                    sat_sig=self.par['scienceframe']['process']['sat_sig'],
                                                    sat_buf=self.par['scienceframe']['process']['sat_buf'],
                                                    sat_order=self.par['scienceframe']['process']['sat_order'],
                                                    low_thresh=self.par['scienceframe']['process']['low_thresh'],
                                                    h_thresh=self.par['scienceframe']['process']['h_thresh'],
                                                    small_edge=self.par['scienceframe']['process']['small_edge'],
                                                    line_len=self.par['scienceframe']['process']['line_len'],
                                                    line_gap=self.par['scienceframe']['process']['line_gap'],
                                                    percentile=self.par['scienceframe']['process']['percentile'],
                                                    replace=self.par['scienceframe']['process']['replace'],
                                                    mask_negative_star=self.par['scienceframe']['process']['mask_negative_star'],
                                                    sextractor_task=self.sextask,
                                                    n_process=self.n_process, overwrite=self.overwrite)

    def build_fringe(self, fringerawfiles):
        '''
        Build a master fringe frame
        Parameters
        ----------
        fringerawfiles

        Returns
        -------

        '''

        masterfringe_name = os.path.join(self.par['calibrations']['master_dir'],
                                         'MasterFringe_{:}'.format(self.master_key))
        if os.path.exists(masterfringe_name) and self.reuse_masters:
            msgs.info('Using existing master file {:}'.format(masterfringe_name))
        else:
            if np.size(fringerawfiles) < 3:
                msgs.warn('The number of Fringe images should be generally >=3.')
            fringefiles = []
            fringemaskfiles = []
            for ifile in fringerawfiles:
                rootname = os.path.join(self.science_path, ifile.split('/')[-1])
                if '.gz' in rootname:
                    rootname = rootname.replace('.gz', '')
                elif '.fz' in rootname:
                    rootname = rootname.replace('.fz', '')
                # prepare input file names
                fringefile = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(self.det))
                fringefiles.append(fringefile)
                fringemaskfile = rootname.replace('.fits', '_det{:02d}_flag.fits'.format(self.det))
                fringemaskfiles.append(fringemaskfile)

            masterframe.fringeframe(fringefiles, masterfringe_name,
                                    fringemaskfiles=fringemaskfiles, mastersuperskyimg=self.mastersuperskyimg,
                                    cenfunc=self.par['calibrations']['fringeframe']['process']['comb_cenfunc'],
                                    stdfunc=self.par['calibrations']['fringeframe']['process']['comb_stdfunc'],
                                    sigma=self.par['calibrations']['fringeframe']['process']['comb_sigrej'],
                                    maxiters=self.par['calibrations']['fringeframe']['process']['comb_maxiter'],
                                    maskbrightstar=self.maskbrightstar,
                                    brightstar_nsigma=self.brightstar_nsigma,
                                    maskbrightstar_method=self.maskbrightstar_method,
                                    conv=self.conv,
                                    sextractor_task=self.sextask)
        _, self.masterfringeimg, self.maskfringeimg = io.load_fits(masterfringe_name)

    def run_defringing(self, scifiles):
        '''
        Remove fringing from science exposures
        Parameters
        ----------
        scifiles

        Returns
        -------

        '''

        # Prepare lists for defringing
        sci_fits_list = []
        for ifile in scifiles:
            rootname = os.path.join(self.science_path, os.path.basename(ifile))
            sci_fits_file = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(self.det))
            sci_fits_list.append(sci_fits_file)

        defringing(sci_fits_list, self.masterfringeimg)

