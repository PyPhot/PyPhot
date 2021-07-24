import os,gc
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Process, Queue

from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from astropy import stats

from pyphot import msgs, io, utils, caloffset
from pyphot import sex, scamp, swarp
from pyphot import crossmatch
from pyphot.query import query
from pyphot.photometry import mask_bright_star, photutils_detect
from pyphot.psf import  psf


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

def astrometric(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, n_process=4, science_path='./', qa_path='./',
                task='sex', detect_thresh=5.0, analysis_thresh=5.0, detect_minarea=5, crossid_radius=1.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', astrefmag_limits=None,
                position_maxerr=1.0, distort_degrees=3, pixscale_maxerr=1.1, posangle_maxerr=10.0,
                stability_type='INSTRUMENT', mosaic_type='LOOSE', weight_type='MAP_WEIGHT',
                skip_swarp_align=False, solve_photom_scamp=False, scamp_second_pass=False,
                delete=False, log=True, verbose=True):

    n_file = len(sci_fits_list)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    sci_fits_list_resample = []
    wht_fits_list_resample = []
    flag_fits_list_resample = []
    cat_fits_list_resample = []

    if n_process == 1:
        for ii, sci_fits_file in enumerate(sci_fits_list):
            sci_fits_resample, wht_fits_resample, flag_fits_resample, cat_fits_resample = _astrometric_one(
                    sci_fits_file, wht_fits_list[ii], flag_fits_list[ii], pixscale,
                    science_path=science_path, qa_path=qa_path, task=task,
                    detect_thresh=detect_thresh, analysis_thresh=analysis_thresh,
                    detect_minarea=detect_minarea, crossid_radius=crossid_radius,
                    astref_catalog=astref_catalog, astref_band=astref_band,
                    astrefmag_limits=astrefmag_limits, position_maxerr=position_maxerr,
                    distort_degrees=distort_degrees, pixscale_maxerr=pixscale_maxerr,
                    posangle_maxerr=posangle_maxerr, stability_type=stability_type,
                    mosaic_type=mosaic_type, weight_type=weight_type,
                    skip_swarp_align=skip_swarp_align, solve_photom_scamp=solve_photom_scamp,
                    scamp_second_pass=scamp_second_pass, delete=delete, log=log, verbose=verbose)
            sci_fits_list_resample.append(sci_fits_resample)
            wht_fits_list_resample.append(wht_fits_resample)
            flag_fits_list_resample.append(flag_fits_resample)
            cat_fits_list_resample.append(cat_fits_resample)
    else:
        msgs.info('Start parallel processing with n_process={:}'.format(n_process))
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put((sci_fits_list[ii], wht_fits_list[ii], flag_fits_list[ii], pixscale))

        # creating processes
        for w in range(n_process):
            p = Process(target=_astrometric_worker, args=(work_queue, done_queue), kwargs={
                'science_path': science_path, 'qa_path': qa_path,
                'task': task, 'detect_thresh': detect_thresh, 'analysis_thresh': analysis_thresh, 'detect_minarea': detect_minarea,
                'crossid_radius': crossid_radius, 'astref_catalog': astref_catalog, 'astref_band': astref_band,
                'astrefmag_limits': astrefmag_limits, 'position_maxerr': position_maxerr,
                'distort_degrees': distort_degrees, 'pixscale_maxerr': pixscale_maxerr, 'posangle_maxerr': posangle_maxerr,
                'stability_type': stability_type, 'mosaic_type': mosaic_type, 'weight_type': weight_type,
                'skip_swarp_align': skip_swarp_align, 'solve_photom_scamp': solve_photom_scamp,
                'scamp_second_pass': scamp_second_pass, 'delete': delete, 'log': log, 'verbose': False})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

        # print the output
        while not done_queue.empty():
            sci_fits_resample, wht_fits_resample, flag_fits_resample, cat_fits_resample = done_queue.get()
            sci_fits_list_resample.append(sci_fits_resample)
            wht_fits_list_resample.append(wht_fits_resample)
            flag_fits_list_resample.append(flag_fits_resample)
            cat_fits_list_resample.append(cat_fits_resample)

    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample, cat_fits_list_resample

def _astrometric_one(sci_fits_file, wht_fits_file, flag_fits_file, pixscale, science_path='./',qa_path='./',
                     task='sex',detect_thresh=5.0, analysis_thresh=5.0, detect_minarea=5, crossid_radius=1.0,
                     astref_catalog='GAIA-DR2', astref_band='DEFAULT', astrefmag_limits=None,
                     position_maxerr=1.0, distort_degrees=3, pixscale_maxerr=1.1, posangle_maxerr=10.0,
                     stability_type='INSTRUMENT', mosaic_type='LOOSE', weight_type='MAP_WEIGHT',
                     skip_swarp_align=False, solve_photom_scamp=False, scamp_second_pass=False,
                     delete=False, log=True, verbose=True):

    msgs.info('Performing astrometric calibration for {:}'.format(sci_fits_file))
    # configuration for the swarp run
    # Note that I would apply the gain correction before doing the astrometric calibration, so I set Gain to 1.0
    # resample science image
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                   "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT": 1.0,
                   "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                   "RESAMPLING_TYPE": 'NEAREST',
                   # I would always set this to NEAREST for individual exposures to avoid interpolation
                   "HEADER_SUFFIX": "_cat.head"}
    # configuration for swarp the flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'

    if skip_swarp_align:
        if verbose:
            msgs.info('Skipping the first alignment step with Swarp')
        os.system('cp {:} {:}'.format(sci_fits_file,
                                      sci_fits_file.replace('.fits', '.resamp.fits')))
        os.system('cp {:} {:}'.format(sci_fits_file.replace('.fits', '.weight.fits'),
                                      sci_fits_file.replace('.fits', '.resamp.weight.fits')))
        os.system('cp {:} {:}'.format(flag_fits_file,
                                      flag_fits_file.replace('.fits', '.resamp.fits')))
    else:
        ## This step is basically align the image to N to the up and E to the left.
        ## It is important if your image has a ~180 degree from the nominal orientation.
        if verbose:
            msgs.info('Running Swarp for the first pass to align the science image.')
        swarp.swarpone(sci_fits_file, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                       delete=delete, log=log, verbose=verbose)
        # resample flag image
        if verbose:
            msgs.info('Running Swarp for the first pass to align the flag image.')
        swarp.swarpone(flag_fits_file, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                       delete=delete, log=False, verbose=False)

    ## remove useless data and change flag image type to int32
    sci_fits_resample = sci_fits_file.replace('.fits', '.resamp.fits')
    wht_fits_resample = sci_fits_file.replace('.fits', '.resamp.weight.fits')
    flag_fits_resample = flag_fits_file.replace('.fits', '.resamp.fits')
    cat_fits_resample = sci_fits_file.replace('.fits', '.resamp_cat.fits')
    if not skip_swarp_align:
        par = fits.open(flag_fits_resample, memmap=False)
        par[0].data = par[0].data.astype('int32')  # change the dtype to be int32
        par[0].writeto(flag_fits_resample, overwrite=True)
        del par[0].data
        par.close()
        gc.collect()
        os.system('rm {:}'.format(flag_fits_file.replace('.fits', '.resamp.weight.fits')))

    # configuration for SExtractor run
    sexconfig0 = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_TYPE": "FITS_LDAC",
                  "DETECT_THRESH": detect_thresh,
                  "ANALYSIS_THRESH": analysis_thresh,
                  "DETECT_MINAREA": detect_minarea}
    sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                  'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                  'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER',
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    if verbose:
        msgs.info('Running SExtractor for the first pass to extract catalog used for SCAMP.')
    sex.sexone(sci_fits_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex', nnw=None, dual=False,
               flag_image=flag_fits_resample, weight_image=wht_fits_resample,
               delete=delete, log=log, verbose=verbose)

    # configuration for the scamp run
    if solve_photom_scamp:
        SOLVE_PHOTOM='Y'
    else:
        SOLVE_PHOTOM='N'

    if astrefmag_limits is not None:
        ASTREFMAG_LIMITS = '{:},{:}'.format(astrefmag_limits[0],astrefmag_limits[1])
    else:
        ASTREFMAG_LIMITS = '-99.0,99.0'

    scampconfig = {"CROSSID_RADIUS": crossid_radius,
                    "ASTREF_CATALOG": astref_catalog,
                    "ASTREF_BAND": astref_band,
                    "ASTREFMAG_LIMITS":ASTREFMAG_LIMITS,
                    "POSITION_MAXERR": position_maxerr,
                    "PIXSCALE_MAXERR": pixscale_maxerr,
                    "POSANGLE_MAXERR": posangle_maxerr,
                    "STABILITY_TYPE": stability_type,
                    "MOSAIC_TYPE": mosaic_type,
                    "SOLVE_PHOTOM": SOLVE_PHOTOM,
                    "DISTORT_DEGREES":distort_degrees,
                    "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                    "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
    if scamp_second_pass:
        # first run with distort_degrees of 1
        scampconfig1 = scampconfig.copy()
        scampconfig1['DISTORT_DEGREES'] = np.max([1,distort_degrees-2])
        if verbose:
            msgs.info('Running the first pass SCAMP with DISTORT_DEGREES of {:}'.format(scampconfig1['DISTORT_DEGREES']))
        scamp.scampone(cat_fits_resample, config=scampconfig1, workdir=science_path, QAdir=qa_path,
                       defaultconfig='pyphot', delete=delete, log=log)
        ## we can make the maximum errors to be smaller for the second pass.
        scampconfig['POSITION_MAXERR'] = np.min([1.0,position_maxerr])
        scampconfig['PIXSCALE_MAXERR'] = np.min([1.2,pixscale_maxerr])
        scampconfig['POSANGLE_MAXERR'] = np.min([5.0,posangle_maxerr])
        # copy the .head to .ahead
        os.system('mv {:} {:}'.format(cat_fits_resample.replace('.fits', '.head'),
                                      cat_fits_resample.replace('.fits', '.ahead')))
    # run the final scamp
    if verbose:
        msgs.info('Running the final pass of SCAMP with DISTORT_DEGREES of {:}'.format(distort_degrees))
    scamp.scampone(cat_fits_resample, config=scampconfig, workdir=science_path, QAdir=qa_path,
                   defaultconfig='pyphot', delete=delete, log=log, verbose=verbose)

    ## copy the .head for flag images
    os.system('cp {:} {:}'.format(sci_fits_resample.replace('.fits', '_cat.head'),
                                  flag_fits_resample.replace('.fits', '_cat.head')))

    ## configuration for the second swarp run
    swarpconfig['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    # resample the science image
    if verbose:
        msgs.info('Running Swarp for the second pass to align the science image.')
    swarp.swarpone(sci_fits_resample, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   delete=delete, log=log, verbose=verbose)
    # resample the flag image
    swarpconfig_flag['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    if verbose:
        msgs.info('Running Swarp for the second pass to align the flag image.')
    swarp.swarpone(flag_fits_resample, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   delete=delete, log=False, verbose=False)

    # delete unnecessary weight maps and head
    os.system('rm {:}'.format(flag_fits_file.replace('.fits', '.resamp.weight.fits')))
    os.system('rm {:}'.format(sci_fits_file.replace('.fits', '.resamp_cat.fits')))
    os.system('rm {:}'.format(sci_fits_file.replace('.fits', '.resamp_cat.head')))
    os.system('rm {:}'.format(flag_fits_file.replace('.fits', '.resamp_cat.head')))

    if scamp_second_pass:
        os.system('rm {:}'.format(sci_fits_file.replace('.fits', '.resamp_cat.ahead')))

    # Update fluxscale
    if solve_photom_scamp:
        if verbose:
            msgs.info('The FLXSCALE was solved with scamp.')
    else:
        if verbose:
            msgs.info('Solving the FLXSCALE for {:} with 1/EXPTIME.'.format(os.path.basename(sci_fits_resample)))
        par = fits.open(sci_fits_resample, memmap=False)
        # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
        # in order to get the correct flag when doing the coadd with swarp in the later step
        par[0].header['FLXSCALE'] = utils.inverse(par[0].header['EXPTIME'])
        par[0].writeto(sci_fits_resample, overwrite=True)
        del par[0].data
        par.close()

    # change flag image type to int32, flux scale to 1.0
    if verbose:
        msgs.info('Saving FLAG image {:} to int32.'.format(os.path.basename(flag_fits_resample)))
    par = fits.open(flag_fits_resample, memmap=False)
    # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
    # in order to get the correct flag when doing the coadd with swarp in the later step
    par[0].header['EXPTIME'] = 1.0
    par[0].header['FLXSCALE'] = 1.0
    par[0].header['FLASCALE'] = 1.0
    par[0].data = par[0].data.astype('int32')
    par[0].writeto(flag_fits_resample, overwrite=True)
    del par[0].data
    par.close()
    gc.collect()

    ## Run SExtractor for the resampled images. The catalogs will be used for calibrating individual chips.
    if verbose:
        msgs.info('Running SExtractor for the second pass to extract catalog for resampled images.')
    sex.sexone(sci_fits_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex', nnw=None, dual=False,
               flag_image=flag_fits_resample, weight_image=wht_fits_resample,
               delete=delete, log=log, verbose=verbose)

    msgs.info('Finished astrometric calibration for {:}'.format(sci_fits_file))

    return sci_fits_resample, wht_fits_resample, flag_fits_resample, cat_fits_resample

def _astrometric_worker(work_queue, done_queue, science_path='./',qa_path='./',
                task='sex',detect_thresh=5.0, analysis_thresh=5.0, detect_minarea=5, crossid_radius=1.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', astrefmag_limits=None,
                position_maxerr=1.0, distort_degrees=3, pixscale_maxerr=1.1, posangle_maxerr=10.0,
                stability_type='INSTRUMENT', mosaic_type='LOOSE', weight_type='MAP_WEIGHT',
                skip_swarp_align=False, solve_photom_scamp=False, scamp_second_pass=False,
                delete=False, log=True, verbose=True):

    """Multiprocessing worker for sciproc."""
    while not work_queue.empty():
        sci_fits_file, wht_fits_file, flag_fits_file, pixscale = work_queue.get()
        sci_fits_resample, wht_fits_resample, flag_fits_resample, cat_fits_resample = _astrometric_one(
            sci_fits_file, wht_fits_file, flag_fits_file, pixscale,
            science_path=science_path, qa_path=qa_path, task=task,
            detect_thresh=detect_thresh, analysis_thresh=analysis_thresh,
            detect_minarea=detect_minarea, crossid_radius=crossid_radius,
            astref_catalog=astref_catalog, astref_band=astref_band,
            astrefmag_limits=astrefmag_limits, position_maxerr=position_maxerr,
            distort_degrees=distort_degrees, pixscale_maxerr=pixscale_maxerr,
            posangle_maxerr=posangle_maxerr, stability_type=stability_type,
            mosaic_type=mosaic_type, weight_type=weight_type,
            skip_swarp_align=skip_swarp_align, solve_photom_scamp=solve_photom_scamp,
            scamp_second_pass=scamp_second_pass, delete=delete, log=log, verbose=verbose)

        done_queue.put((sci_fits_resample, wht_fits_resample, flag_fits_resample, cat_fits_resample))

def coadd(scifiles, flagfiles, coaddroot, pixscale, science_path, coadddir, weight_type='MAP_WEIGHT',
          rescale_weights=False, combine_type='median', clip_ampfrac=0.3, clip_sigma=4.0, blank_badpixels=False,
          subtract_back= False, back_type='AUTO', back_default=0.0, back_size=100, back_filtersize=3,
          back_filtthresh=0.0, resampling_type='LANCZOS3', detect_thresh=5, analysis_thresh=5,
          detect_minarea=5, sextractor_task='sex',delete=True, log=True):

    ## configuration for the Swarp run and do the coadd with Swarp
    msgs.info('Coadding image for {:}'.format(os.path.join(coadddir,coaddroot)))
    if rescale_weights:
        rescale='Y'
    else:
        rescale='N'
    if blank_badpixels:
        blank='Y'
    else:
        blank='N'
    if subtract_back:
        subtract='Y'
    else:
        subtract='N'

    ## parameters for coadding science images
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "RESAMPLE_SUFFIX": ".tmp.fits",
                   "PIXELSCALE_TYPE": "MANUAL", "PIXEL_SCALE": pixscale,
                   "WEIGHT_TYPE": weight_type,"RESCALE_WEIGHTS": rescale,"BLANK_BADPIXELS":blank,
                   "COMBINE_TYPE": combine_type.upper(),"CLIP_AMPFRAC":clip_ampfrac,"CLIP_SIGMA":clip_sigma,
                   "SUBTRACT_BACK": subtract,"BACK_TYPE":back_type,"BACK_DEFAULT":back_default,"BACK_SIZE":back_size,
                   "BACK_FILTERSIZE":back_filtersize,"BACK_FILTTHRESH":back_filtthresh, "RESAMPLING_TYPE":resampling_type}

    swarp.swarpall(scifiles, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coadddir=coadddir, coaddroot=coaddroot + '_sci', delete=delete, log=log)

    ## parameters for coadding flag images
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = "NONE"
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    swarp.swarpall(flagfiles, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coadddir=coadddir, coaddroot=coaddroot + '_flag', delete=delete, log=log)
    # delete unnecessary files
    os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.swarp.xml')))
    os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.weight.fits')))

    # useful file names
    coadd_file = os.path.join(coadddir,coaddroot+'_sci.fits')
    coadd_flag_file = os.path.join(coadddir, coaddroot + '_flag.fits')
    coadd_wht_file = os.path.join(coadddir, coaddroot + '_sci.weight.fits')

    # ToDO: Subtract background for the final coadded image

    # change flag image type to int32
    par = fits.open(coadd_flag_file, memmap=False)
    par[0].data = np.round(par[0].data).astype('int32')
    par[0].writeto(coadd_flag_file, overwrite=True)

    # Extract a catalog for zero point calibration in the next step.
    # configuration for the SExtractor run
    sexconfig = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_TYPE": "FITS_LDAC",
                  "DETECT_THRESH": detect_thresh,
                  "ANALYSIS_THRESH": analysis_thresh,
                  "DETECT_MINAREA": detect_minarea}
    sexparams = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                  'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                  'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER',
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    msgs.info('Running SExtractor for the coadded image. The catalog will be used for zero-point calibration.')
    sex.sexone(coadd_file, catname=coadd_file.replace('.fits','_zptcat.fits'),
               flag_image=coadd_flag_file, weight_image=coadd_wht_file,
               task=sextractor_task, config=sexconfig, workdir=coadddir, params=sexparams,
               defaultconfig='pyphot', dual=False, conv='sex', nnw=None, delete=True, log=False)

    return coadd_file, coadd_wht_file, coadd_flag_file

def detect(sci_image, outroot=None, flag_image=None, weight_image=None, bkg_image=None, rms_image=None,
           workdir='./', detection_method='sextractor', zpt=0.,
           effective_gain=None, pixscale=1.0, detect_thresh=2., analysis_thresh=2., detect_minarea=5, fwhm=5, nlevels=32, contrast=0.001,
           back_type='median', back_rms_type='std', back_size=(100, 100), back_filter_size=(3, 3), back_default=0.,
           backphoto_type='GLOBAL', backphoto_thick=100, weight_type='MAP_WEIGHT', check_type='BACKGROUND_RMS',
           back_nsigma=3, back_maxiters=10, morp_filter=False,
           defaultconfig='pyphot', dual=False, conv=None, nnw=None, delete=True, log=False,
           sextractor_task='sex', phot_apertures=[1.0,2.0,3.0,4.0,5.0]):

    if outroot is None:
        if dual:
            outroot = sci_image[0].replace('.fits','')
        else:
            outroot = sci_image.replace('.fits','')

    catname = outroot+'_cat.fits'

    if detection_method.lower() == 'photutils':
        # detection with photoutils
        msgs.info('Detecting sources with Photutils.')
        header, data, _ = io.load_fits(os.path.join(workdir,sci_image))
        wcs_info = wcs.WCS(header)
        if effective_gain is None:
            try:
                effective_gain = header['EXPTIME']
            except:
                effective_gain = 1.0

        # prepare mask
        if flag_image is not None:
            _, flag, _ = io.load_fits(os.path.join(workdir,flag_image))
            mask = flag > 0.
        else:
            mask = np.isinf(data) | np.isnan(data) | (data == 0.)

        if rms_image is not None:
            _, rmsmap, _ = io.load_fits(os.path.join(workdir,rms_image))
        else:
            rmsmap = None

        if bkg_image is not None:
            _, bkgmap, _ = io.load_fits(os.path.join(workdir,bkg_image))
        else:
            bkgmap = None

        # do the detection with photutils
        phot_table, phot_rmsmap, phot_bkgmap = photutils_detect(data, wcs_info, mask=mask, rmsmap=rmsmap, bkgmap=bkgmap, zpt=zpt,
                                                      effective_gain=effective_gain, nsigma=detect_thresh, npixels=detect_minarea,
                                                      fwhm=fwhm, nlevels=nlevels, contrast=contrast, back_nsigma=back_nsigma,
                                                      back_maxiters=back_maxiters, back_type=back_type, back_rms_type=back_rms_type,
                                                      back_size=back_size, back_filter_size=back_filter_size,
                                                      morp_filter=morp_filter, phot_apertures=phot_apertures)
        ## save the table and maps
        phot_table.write(os.path.join(workdir, catname), overwrite=True)
        if rms_image is None:
            io.save_fits(os.path.join(workdir, '{:}_rms.fits'.format(outroot)), phot_rmsmap, header, 'RMSMAP', overwrite=True)
        if bkg_image is None:
            io.save_fits(os.path.join(workdir, '{:}_bkg.fits'.format(outroot)), phot_rmsmap, header, 'BKGMAP', overwrite=True)

    elif detection_method.lower() == 'sextractor':
        # detection with SExtractor
        msgs.info('Detecting sources with SExtractor.')
        if check_type is not None:
            check_list = check_type.split(',')
            check_name_tmp = []
            for icheck in check_list:
                if icheck == 'BACKGROUND_RMS':
                    check_name_tmp.append('{:}_rms.fits'.format(outroot))
                elif icheck=='BACKGROUND':
                    check_name_tmp.append('{:}_bkg.fits'.format(outroot))
                else:
                    check_name_tmp.append('{:}_{:}.fits'.format(outroot, icheck.lower()))

            check_name = ','.join([str(elem) for elem in check_name_tmp])
        else:
            check_type='NONE'
            check_name='NONE'
        # configuration for the SExtractor run
        det_params = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE',
                      'ERRBWIN_IMAGE', 'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE',
                      'ISOAREA_IMAGE', 'ELLIPTICITY', 'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO',
                      'FLUXERR_AUTO', 'MAG_APER({:})'.format(len(phot_apertures)),
                      'MAGERR_APER({:})'.format(len(phot_apertures)),
                      'FLUX_APER({:})'.format(len(phot_apertures)),
                      'FLUXERR_APER({:})'.format(len(phot_apertures)),
                      'IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS']
        det_config = {"CATALOG_TYPE": "FITS_LDAC",
                      "MAG_ZEROPOINT": zpt,
                      "BACK_TYPE": back_type, "BACK_VALUE": back_default, "BACK_SIZE": back_size,
                      "BACK_FILTERSIZE": back_filter_size, "BACKPHOTO_TYPE": backphoto_type,
                      "BACKPHOTO_THICK": backphoto_thick, "WEIGHT_TYPE": weight_type,
                      "DETECT_THRESH": detect_thresh, "ANALYSIS_THRESH": analysis_thresh,
                      "DETECT_MINAREA": detect_minarea, "DEBLEND_NTHRESH": nlevels,
                      "DEBLEND_MINCONT": contrast, "CHECKIMAGE_TYPE": check_type,
                      "CHECKIMAGE_NAME": check_name,
                      "PHOT_APERTURES": np.array(phot_apertures) / pixscale}
        if weight_image is None and weight_type!= 'NONE':
            msgs.error('Please either provide weight_image or set weight_type to be NONE.')
        if flag_image is None:
            msgs.warn('flag_image is not given, generate a mock zero flag image')
            flag_image = os.path.join(workdir, 'flag_tmp_{:03d}.fits'.format(np.random.randint(1, 999)))
            tmp_flag = flag_image
            header, data, _ = io.load_fits(os.path.join(workdir, sci_image))
            io.save_fits(flag_image, np.zeros_like(data,dtype='int32'), header, 'FLAGMAP', overwrite=True)
        else:
            tmp_flag=None

        # do the source extraction
        sex.sexone(sci_image, catname=catname, flag_image=flag_image, weight_image=weight_image,
                   task=sextractor_task, config=det_config, workdir=workdir, params=det_params,
                   defaultconfig=defaultconfig, dual=dual, conv=conv, nnw=nnw, delete=delete, log=log)

        # delete temperary file
        if tmp_flag is not None:
            os.system('rm {:}'.format(tmp_flag))

        if 'BACKGROUND_RMS' in check_list:
            phot_rmsmap = fits.getdata(os.path.join(workdir, '{:}_rms.fits'.format(outroot)))
        elif rms_image is not None:
            _, phot_rmsmap, _ = io.load_fits(os.path.join(workdir,rms_image))
        else:
            phot_rmsmap = None

        if 'BACKGROUND' in check_list:
            phot_bkgmap = fits.getdata(os.path.join(workdir, '{:}_bkg.fits'.format(outroot)))
        elif bkg_image is not None:
            _, phot_bkgmap, _ = io.load_fits(os.path.join(workdir,bkg_image))
        else:
            phot_bkgmap = None

        phot_table = Table.read(os.path.join(workdir, catname), 2)

    else:
        msgs.warn('{:} is not supported yet.'.format(detection_method))
        phot_table, phot_rmsmap, phot_bkgmap =  None, None, None

    return phot_table, phot_rmsmap, phot_bkgmap

def calzpt(catalogfits, refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.],
           oversize=1.0, external_flag=True, FLXSCALE=1.0, FLASCALE=1.0,
           nstar_min=10, out_refcat=None, outqaroot=None, verbose=True):

    try:
        if verbose:
            msgs.info('Reading SExtractor catalog')
        catalog = Table.read(catalogfits, 2)
        # ToDo:  (catalog['NIMAFLAGS_ISO']<1) will reject most of the targets for dirty IR detector, i.e. WIRCam
        #       So, we should save another flat image that only counts for the number of bad exposures associated to the pixel
        #       and then use this number as a cut.
        #       Currently we only remove saturated targets, catalog['NIMAFLAGS_ISO'] & 2**2<1 used for remove saturated targets, see procimg.ccdproc
        if external_flag:
            #flag = (catalog['NIMAFLAGS_ISO'] & 2**2<1) & (catalog['IMAFLAGS_ISO'] & 2**2<1) #& (catalog['IMAFLAGS_ISO']<1)  #(catalog['NIMAFLAGS_ISO']<1)
            flag = (catalog['IMAFLAGS_ISO']<1) & (catalog['NIMAFLAGS_ISO']<1)
        else:
            flag = np.ones_like(catalog['FLAGS'], dtype='bool')
        good_cat = (catalog['FLAGS']<1) & flag
        #& (catalog['CLASS_STAR']>0.9) & (catalog['NIMAFLAGS_ISO']<1)
        good_cat &= catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO']>10
        catalog = catalog[good_cat]
        ra, dec = catalog['ALPHA_J2000'], catalog['DELTA_J2000']
    except:
        if verbose:
            msgs.info('Reading SExtractor catalog failed. Reading photoutils catalog')
        catalog = Table.read(catalogfits)
        good_cat = catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO']>10
        catalog = catalog[good_cat]
        ra, dec = catalog['sky_centroid_icrs.ra'], catalog['sky_centroid_icrs.dec']

    pos = np.zeros((len(ra), 2))
    pos[:,0], pos[:,1] = ra, dec

    ra_cen, dec_cen = np.median(ra), np.median(dec)
    distance = np.sqrt((ra-ra_cen)*np.cos(dec_cen/180.*np.pi)**2 + (dec-dec_cen)**2)
    radius = np.nanmax(distance)*oversize

    # Read/Query a reference catalog
    if (out_refcat is not None) and os.path.exists(out_refcat):
        if verbose:
            msgs.info('Using the existing reference catalog {:} rather than downloading a new one.'.format(out_refcat))
        ref_data = Table.read(out_refcat, format='fits')
        ref_ra_cen, ref_dec_ren =  np.median(ref_data['RA']), np.median(ref_data['DEC'])
        dist_cen = np.sqrt((ref_ra_cen-ra_cen)*np.cos(dec_cen/180.*np.pi)**2 + (ref_dec_ren-dec_cen)**2)
        if dist_cen>0.7*radius:
            if verbose:
                msgs.info('Existing catalog does not fully cover the field, re-download the catalog.')
            ref_data = query.query_standard(ra_cen, dec_cen, catalog=refcatalog, radius=radius)
    else:
        ref_data = query.query_standard(ra_cen, dec_cen, catalog=refcatalog, radius=radius)
    # save the reference catalog to fits
    if (out_refcat is not None) and np.invert(os.path.exists(out_refcat)):
        if verbose:
            msgs.info('Saving the reference catalog to {:}'.format(out_refcat))
        ref_data.write(out_refcat, format='fits', overwrite=True)

    if ref_data is not None:
        # Select high S/N stars
        good_ref = (1.0857/ref_data['{:}_MAG_ERR'.format(primary)]>10)
        if coefficients[1]*coefficients[2] !=0:
            good_ref &= (1.0857/ref_data['{:}_MAG_ERR'.format(secondary)]>10)
        try:
            ref_data = ref_data[good_ref.data]
        except:
            ref_data = ref_data[good_ref]

        #ref_data = query.query_region(ra_cen, dec_cen, catalog=refcatalog, radius=radius)
        #try:
        #    good_ref &= (ref_data['class']==6).data
        #except:
        #    msgs.warn('No point-source selection was applied to the reference catalog')

        ref_ra, ref_dec = ref_data['RA'], ref_data['DEC']
        ref_mag = ref_data['{:}_MAG'.format(primary)] + coefficients[0] + \
                  coefficients[1]*(ref_data['{:}_MAG'.format(primary)]-ref_data['{:}_MAG'.format(secondary)])+ \
                  coefficients[2] * (ref_data['{:}_MAG'.format(primary)] - ref_data['{:}_MAG'.format(secondary)])**2

        ref_pos = np.zeros((len(ref_ra), 2))
        ref_pos[:,0], ref_pos[:,1] = ref_ra, ref_dec

        ## cross-match with 1 arcsec
        dist, ind = crossmatch.crossmatch_angular(pos, ref_pos, max_distance=1.0/3600.)
        matched = np.invert(np.isinf(dist))

        matched_cat_mag = catalog['MAG_AUTO'][matched] - 2.5*np.log10(FLXSCALE*FLASCALE)
        matched_ref_mag = ref_mag[ind[matched]]
        #matched_ref_mag =  ref_data['{:}mag'.format(secondary)][ind[matched]]

        nstar = np.sum(matched)
    else:
        nstar=0

    if nstar==0:
        msgs.warn('No matched standard stars were found')
        return 0., 0., nstar, None
    elif nstar < nstar_min:
            msgs.warn('Only {:} standard stars were found'.format(nstar))
            _, zp, zp_std = stats.sigma_clipped_stats(matched_ref_mag - matched_cat_mag,
                                                      sigma=3, maxiters=20, cenfunc='median', stdfunc='std')
            return zp, zp_std, nstar, catalog[matched]
    else:
        _, zp, zp_std = stats.sigma_clipped_stats(matched_ref_mag - matched_cat_mag,
                                                  sigma=3, maxiters=20, cenfunc='median', stdfunc='std')
        msgs.info('The zeropoint measured from {:} stars for {:} is {:0.3f}+/-{:0.3f}'.format(nstar, catalogfits, zp, zp_std))

        '''
        # No need to measure the color-term here
        from scipy import stats as sp_stats
        from sklearn import linear_model
        if nstar>10:
            # measure the color-term
            this_mag = zp+matched_cat_mag
            yy = this_mag - ref_data['{:}_MAG'.format(primary)][ind[matched]]
            xx = ref_data['{:}_MAG'.format(primary)][ind[matched]] - ref_data['{:}_MAG'.format(secondary)][ind[matched]]
            #_, color_term, color_term_std = stats.sigma_clipped_stats(yy*utils.inverse(xx), sigma=3, maxiters=20, cenfunc='median', stdfunc='std')
            color_term_sp, intercept, r_value, p_value, color_term_sp_std = sp_stats.linregress(xx, yy)
            msgs.info('Color-term coefficient is estimated to be {:}+/-{:} by Scipy'.format(color_term_sp,color_term_sp_std))
            #from scipy.optimize import curve_fit
            #def func(x, a, b):
            #    return b * x + a
            #popt, pcov = curve_fit(func, xx, yy)
            #from sklearn.preprocessing import StandardScaler
            XX = xx.data[..., None]
            ransac = linear_model.RANSACRegressor()
            ransac.fit(XX, yy)
            color_term = ransac.estimator_.coef_[0]
            msgs.info('Color-term coefficient is estimated to be {:} by RANSAC'.format(color_term))
        else:
            color_term = 0.
        '''

        if outqaroot is not None:
            if verbose:
                msgs.info('Make a histogram plot for the zpt')
            zp0 = zp - 7*zp_std
            zp1 = zp + 7*zp_std
            num = plt.hist(matched_ref_mag - matched_cat_mag, bins=np.arange(zp0,zp1,0.05))
            nmax = np.max(num[0]*1.1)
            plt.vlines(zp,0,nmax, linestyles='--', colors='r')
            plt.vlines(zp+zp_std,0,nmax, linestyles=':', colors='r')
            plt.vlines(zp-zp_std,0,nmax, linestyles=':', colors='r')
            plt.ylim(0,nmax)
            plt.xlabel('Zero point',fontsize=14)
            plt.ylabel('# stars used for the calibration',fontsize=14)
            plt.text(zp-6.5*zp_std,0.8*nmax,r'{:0.3f}$\pm${:0.3f}'.format(zp,zp_std),fontsize=14)
            plt.savefig(outqaroot+'_zpt_hist.pdf')
            plt.close()

            if verbose:
                msgs.info('Make a scatter plot for the zpt')
            plt.plot(matched_ref_mag, matched_cat_mag+zp, 'k.')
            plt.plot([matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5],[matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5],'r--')
            plt.xlim(np.nanmin(matched_ref_mag)-0.5,np.nanmax(matched_ref_mag)+0.5)
            plt.ylim(np.nanmin(matched_ref_mag)-0.5,np.nanmax(matched_ref_mag)+0.5)
            plt.xlabel('Reference magnitude',fontsize=14)
            plt.ylabel('Calibrated magnitude',fontsize=14)
            plt.savefig(outqaroot+'_zpt_scatter.pdf')
            plt.close()

            if verbose:
                msgs.info('Make a scatter plot for the coordinate difference')
            ## estimate the coordinates differences
            delta_ra, delta_dec = caloffset.offset(ref_ra[ind[matched]], ref_dec[ind[matched]], ra[matched],
                                                   dec[matched], center=False)
            plt.plot([0.,0.],[delta_dec.min()-0.2,delta_dec.max()+0.2],'r:')
            plt.plot([delta_ra.min()-0.2,delta_ra.max()+0.2],[0.,0.],'r:')
            plt.plot(delta_ra,delta_dec,'k.')
            plt.xlim(delta_ra.min()-0.2,delta_ra.max()+0.2)
            plt.ylim(delta_dec.min()-0.2,delta_dec.max()+0.2)
            plt.xlabel(r'$\Delta$ RA (arcsec)',fontsize=14)
            plt.ylabel(r'$\Delta$ DEC (arcsec)',fontsize=14)
            plt.savefig(outqaroot+'_pos_scatter.pdf')
            plt.close()

    return zp, zp_std, nstar, catalog[matched]

def _cal_chip(cat_fits, sci_fits=None, ref_fits=None, outqa_root=None, ZP=25.0, external_flag=True,
              refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.],
              nstar_min=10, pixscale=None, verbose=True):

    if sci_fits is None:
        sci_fits = cat_fits.replace('.cat','.fits')

    if ref_fits is None:
        ref_fits = cat_fits.replace('.cat','_ref.cat')

    if outqa_root is None:
        outqa_root = cat_fits.replace('.cat','')

    if not os.path.exists(sci_fits):
        msgs.error('{:} was not found, make sure you have an associated image!'.format(sci_fits))
    else:
        msgs.info('Calibrating the zero point for {:}'.format(sci_fits))

    par = fits.open(sci_fits, memmap=False)
    ## ToDo: If we read in FLXSCALE, the zpt would change if you run this code twice
    ##  Maybe just give the input of 1.0/EXPTIME and measure the FLXSCALE use calzpt?
    ##  Thus that the ZPT won't change no matter how many times you run the code.
    try:
        FLXSCALE = 1.0 / par[0].header['EXPTIME']
        #FLXSCALE = par[0].header['FLXSCALE']
    except:
        if verbose:
            msgs.warn('EXPTIME was not found in the FITS Image Header, assuming the image unit is counts/sec.')
        FLXSCALE = 1.0
    try:
        FLASCALE = par[0].header['FLASCALE']
    except:
        if verbose:
            msgs.warn('FLASCALE was not found in the FITS Image Header.')
        FLASCALE = 1.0

    # query a bigger (by a factor of 2 as specified by oversize) radius for safe given that you might have a big dither when re-use this catalog
    zp_this, zp_this_std, nstar, cat_matched = calzpt(cat_fits, refcatalog=refcatalog, primary=primary, secondary=secondary,
                               coefficients=coefficients, FLXSCALE=FLXSCALE, FLASCALE=FLASCALE, external_flag=external_flag,
                               oversize=2.0, out_refcat=ref_fits, outqaroot=outqa_root, verbose=verbose)
    if nstar>nstar_min:
        if verbose:
            msgs.info('Calibrating the zero point of {:} to {:} AB magnitude.'.format(os.path.basename(sci_fits),ZP))
        mag_ext = ZP-zp_this
        FLXSCALE *= 10**(0.4*(mag_ext))
        if mag_ext>0.5:
            msgs.warn('{:} has an extinction of {:0.3f} magnitude, cloudy???'.format(os.path.basename(sci_fits), mag_ext))
        elif mag_ext>0.1:
            msgs.info('{:} has an extinction of {:0.3f} magnitude, good conditions!'.format(os.path.basename(sci_fits), mag_ext))
        else:
            msgs.info('{:} has an extinction of {:0.3f} magnitude, Excellent conditions!'.format(os.path.basename(sci_fits), mag_ext))

        # measure the PSF
        star_table=Table()
        try:
            star_table['x'] = cat_matched['XWIN_IMAGE']
            star_table['y'] = cat_matched['YWIN_IMAGE']
        except:
            star_table['x'] = cat_matched['xcentroid']
            star_table['y'] = cat_matched['ycentroid']
        fwhm, _, _,_ = psf.buildPSF(star_table, sci_fits, size=51, sigclip=5, maxiters=10, norm_radius=2.5,
                               pixscale=pixscale, cenfunc='median', outroot=outqa_root, verbose=verbose)

        # Save the important parameters
        par[0].header['FLXSCALE'] = FLXSCALE
        par[0].header['FLASCALE'] = FLASCALE
        par[0].header['ZP'] = (zp_this, 'Zero point measured from stars')
        par[0].header['ZP_STD'] = (zp_this_std, 'The standard deviration of ZP')
        par[0].header['ZP_NSTAR'] = (nstar, 'The number of stars used for ZP and FWHM')
        par[0].header['FWHM'] = (fwhm, 'FWHM in units of arcsec measured from stars')
        par.writeto(sci_fits, overwrite=True)
    else:
        msgs.warn('The number of stars found for calibration is smaller than nstar_min. skipping the ZPT calibrations.')
        fwhm = 0

    return zp_this, zp_this_std, nstar, fwhm

def cal_chips(cat_fits_list, sci_fits_list=None, ref_fits_list=None, outqa_root_list=None, n_process=4,
              ZP=25.0, external_flag=True, refcatalog='Panstarrs', primary='i', secondary='z',
              coefficients=[0.,0.,0.], nstar_min=10, pixscale=None, verbose=True):

    n_file = len(cat_fits_list)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    if sci_fits_list is not None:
        if len(sci_fits_list) != len(cat_fits_list):
            msgs.error('The length of sci_fits_list should be the same with the length of cat_fits_list.')
    else:
        sci_fits_list = []
        for this_cat in cat_fits_list:
            sci_fits_list.append(this_cat.replace('.cat', '.fits'))

    if ref_fits_list is not None:
        if len(ref_fits_list) != len(cat_fits_list):
            msgs.error('The length of ref_fits_list should be the same with the length of cat_fits_list.')
    else:
        ref_fits_list = []
        for this_cat in cat_fits_list:
            ref_fits_list.append(this_cat.replace('.cat', '_ref.cat'))

    if outqa_root_list is not None:
        if len(outqa_root_list) != len(cat_fits_list):
            msgs.error('The length of outqa_root_list should be the same with the length of cat_fits_list.')
    else:
        outqa_root_list = []
        for this_cat in cat_fits_list:
            outqa_root_list.append(this_cat.replace('.cat',''))

    zp_all, zp_std_all = np.zeros(n_file), np.zeros(n_file)
    nstar_all = np.zeros(n_file)
    fwhm_all = np.zeros(n_file)

    if n_process == 1:
        for ii in range(n_file):
            zp_this, zp_this_std, nstar, fwhm = _cal_chip(cat_fits_list[ii], sci_fits=sci_fits_list[ii],
                        ref_fits=ref_fits_list[ii], outqa_root_list=cat_fits_list[ii],
                        ZP=ZP, external_flag=external_flag, refcatalog=refcatalog, primary=primary, secondary=secondary,
                        coefficients=coefficients, nstar_min=nstar_min, pixscale=pixscale, verbose=verbose)
            zp_all[ii] = zp_this
            zp_std_all[ii] = zp_this_std
            nstar_all[ii] = nstar
            fwhm_all[ii] = fwhm
    else:
        msgs.info('Start parallel processing with n_process={:}'.format(n_process))
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for ii in range(n_file):
            work_queue.put((cat_fits_list[ii], sci_fits_list[ii], ref_fits_list[ii], cat_fits_list[ii]))

        # creating processes
        for w in range(n_process):
            p = Process(target=_cal_chip_worker, args=(work_queue, done_queue), kwargs={
                'ZP': ZP, 'external_flag': external_flag,
                'refcatalog': refcatalog, 'primary': primary, 'secondary': secondary, 'coefficients': coefficients,
                'nstar_min': nstar_min, 'pixscale': pixscale, 'verbose': False})
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

        # print the output
        ii = 0
        while not done_queue.empty():
            zp_this, zp_this_std, nstar, fwhm = done_queue.get()
            zp_all[ii] = zp_this
            zp_std_all[ii] = zp_this_std
            nstar_all[ii] = nstar
            fwhm_all[ii] = fwhm
            ii +=1
    return zp_all, zp_std_all, nstar_all, fwhm_all

def _cal_chip_worker(work_queue, done_queue, ZP=25.0, external_flag=True, refcatalog='Panstarrs',
                     primary='i', secondary='z', coefficients=[0.,0.,0.], nstar_min=10, pixscale=None, verbose=False):

    """Multiprocessing worker for cal_chips."""
    while not work_queue.empty():
        cat_fits, sci_fits, ref_fits, outqa_root = work_queue.get()
        zp_this, zp_this_std, nstar, fwhm = _cal_chip(cat_fits, sci_fits=sci_fits, ref_fits=ref_fits, outqa_root=outqa_root,
                ZP=ZP, external_flag=external_flag, refcatalog=refcatalog, primary=primary, secondary=secondary,
                coefficients=coefficients, nstar_min=nstar_min, pixscale=pixscale, verbose=verbose)

        done_queue.put((zp_this, zp_this_std, nstar, fwhm))


'''
# Moved this part into sciproc
def negativestar(sci_fits_list, wht_fits_list, flag_fits_list, sigma=5, maxiters=3,
                 brightstar_nsigma=5, maskbrightstar_method='sextractor', sextractor_task='sex'):

    for i in range(len(sci_fits_list)):
        #ToDo: parallel this
        msgs.info('Masking negative stars for {:}'.format(sci_fits_list[i]))
        header, data, _ = io.load_fits(sci_fits_list[i])
        header_wht, weight, _ = io.load_fits(wht_fits_list[i])
        header_flag, flag, _ = io.load_fits(flag_fits_list[i])
        bpm = flag>0
        starmask = mask_bright_star(0-data, mask=bpm, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigma,
                                    back_maxiters=maxiters, method=maskbrightstar_method, task=sextractor_task)

        weight[starmask] = 0.
        flag[starmask] = 2**8
        io.save_fits(wht_fits_list[i], weight, header_wht, 'ScienceImage', overwrite=True)
        io.save_fits(flag_fits_list[i], flag, header_flag, 'ScienceImage', overwrite=True)
'''

### The following functions are old code that do not support parallel processing.
def cal_chips_old(cat_fits_list, sci_fits_list=None, ref_fits_list=None, outqa_root_list=None, ZP=25.0, external_flag=True,
                 refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.], nstar_min=10, pixscale=None):


    zp_all, zp_std_all = np.zeros(len(cat_fits_list)), np.zeros(len(cat_fits_list))
    nstar_all = np.zeros(len(cat_fits_list))
    fwhm_all = np.zeros(len(cat_fits_list))
    for ii, this_cat in enumerate(cat_fits_list):
        if sci_fits_list is None:
            this_sci_fits = this_cat.replace('.cat','.fits')
        else:
            this_sci_fits = sci_fits_list[ii]
        if ref_fits_list is None:
            this_ref_name = this_cat.replace('.cat','_ref.cat')
        else:
            this_ref_name = ref_fits_list[ii]
        if outqa_root_list is None:
            this_qa_root = this_cat.replace('.cat','')
        else:
            this_qa_root = outqa_root_list[ii]

        if not os.path.exists(this_sci_fits):
            msgs.error('{:} was not found, make sure you have an associated image!'.format(this_sci_fits))
        else:
            msgs.info('Calibrating the zero point for {:}'.format(this_sci_fits))

        par = fits.open(this_sci_fits, memmap=False)
        ## ToDo: If we read in FLXSCALE, the zpt would change if you run this code twice
        ##  Maybe just give the input of 1.0/EXPTIME and measure the FLXSCALE use calzpt?
        ##  Thus that the ZPT won't change no matter how many times you run the code.
        try:
            FLXSCALE = 1.0 / par[0].header['EXPTIME']
            #FLXSCALE = par[0].header['FLXSCALE']
        except:
            msgs.warn('EXPTIME was not found in the FITS Image Header, assuming the image unit is counts/sec.')
            FLXSCALE = 1.0
        try:
            FLASCALE = par[0].header['FLASCALE']
        except:
            msgs.warn('FLASCALE was not found in the FITS Image Header.')
            FLASCALE = 1.0

        # query a bigger (by a factor of 2 as specified by oversize) radius for safe given that you might have a big dither when re-use this catalog
        zp_this, zp_this_std, nstar, cat_matched = calzpt(this_cat, refcatalog=refcatalog, primary=primary, secondary=secondary,
                                   coefficients=coefficients, FLXSCALE=FLXSCALE, FLASCALE=FLASCALE, external_flag=external_flag,
                                   oversize=2.0, out_refcat=this_ref_name, outqaroot=this_qa_root)
        if nstar>nstar_min:
            msgs.info('Calibrating the zero point of {:} to {:} AB magnitude.'.format(os.path.basename(this_sci_fits),ZP))
            mag_ext = ZP-zp_this
            FLXSCALE *= 10**(0.4*(mag_ext))
            if mag_ext>0.5:
                msgs.warn('{:} has an extinction of {:0.3f} magnitude, cloudy???'.format(os.path.basename(this_sci_fits), mag_ext))
            elif mag_ext>0.1:
                msgs.info('{:} has an extinction of {:0.3f} magnitude, good conditions!'.format(os.path.basename(this_sci_fits), mag_ext))
            else:
                msgs.info('{:} has an extinction of {:0.3f} magnitude, Excellent conditions!'.format(os.path.basename(this_sci_fits), mag_ext))

            # measure the PSF
            star_table=Table()
            try:
                star_table['x'] = cat_matched['XWIN_IMAGE']
                star_table['y'] = cat_matched['YWIN_IMAGE']
            except:
                star_table['x'] = cat_matched['xcentroid']
                star_table['y'] = cat_matched['ycentroid']
            fwhm, _, _,_ = psf.buildPSF(star_table, this_sci_fits, size=51, sigclip=5, maxiters=10, norm_radius=2.5,
                                   pixscale=pixscale, cenfunc='median', outroot=this_qa_root)

            # Save the important parameters
            par[0].header['FLXSCALE'] = FLXSCALE
            par[0].header['FLASCALE'] = FLASCALE
            par[0].header['ZP'] = (zp_this, 'Zero point measured from stars')
            par[0].header['ZP_STD'] = (zp_this_std, 'The standard deviration of ZP')
            par[0].header['ZP_NSTAR'] = (nstar, 'The number of stars used for ZP and FWHM')
            par[0].header['FWHM'] = (fwhm, 'FWHM in units of arcsec measured from stars')
            par.writeto(this_sci_fits, overwrite=True)
        else:
            msgs.warn('The number of stars found for calibration is smaller than nstar_min. skipping the ZPT calibrations.')
            fwhm = 0
        zp_all[ii] = zp_this
        zp_std_all[ii] = zp_this_std
        nstar_all[ii] = nstar
        fwhm_all[ii] = fwhm

    return zp_all, zp_std_all, nstar_all, fwhm_all

def astrometric_old(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, science_path='./',qa_path='./',
                task='sex',detect_thresh=5.0, analysis_thresh=5.0, detect_minarea=5, crossid_radius=1.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', astrefmag_limits=None,
                position_maxerr=1.0, distort_degrees=3, pixscale_maxerr=1.1, posangle_maxerr=10.0,
                stability_type='INSTRUMENT', mosaic_type='LOOSE', weight_type='MAP_WEIGHT',
                skip_swarp_align=False, solve_photom_scamp=False, scamp_second_pass=False,delete=False, log=True):
    # ToDo: parallel this. Maybe parallel sexall, scampall, and swarpall
    # configuration for the swarp run
    # Note that I would apply the gain correction before doing the astrometric calibration, so I set Gain to 1.0
    # resample science image
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                   "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT": 1.0,
                   "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                   "RESAMPLING_TYPE": 'NEAREST',
                   # I would always set this to NEAREST for individual exposures to avoid interpolation
                   "HEADER_SUFFIX": "_cat.head"}
    # configuration for swarp the flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'

    if skip_swarp_align:
        msgs.info('Skipping the first alignment step with Swarp')
        for i in range(len(sci_fits_list)):
            os.system('cp {:} {:}'.format(sci_fits_list[i],
                                          sci_fits_list[i].replace('.fits', '.resamp.fits')))
            os.system('cp {:} {:}'.format(sci_fits_list[i].replace('.fits', '.weight.fits'),
                                          sci_fits_list[i].replace('.fits', '.resamp.weight.fits')))
            os.system('cp {:} {:}'.format(flag_fits_list[i],
                                          flag_fits_list[i].replace('.fits', '.resamp.fits')))
    else:
        ## This step is basically align the image to N to the up and E to the left.
        ## It is important if your image has a ~180 degree from the nominal orientation.
        msgs.info('Running Swarp for the first pass to align the science image.')
        swarp.swarpall(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                       coaddroot=None, delete=delete, log=log)
        # resample flag image
        msgs.info('Running Swarp for the first pass to align the flag image.')
        swarp.swarpall(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                       coaddroot=None, delete=delete, log=False)

    ## remove useless data and change flag image type to int32
    sci_fits_list_resample = []
    wht_fits_list_resample = []
    flag_fits_list_resample = []
    cat_fits_list_resample = []
    for i in range(len(sci_fits_list)):
        sci_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.fits'))
        wht_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.weight.fits'))
        flag_fits_list_resample.append(flag_fits_list[i].replace('.fits', '.resamp.fits'))
        cat_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp_cat.fits'))
        if not skip_swarp_align:
            par = fits.open(flag_fits_list[i].replace('.fits', '.resamp.fits'), memmap=False)
            par[0].data = par[0].data.astype('int32')  # change the dtype to be int32
            par[0].writeto(flag_fits_list[i].replace('.fits', '.resamp.fits'), overwrite=True)
            del par[0].data
            par.close()
            gc.collect()

            os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))

    # configuration for the first SExtractor run
    sexconfig0 = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_TYPE": "FITS_LDAC",
                  "DETECT_THRESH": detect_thresh,
                  "ANALYSIS_THRESH": analysis_thresh,
                  "DETECT_MINAREA": detect_minarea}
    sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                  'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                  'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER',
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    msgs.info('Running SExtractor for the first pass to extract catalog used for SCAMP.')
    sex.sexall(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    #for ii in range(len(cat_fits_list_resample)):
    #    this_cat = cat_fits_list_resample[ii]
    #    par = fits.open(this_cat)
    #    this_clean = (par[2].data['NIMAFLAGS_ISO']<1) & (par[2].data['IMAFLAGS_ISO']<1) & (par[2].data['FLAGS']<1)
    #    par[2].data = par[2].data[this_clean]
    #    par.writeto(this_cat,overwrite=True)

    # configuration for the scamp run
    if solve_photom_scamp:
        SOLVE_PHOTOM='Y'
    else:
        SOLVE_PHOTOM='N'

    if astrefmag_limits is not None:
        ASTREFMAG_LIMITS = '{:},{:}'.format(astrefmag_limits[0],astrefmag_limits[1])
    else:
        ASTREFMAG_LIMITS = '-99.0,99.0'

    scampconfig = {"CROSSID_RADIUS": crossid_radius,
                    "ASTREF_CATALOG": astref_catalog,
                    "ASTREF_BAND": astref_band,
                    "ASTREFMAG_LIMITS":ASTREFMAG_LIMITS,
                    "POSITION_MAXERR": position_maxerr,
                    "PIXSCALE_MAXERR": pixscale_maxerr,
                    "POSANGLE_MAXERR": posangle_maxerr,
                    "STABILITY_TYPE": stability_type,
                    "MOSAIC_TYPE": mosaic_type,
                    "SOLVE_PHOTOM": SOLVE_PHOTOM,
                    "DISTORT_DEGREES":distort_degrees,
                    "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                    "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
    if scamp_second_pass:
        # first run with distort_degrees of 1
        scampconfig1 = scampconfig.copy()
        scampconfig1['DISTORT_DEGREES'] = np.max([1,distort_degrees-2])
        msgs.info('Running the first pass SCAMP with DISTORT_DEGREES of {:}'.format(scampconfig1['DISTORT_DEGREES']))
        scamp.scampall(cat_fits_list_resample, config=scampconfig1, workdir=science_path, QAdir=qa_path,
                       defaultconfig='pyphot', delete=delete, log=log)
        ## we can make the maximum errors to be smaller for the second pass.
        scampconfig['POSITION_MAXERR'] = np.min([1.0,position_maxerr])
        scampconfig['PIXSCALE_MAXERR'] = np.min([1.2,pixscale_maxerr])
        scampconfig['POSANGLE_MAXERR'] = np.min([5.0,posangle_maxerr])
        # copy the .head to .ahead
        for i in range(len(sci_fits_list_resample)):
            os.system('mv {:} {:}'.format(cat_fits_list_resample[i].replace('.fits', '.head'),
                                          cat_fits_list_resample[i].replace('.fits', '.ahead')))
    # run the final scamp
    msgs.info('Running the final pass of SCAMP with DISTORT_DEGREES of {:}'.format(distort_degrees))
    scamp.scampall(cat_fits_list_resample, config=scampconfig, workdir=science_path, QAdir=qa_path,
                   defaultconfig='pyphot', delete=delete, log=log)

    ## copy the .head for flag images
    for i in range(len(sci_fits_list_resample)):
        os.system('cp {:} {:}'.format(sci_fits_list_resample[i].replace('.fits', '_cat.head'),
                                      flag_fits_list_resample[i].replace('.fits', '_cat.head')))

    ## configuration for the second swarp run
    swarpconfig['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    # resample the science image
    msgs.info('Running Swarp for the second pass to align the science image.')
    swarp.swarpall(sci_fits_list_resample, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)
    # resample the flag image
    swarpconfig_flag['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    msgs.info('Running Swarp for the second pass to align the flag image.')
    swarp.swarpall(flag_fits_list_resample, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=False)

    # delete unnecessary weight maps and head
    for i in range(len(sci_fits_list)):
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp_cat.fits')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp_cat.head')))
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp_cat.head')))

    if scamp_second_pass:
        for i in range(len(sci_fits_list)):
            os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp_cat.ahead')))

    # Update fluxscale
    if solve_photom_scamp:
        msgs.info('The FLXSCALE was solved with scamp.')
    else:
        for i in range(len(sci_fits_list)):
            msgs.info('Solving the FLXSCALE for {:} with 1/EXPTIME.'.format(os.path.basename(sci_fits_list_resample[i])))
            par = fits.open(sci_fits_list_resample[i], memmap=False)
            # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
            # in order to get the correct flag when doing the coadd with swarp in the later step
            par[0].header['FLXSCALE'] = utils.inverse(par[0].header['EXPTIME'])
            par[0].writeto(sci_fits_list_resample[i], overwrite=True)
            del par[0].data
            par.close()

    # change flag image type to int32, flux scale to 1.0
    for i in range(len(sci_fits_list)):
        msgs.info('Saving FLAG image {:} to int32.'.format(os.path.basename(flag_fits_list_resample[i])))
        par = fits.open(flag_fits_list_resample[i], memmap=False)
        # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
        # in order to get the correct flag when doing the coadd with swarp in the later step
        par[0].header['EXPTIME'] = 1.0
        par[0].header['FLXSCALE'] = 1.0
        par[0].header['FLASCALE'] = 1.0
        par[0].data = par[0].data.astype('int32')
        par[0].writeto(flag_fits_list_resample[i], overwrite=True)
        del par[0].data
        par.close()
        gc.collect()


    ## Run SExtractor for the resampled images. The catalogs will be used for calibrating individual chips.
    msgs.info('Running SExtractor for the second pass to extract catalog for resampled images.')
    sex.sexall(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample, cat_fits_list_resample
