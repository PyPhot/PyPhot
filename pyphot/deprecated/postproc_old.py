

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
            FLXSCALE = 1.0 * utils.inverse(par[0].header['EXPTIME'])
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

def astrometric_mosaic(sci_fits_list, wht_fits_list, var_fits_list, flag_fits_list, pixscale, n_process=4, science_path='./', qa_path='./',
                task='sex', detect_thresh=5.0, analysis_thresh=5.0, detect_minarea=5, crossid_radius=1.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', astrefmag_limits=None,
                position_maxerr=1.0, distort_degrees=3, pixscale_maxerr=1.1, posangle_maxerr=10.0,
                stability_type='INSTRUMENT', mosaic_type='LOOSE', weight_type='MAP_WEIGHT', conv='sex',
                group=True, skip_swarp_align=False, scamp_second_pass=False, solve_photom_scamp=False,
                delete=False, log=True, verbose=True):

    # This function has a better structure and faster scamp than astrometric but need to be finished.
    n_file = len(sci_fits_list)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    ## Prepare output list
    cat_fits_list = []
    for ifits in sci_fits_list:
        cat_fits = ifits.replace('.fits','_cat.fits')
        cat_fits_list.append(cat_fits)

    ndet = len(fits.open(sci_fits_list[0])) - 1
    sci_fits_list_resample = []
    wht_fits_list_resample = []
    var_fits_list_resample = []
    flag_fits_list_resample = []
    cat_fits_list_resample = []
    for ifits in sci_fits_list:
        for idet in range(1, ndet+1):
            sci_fits_resample = ifits.replace('.fits','.{:04d}.resamp.fits'.format(idet))
            wht_fits_resample = ifits.replace('.fits','.{:04d}.resamp.weight.fits'.format(idet))
            var_fits_resample = ifits.replace('.fits','.var.{:04d}.resamp.fits'.format(idet))
            flag_fits_resample = ifits.replace('sci.fits','flag.{:04d}.resamp.fits'.format(idet))
            cat_fits_resample = ifits.replace('.fits','.{:04d}.resamp_cat.fits'.format(idet))
            sci_fits_list_resample.append(sci_fits_resample)
            var_fits_list_resample.append(var_fits_resample)
            wht_fits_list_resample.append(wht_fits_resample)
            flag_fits_list_resample.append(flag_fits_resample)
            cat_fits_list_resample.append(cat_fits_resample)

    ## step one: extract catalogs from given images
    # configuration for SExtractor run
    sexconfig0 = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "MAP_WEIGHT", "CATALOG_TYPE": "FITS_LDAC",
                  "DETECT_THRESH": detect_thresh,
                  "ANALYSIS_THRESH": analysis_thresh,
                  "DETECT_MINAREA": detect_minarea}
    sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                  'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                  'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER',
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    if verbose:
        msgs.info('Running SExtractor for the first pass to extract catalog used for SCAMP.')
    sex.run_sex(sci_fits_list, flag_image_list=flag_fits_list, weight_image_list=wht_fits_list,
                n_process=n_process, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
                defaultconfig='pyphot', conv=conv, nnw=None, dual=False, delete=delete, log=log, verbose=verbose)

    ## step two: scamp
    # configuration for the scamp run
    if solve_photom_scamp:
        SOLVE_PHOTOM='Y'
    else:
        SOLVE_PHOTOM='N'

    if astrefmag_limits is not None:
        ASTREFMAG_LIMITS = '{:},{:}'.format(astrefmag_limits[0],astrefmag_limits[1])
    else:
        ASTREFMAG_LIMITS = '-99.0,99.0'

    mosaic_type = 'FIX_FOCALPLANE'
    group = True
    delete=False
    log=True
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

    scamp.run_scamp(cat_fits_list, config=scampconfig, workdir=science_path, QAdir=qa_path, n_process=n_process,
                    defaultconfig='pyphot', group=group, delete=delete, log=log, verbose=verbose)

    ## step three: swarp
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                   "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT": 1.0,
                   "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                   "RESAMPLING_TYPE": 'NEAREST',
                   # I would always set this to NEAREST for individual exposures to avoid interpolation
                   "HEADER_SUFFIX": "_cat.head"}

    # resample science image
    if verbose:
        msgs.info('Running Swarp to align the science image.')
    swarp.run_swarp(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                    n_process=n_process, delete=delete, log=log, verbose=verbose)

    # resample flag image
    if verbose:
        msgs.info('Running Swarp to align the flag image.')
    # copy the .head for flag images
    # configuration for swarp the flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    for ii, icat in enumerate(cat_fits_list):
        os.system('cp {:} {:}'.format(icat.replace('.fits', '.head'),
                                      flag_fits_list[ii].replace('.fits', '_cat.head')))
    swarp.run_swarp(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                    n_process=n_process, delete=delete, log=log, verbose=verbose)

    # resample var image
    if verbose:
        msgs.info('Running Swarp to align the flag image.')
    # copy the .head for flag images
    # configuration for swarp the var image
    swarpconfig_var = swarpconfig.copy()
    swarpconfig_var['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_var['COMBINE_TYPE'] = 'SUM'
    for ii, icat in enumerate(cat_fits_list):
        os.system('cp {:} {:}'.format(icat.replace('.fits', '.head'),
                                      var_fits_list[ii].replace('.fits', '_cat.head')))
    swarp.run_swarp(var_fits_list, config=swarpconfig_var, workdir=science_path, defaultconfig='pyphot',
                    n_process=n_process, delete=delete, log=log, verbose=verbose)

    # remove useless data and change flag image type
    # change flag image type and delte unnecessary files
    for ii, sci_fits_resample in enumerate(sci_fits_list_resample):
        var_fits_resample = var_fits_list_resample[ii]
        flag_fits_resample = flag_fits_list_resample[ii]
        # delete unnecessary files
        os.system('rm {:}'.format(var_fits_resample.replace('.fits', '.weight.fits')))
        os.system('rm {:}'.format(flag_fits_resample.replace('.fits', '.weight.fits')))

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

    ## step four: Run SExtractor on the resampled images
    ## Run SExtractor for the resampled images. The catalogs will be used for calibrating individual chips.
    if verbose:
        msgs.info('Running SExtractor on the resampled images.')
    sex.run_sex(sci_fits_list_resample, flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample,
                n_process=n_process, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
                defaultconfig='pyphot', conv=conv, nnw=None, dual=False,
                delete=delete, log=log, verbose=verbose)

    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample, cat_fits_list_resample



def astrometric_old(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, n_process=4, science_path='./', qa_path='./',
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

def astrometric_old_old(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, science_path='./',qa_path='./',
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
        swarp.run_swarp(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                        coaddroot=None, delete=delete, log=log)
        # resample flag image
        msgs.info('Running Swarp for the first pass to align the flag image.')
        swarp.run_swarp(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
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
    sex.run_sex(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
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
        scamp.run_scamp(cat_fits_list_resample, config=scampconfig1, workdir=science_path, QAdir=qa_path,
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
    scamp.run_scamp(cat_fits_list_resample, config=scampconfig, workdir=science_path, QAdir=qa_path,
                    defaultconfig='pyphot', delete=delete, log=log)

    ## copy the .head for flag images
    for i in range(len(sci_fits_list_resample)):
        os.system('cp {:} {:}'.format(sci_fits_list_resample[i].replace('.fits', '_cat.head'),
                                      flag_fits_list_resample[i].replace('.fits', '_cat.head')))

    ## configuration for the second swarp run
    swarpconfig['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    # resample the science image
    msgs.info('Running Swarp for the second pass to align the science image.')
    swarp.run_swarp(sci_fits_list_resample, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                    coaddroot=None, delete=delete, log=log)
    # resample the flag image
    swarpconfig_flag['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    msgs.info('Running Swarp for the second pass to align the flag image.')
    swarp.run_swarp(flag_fits_list_resample, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
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
    sex.run_sex(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
                defaultconfig='pyphot', conv='sex', nnw=None, dual=False, delete=delete, log=log,
                flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample, cat_fits_list_resample
