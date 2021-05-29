import os
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import stats
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord

from photutils import detect_sources
from photutils import deblend_sources
from photutils import source_properties
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils.utils import calc_total_error
from photutils import StdBackgroundRMS, MADStdBackgroundRMS, BiweightScaleBackgroundRMS
from photutils import Background2D, MeanBackground, MedianBackground, SExtractorBackground
from photutils import MMMBackground, BiweightLocationBackground, ModeEstimatorBackground

from pyphot import msgs, io, utils
from pyphot import sex, scamp, swarp
from pyphot import query, crossmatch

def defringing(sci_fits_list, masterfringeimg):

    ## ToDo: matching the amplitude of friging rather than scale with exposure time.
    for i in range(len(sci_fits_list)):
        header, data, _ = io.load_fits(sci_fits_list[i])
        if 'DEFRING' in header.keys():
            msgs.info('The De-fringed image {:} exists, skipping...'.format(sci_fits_list[i]))
        else:
            data -= masterfringeimg * header['EXPTIME']
            header['DEFRING'] = ('TRUE', 'De-Fringing is done?')
            io.save_fits(sci_fits_list[i], data, header, 'ScienceImage', overwrite=True)
            msgs.info('De-fringed science image {:} saved'.format(sci_fits_list[i]))


def astrometric(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, science_path='./',qa_path='./',
                task='sex',detect_thresh=3.0, analysis_thresh=3.0, detect_minarea=5, crossid_radius=1.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', position_maxerr=1.0, distort_degrees=3,
                pixscale_maxerr=1.1, posangle_maxerr=10.0, stability_type='INSTRUMENT', mosaic_type='LOOSE',
                weight_type='MAP_WEIGHT', solve_photom_scamp=False, scamp_second_pass=False, delete=False, log=True):

    ## This step is basically align the image to N to the up and E to the left.
    ## this step is important if your image have a very different origination from the regular one.
    # configuration for the first swarp run
    # Note that I would apply the gain correction before doing the astrometric calibration, so I set Gain to 1.0
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                   "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT":1.0,
                   "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                   "RESAMPLING_TYPE": 'NEAREST', #I would always set this to NEAREST for individual exposures to avoid interpolation
                   "HEADER_SUFFIX":"_cat.head"}
    # resample science image
    msgs.info('Running Swarp for the first pass to align the science image.')
    swarp.swarpall(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)

    # resample flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    msgs.info('Running Swarp for the first pass to align the flag image.')
    swarp.swarpall(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=False)

    ## remove useless data and change flag image type to int32
    sci_fits_list_resample = []
    wht_fits_list_resample = []
    flag_fits_list_resample = []
    cat_fits_list_resample = []
    for i in range(len(sci_fits_list)):
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))
        sci_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.fits'))
        wht_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.weight.fits'))
        flag_fits_list_resample.append(flag_fits_list[i].replace('.fits', '.resamp.fits'))
        cat_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp_cat.fits'))
        par = fits.open(flag_fits_list[i].replace('.fits', '.resamp.fits'))
        par[0].data = par[0].data.astype('int32') # change the dtype to be int32
        par[0].writeto(flag_fits_list[i].replace('.fits', '.resamp.fits'), overwrite=True)

    # configuration for the first SExtractor run
    sexconfig0 = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                  "CATALOG_TYPE": "FITS_LDAC",
                  "DETECT_THRESH": detect_thresh,
                  "ANALYSIS_THRESH": analysis_thresh,
                  "DETECT_MINAREA": detect_minarea}
    sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                  'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                  'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER',
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    msgs.info('Running SExtractor for the first pass to extract catalog used for SCAMP.')
    sex.sexall(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex995', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    # configuration for the scamp run
    if solve_photom_scamp:
        SOLVE_PHOTOM='Y'
    else:
        SOLVE_PHOTOM='N'
    scampconfig = {"CROSSID_RADIUS": crossid_radius,
                    "ASTREF_CATALOG": astref_catalog,
                    "ASTREF_BAND": astref_band,
                    "POSITION_MAXERR": position_maxerr,
                    "PIXSCALE_MAXERR": pixscale_maxerr,
                    "POSANGLE_MAXERR": posangle_maxerr, # ToDo: add to parset
                    "STABILITY_TYPE": stability_type,
                    "MOSAIC_TYPE": mosaic_type,
                    "SOLVE_PHOTOM": SOLVE_PHOTOM,
                    "DISTORT_DEGREES":distort_degrees,
                    "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                    "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
    if scamp_second_pass:
        # first run with distort_degrees of 1
        msgs.info('Running the first pass SCAMP with DISTORT_DEGREES of 1')
        scampconfig1 = scampconfig.copy()
        scampconfig1['DISTORT_DEGREES'] = 1
        scamp.scampall(cat_fits_list_resample, config=scampconfig1, workdir=science_path, QAdir=qa_path,
                       defaultconfig='pyphot', delete=delete, log=log)
        # copy the .head to .ahead
        for i in range(len(sci_fits_list_resample)):
            os.system('mv {:} {:}'.format(cat_fits_list_resample[i].replace('.fits', '.head'),
                                          cat_fits_list_resample[i].replace('.fits', '.ahead')))
    # run the final scamp
    msgs.info('Running the final pass of SCAMP')
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
        msgs.info('The FLXSCALE was solved with 1/EXPTIME.')
        for i in range(len(sci_fits_list)):
            par = fits.open(sci_fits_list_resample[i])
            # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
            # in order to get the correct flag when doing the coadd with swarp in the later step
            par[0].header['FLXSCALE'] = utils.inverse(par[0].header['EXPTIME'])
            par[0].writeto(sci_fits_list_resample[i], overwrite=True)

    # change flag image type to int32, flux scale to 1.0
    for i in range(len(sci_fits_list)):
        par = fits.open(flag_fits_list_resample[i])
        # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
        # in order to get the correct flag when doing the coadd with swarp in the later step
        par[0].header['EXPTIME'] = 1.0
        par[0].header['FLXSCALE'] = 1.0
        par[0].header['FLASCALE'] = 1.0
        par[0].data = par[0].data.astype('int32')
        par[0].writeto(flag_fits_list_resample[i], overwrite=True)

    ## Run SExtractor for the resampled images. The catalogs will be used for calibrating individual chips.
    msgs.info('Running SExtractor for the second pass to extract catalog for resampled images.')
    sex.sexall(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex995', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)


    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample, cat_fits_list_resample


def coadd(scifiles, flagfiles, coaddroot, pixscale, science_path, coadddir, weight_type='MAP_WEIGHT',
          rescale_weights=False, combine_type='median', clip_ampfrac=0.3, clip_sigma=4.0, blank_badpixels=False,
          subtract_back= False, back_type='AUTO', back_default=0.0, back_size=200, back_filtersize=3,
          back_filtthresh=0.0, resampling_type='LANCZOS3', delete=True, log=True):

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
                   "WEIGHT_TYPE": weight_type,"RECALE_WEIGHTS": rescale,"BLANK_BADPIXELS":blank,
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

    # change flag image type to int32
    par = fits.open(coadd_flag_file)
    par[0].data = np.round(par[0].data).astype('int32')
    par[0].writeto(coadd_flag_file, overwrite=True)

    return coadd_file, coadd_wht_file, coadd_flag_file


def detect(data, wcs_info=None, rmsmap=None, bkgmap=None, mask=None, effective_gain=None, nsigma=2., npixels=5, fwhm=5,
           nlevels=32, contrast=0.001, back_nsigma=3, back_maxiters=10, back_type='median', back_rms_type='std',
           back_size=(200, 200), back_filter_size=(3, 3), morp_filter=False,
           phot_apertures=[1.0,2.0,3.0,4.0,5.0], return_seg_only=False):
    '''
        Identify cosmic rays using the L.A.Cosmic algorithm
    U{http://www.astro.yale.edu/dokkum/lacosmic/}
    (article : U{http://arxiv.org/abs/astro-ph/0108003})
    This routine is mostly courtesy of Malte Tewes

    Args:
        data:
        wcs_info:
        rmsmap:
        bkgmap:
        mask:
        effective_gain (float or 2D array): should be a 2D map of exposure time
        nsigma (int or float): how many sigma of your detection
        npixels:  Let’s find sources that have 5 connected pixels that are each greater than the
                  corresponding pixel-wise threshold level defined above  (i.e., 2 sigma per pixel above the background noise)
        fwhm (int or float): seeing in units of pixel
        nlevels:
        contrast:
        back_nsigma:
        back_maxiters:
        back_type:
        back_rms_type:
        back_box_size:
        back_filter_size:
        morp_filter (bool): whether you want to use the kernel filter when measuring morphology and centroid
                            If set true, it should be similar with SExtractor. False gives a better morphology
    Returns:
        astropy Table
    '''

    if mask is None:
        mask = np.isinf(data)

    if effective_gain is None:
        effective_gain = 1.0

    sigma_clip = SigmaClip(sigma=back_nsigma, maxiters=back_maxiters)

    if back_type.lower() == 'median':
        bkg_estimator = MedianBackground()
    elif back_type.lower() == 'mean':
        bkg_estimator = MeanBackground()
    elif back_type.lower() == 'sextractor':
        bkg_estimator = SExtractorBackground()
    elif back_type.lower() == 'mmm':
        bkg_estimator = MMMBackground()
    elif back_type.lower() == 'biweight':
        bkg_estimator = BiweightLocationBackground()
    elif back_type.lower() == 'mode':
        bkg_estimator = ModeEstimatorBackground()
    else:
        msgs.warn('{:}Background is not found, using MedianBackground Instead.'.format(back_type))
        bkg_estimator = MedianBackground()

    if back_rms_type.lower() == 'std':
        bkgrms_estimator = StdBackgroundRMS()
    elif back_rms_type.lower() == 'mad':
        bkgrms_estimator = MADStdBackgroundRMS()
    elif back_rms_type.lower() == 'biweight':
        bkgrms_estimator = BiweightScaleBackgroundRMS()

    if (rmsmap is None) or (bkgmap is None):
        bkg = Background2D(data, back_size, mask=mask, filter_size=back_filter_size, sigma_clip=sigma_clip,
                           bkg_estimator=bkg_estimator, bkgrms_estimator=bkgrms_estimator)
        if rmsmap is None:
            rmsmap = bkg.background_rms
        if bkgmap is None:
            bkgmap = bkg.background

    threshold = bkgmap + nsigma * rmsmap
    error = calc_total_error(data,rmsmap, effective_gain)

    ## Build a Gaussian kernel
    sigma = fwhm * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()

    ## Do the detection using the Image Segmentation technique
    ## The return is a SegmentationImage
    segm = detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)

    if return_seg_only:
        return segm

    # Source Deblending
    segm_deblend = deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel,
                                   nlevels=nlevels, contrast=contrast)

    # Check the Seg image
    '''
    import matplotlib.pyplot as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    image_norm = ImageNormalize(stretch=SqrtStretch())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    ax1.imshow(data, origin='lower', cmap='Greys_r', norm=image_norm)
    ax1.set_title('Data')
    cmap = segm.make_cmap(random_state=123)
    ax2.imshow(segm, origin='lower', cmap=cmap, interpolation='nearest')
    ax2.set_title('Segmentation Image')
    '''

    ## measure the source properties
    if morp_filter:
        cat = source_properties(data, segm_deblend, mask=mask, background=bkgmap, error=error,
                                filter_kernel=kernel, wcs=wcs_info)
    else:
        cat = source_properties(data, segm_deblend, mask=mask, background=bkgmap, error=error, wcs=wcs_info)
    tbl = cat.to_table()
    tbl = tbl[np.invert(np.isnan(tbl['xcentroid']))] # remove sources with nan positions
    tbl['MAG_AUTO'] = -2.5*np.log10(tbl['source_sum'])
    tbl['MAGERR_AUTO'] = 2.5/np.log(10)*tbl['source_sum_err']/tbl['source_sum']
    tbl['FLUX_AUTO'] = tbl['source_sum']
    tbl['FLUXERR_AUTO'] = tbl['source_sum_err']

    ## Perform Aperture photometry
    msgs.info('Performing Aperture photometry')
    positions = tbl['sky_centroid']
    apertures = [SkyCircularAperture(positions, r=d/2*u.arcsec) for d in phot_apertures]
    tbl_aper = aperture_photometry(data, apertures, error=error, mask=mask, method='exact', wcs=wcs_info)
    flux_aper = np.zeros((len(tbl_aper), np.size(phot_apertures)))
    fluxerr_aper = np.zeros_like(flux_aper)
    mag_aper = np.zeros_like(flux_aper)
    magerr_aper = np.zeros_like(flux_aper)
    for ii in range(np.size(phot_apertures)):
        flux_aper[:,ii] = tbl_aper['aperture_sum_{:d}'.format(ii)]
        fluxerr_aper[:,ii] = tbl_aper['aperture_sum_err_{:d}'.format(ii)]
        mag_aper[:,ii] =  -2.5*np.log10(tbl_aper['aperture_sum_{:d}'.format(ii)])
        magerr_aper[:,ii] = 2.5/np.log(10)*tbl_aper['aperture_sum_err_{:d}'.format(ii)]/tbl_aper['aperture_sum_{:d}'.format(ii)]
    tbl['MAG_APER'] = mag_aper
    tbl['MAGERR_APER'] = magerr_aper
    tbl['FLUX_APER'] = flux_aper
    tbl['FLUXERR_APER'] = fluxerr_aper

    ## ToDo: Add PSF photometry

    return tbl, rmsmap, bkgmap

def mask_bright_star(data, mask=None, brightstar_nsigma=3, back_nsigma=3, back_maxiters=10, npixels=3, fwhm=5,
                     method='sextractor', task='sex'):

    if mask is not None:
        data[mask] = 0. # zero out bad pixels

    if method.lower()=='photoutils':
        msgs.info('Masking bright stars with Photoutils')
        back_box_size = (data.shape[0] // 10, data.shape[1] // 10)
        seg = detect(data, nsigma=brightstar_nsigma, npixels=npixels, fwhm=fwhm,
                     back_type='median', back_rms_type='mad', back_nsigma=back_nsigma, back_maxiters=back_maxiters,
                     back_size=back_box_size, back_filter_size=(3, 3), return_seg_only=True)
        mask = seg.data>0
    else:
        msgs.info('Masking bright stars with SExtractor.')
        tmp_root = 'mask_bright_star_tmp'
        par = fits.PrimaryHDU(data)
        par.writeto('{:}.fits'.format(tmp_root),overwrite=True)
        # configuration for the first SExtractor run
        sexconfig0 = {"CHECKIMAGE_TYPE": "OBJECTS", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                      "CATALOG_TYPE": "FITS_LDAC",
                      "CHECKIMAGE_NAME":"{:}_check.fits".format(tmp_root),
                      "DETECT_THRESH": brightstar_nsigma,
                      "ANALYSIS_THRESH": brightstar_nsigma,
                      "DETECT_MINAREA": 5}
        sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                      'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE',
                      'ELLIPTICITY',
                      'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER']
        sex.sexone('{:}.fits'.format(tmp_root), task=task, config=sexconfig0, workdir='./', params=sexparams0,
                   defaultconfig='pyphot', conv='sex995', nnw=None, dual=False, delete=True, log=False)
        data_check = fits.getdata("{:}_check.fits".format(tmp_root))
        mask = data_check>0
        msgs.info('Removing temporary files generated by SExtractor')
        os.system('rm {:}.fits'.format(tmp_root))
        os.system('rm {:}_check.fits'.format(tmp_root))
        os.system('rm {:}_cat.fits'.format(tmp_root))

    return mask

def calzpt(catalogfits, refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.],
           FLXSCALE=1.0, FLASCALE=1.0, external_flag=True, # This two paramers are exactly same with that used in SCAMP
           out_refcat=None, outqaroot=None):

    try:
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
        msgs.info('Reading SExtractor catalog failed. Reading photoutils catalog')
        catalog = Table.read(catalogfits)
        good_cat = catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO']>10
        catalog = catalog[good_cat]
        ra, dec = catalog['sky_centroid_icrs.ra'], catalog['sky_centroid_icrs.dec']

    pos = np.zeros((len(ra), 2))
    pos[:,0], pos[:,1] = ra, dec

    ra_cen, dec_cen = np.median(ra), np.median(dec)
    distance = np.sqrt((ra-ra_cen)*np.cos(dec_cen/180.*np.pi)**2 + (dec-dec_cen)**2)
    radius = np.nanmax(distance)*2.0 # query a bigger radius for safe given that you might have a big dither when re-use this catalog

    # Read/Query a reference catalog
    if (out_refcat is not None) and os.path.exists(out_refcat):
        msgs.info('Using the existing reference catalog {:} rather than downloading a new one.'.format(out_refcat))
        ref_data = Table.read(out_refcat, format='fits')
    else:
        ref_data = query.query_standard(ra_cen, dec_cen, catalog=refcatalog, radius=radius)
    # save the reference catalog to fits
    if (out_refcat is not None) and np.invert(os.path.exists(out_refcat)):
        msgs.info('Saving the reference catalog to {:}'.format(out_refcat))
        ref_data.write(out_refcat, format='fits')

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

    if nstar==0:
        msgs.warn('No matched standard stars were found')
        return 0., 0., nstar
    elif nstar < 10:
            msgs.warn('Only {:} standard stars were found'.format(nstar))
            _, zp, zp_std = stats.sigma_clipped_stats(matched_ref_mag - matched_cat_mag,
                                                      sigma=3, maxiters=20, cenfunc='median', stdfunc='std')
            return zp, zp_std, nstar
    else:
        _, zp, zp_std = stats.sigma_clipped_stats(matched_ref_mag - matched_cat_mag,
                                                  sigma=3, maxiters=20, cenfunc='median', stdfunc='std')
        # rerun the SExtractor with the zero point
        msgs.info('The zeropoint measured from {:} stars is {:0.3f}+/-{:0.3f}'.format(nstar, zp, zp_std))

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

            plt.plot(matched_ref_mag, matched_cat_mag+zp, 'k.')
            plt.plot([matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5],[matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5],'r--')
            plt.xlim(matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5)
            plt.ylim(matched_ref_mag.min()-0.5,matched_ref_mag.max()+0.5)
            plt.xlabel('Reference magnitude',fontsize=14)
            plt.ylabel('Calibrated magnitude',fontsize=14)
            plt.savefig(outqaroot+'_zpt_scatter.pdf')
            plt.close()

    return zp, zp_std, nstar

def cal_chips(cat_fits_list, sci_fits_list=None, ref_fits_list=None, outqa_root_list=None, ZP=25.0, external_flag=True,
              refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.], nstar_min=10):


    zp_all, zp_std_all = np.zeros(len(cat_fits_list)), np.zeros(len(cat_fits_list))
    nstar_all = np.zeros(len(cat_fits_list))
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

        par = fits.open(this_sci_fits)
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


        zp_this, zp_this_std, nstar = calzpt(this_cat, refcatalog=refcatalog, primary=primary, secondary=secondary,
                                   coefficients=coefficients, FLXSCALE=FLXSCALE, FLASCALE=FLASCALE, external_flag=external_flag,
                                   out_refcat=this_ref_name, outqaroot=this_qa_root)
        if nstar>nstar_min:
            msgs.info('Calibrating the zero point of {:} to {:} AB magnitude.'.format(os.path.basename(this_sci_fits),ZP))
            mag_ext = ZP-zp_this
            if mag_ext>0.5:
                msgs.warn('{:} has an extinction of {:} magnitude, cloudy???'.format(os.path.basename(this_sci_fits), mag_ext))
            else:
                msgs.info('{:} has an extinction of {:} magnitude, great conditions!'.format(os.path.basename(this_sci_fits), mag_ext))

            FLXSCALE *= 10**(0.4*(mag_ext))
            par[0].header['FLXSCALE'] = FLXSCALE
            par[0].header['FLASCALE'] = FLASCALE
            par[0].header['ZP'] = zp_this
            par[0].header['ZP_STD'] = zp_this_std
            par[0].header['ZP_NSTAR'] = nstar
            par.writeto(this_sci_fits, overwrite=True)
        else:
            msgs.warn('The number of stars found for calibration is smaller than nstar_min. skipping the ZPT calibrations.')

        zp_all[ii] = zp_this
        zp_std_all[ii] = zp_this_std
        nstar_all[ii] = nstar

    return zp_all, zp_std_all, nstar_all


def ForcedAperPhot(catalogs, images, rmsmaps, flagmaps, outfile=None, phot_apertures=[1.0,2.0,3.0,4.0,5.0], cat_ids=None, unique_dist=1.0):

    ncat = np.size(catalogs)
    if cat_ids is None:
        cat_ids = (np.arange(ncat)+1).astype('U').tolist()
    assert ncat == np.size(images), 'The numbers of images and catalogs should be the same'
    assert ncat == np.size(cat_ids), 'The numbers of cat_ids and catalogs should be the same'

    if rmsmaps is not None:
        assert ncat == np.size(rmsmaps), 'The numbers of images and rmsmaps should be the same'
    if flagmaps is not None:
        assert ncat == np.size(flagmaps), 'The numbers of images and flagmaps should be the same'

    ## Merge catalogs
    Table_Merged= Table.read(catalogs[0], 2)
    Table_Merged['CAT_ID'] = cat_ids[0]
    for icat in range(1,ncat):
        table_icat = Table.read(catalogs[icat], 2)
        table_icat['CAT_ID'] = cat_ids[icat]

        pos1 = np.zeros((len(Table_Merged), 2))
        try:
            pos1[:, 0], pos1[:, 1] = Table_Merged['ALPHA_J2000'],Table_Merged['DELTA_J2000']
        except:
            pos1[:, 0], pos1[:, 1] = Table_Merged['sky_centroid_icrs.ra'],Table_Merged['sky_centroid_icrs.dec']

        pos2 = np.zeros((len(table_icat), 2))
        try:
            pos2[:, 0], pos2[:, 1] = table_icat['ALPHA_J2000'],table_icat['DELTA_J2000']
        except:
            pos2[:, 0], pos2[:, 1] = table_icat['sky_centroid_icrs.ra'],table_icat['sky_centroid_icrs.dec']

        ## cross-match with 1 arcsec
        dist, ind = crossmatch.crossmatch_angular(pos2, pos1, max_distance=unique_dist / 3600.)
        no_match = np.isinf(dist)
        Table_Merged =vstack([Table_Merged, table_icat[no_match]])

    ## Prepare the output forced photometry catalog
    ## ToDo: Currently only used the posotions and flags of the merged catalog is included.
    ##       Next step is to keep all the origin columns of the input catalog
    Table_Forced = Table()
    Table_Forced['CAT_ID'] = Table_Merged['CAT_ID']
    Table_Forced['FORCED_ID'] = (np.arange(len(Table_Forced))+1).astype('int32')
    try:
        Table_Forced['RA'], Table_Forced['DEC']= Table_Merged['ALPHA_J2000'],Table_Merged['DELTA_J2000']
    except:
        Table_Forced['RA'], Table_Forced['DEC'] = Table_Merged['sky_centroid_icrs.ra'], Table_Merged['sky_centroid_icrs.dec']

    if 'CLASS_STAR' in Table_Merged.keys():
        Table_Forced['CLASS_STAR'] = Table_Merged['CLASS_STAR']
    if 'FLAGS' in Table_Merged.keys():
        Table_Forced['FLAGS'] = Table_Merged['FLAGS']
    if 'IMAFLAGS_ISO' in Table_Merged.keys():
        Table_Forced['IMAFLAGS_ISO'] = Table_Merged['IMAFLAGS_ISO']
    if 'NIMAFLAGS_ISO' in Table_Merged.keys():
        Table_Forced['NIMAFLAGS_ISO'] = Table_Merged['NIMAFLAGS_ISO']

    ## Let's perform the forced aperture photometry on each image
    positions = SkyCoord(ra=Table_Forced['RA'], dec=Table_Forced['DEC'], unit=(u.deg, u.deg))

    ## Perform aperture photometry for all merged sources
    for ii, this_image in enumerate(images):
        msgs.info('Performing forced aperture photometry on {:}'.format(this_image))
        data = fits.getdata(this_image)
        header = fits.getheader(this_image)
        wcs_info = wcs.WCS(header)

        try:
            zpt = header['ZP']
        except:
            zpt = 0.

        if flagmaps is not None:
            ## good pixels with flag==0
            flag = fits.getdata(flagmaps[ii])
            mask = flag > 0.
        else:
            mask = None

        if rmsmaps is not None:
            error = fits.getdata(rmsmaps[ii])
        else:
            error = None

        apertures = [SkyCircularAperture(positions, r=d/2*u.arcsec) for d in phot_apertures]
        tbl_aper = aperture_photometry(data, apertures, error=error, mask=mask, method='exact', wcs=wcs_info)
        flux_aper = np.zeros((len(tbl_aper), np.size(phot_apertures)))
        fluxerr_aper = np.zeros_like(flux_aper)
        mag_aper = np.zeros_like(flux_aper)
        magerr_aper = np.zeros_like(flux_aper)
        for jj in range(np.size(phot_apertures)):
            flux_aper[:,jj] = tbl_aper['aperture_sum_{:d}'.format(jj)]
            fluxerr_aper[:,jj] = tbl_aper['aperture_sum_err_{:d}'.format(jj)]
            mag_aper[:,jj] =  -2.5*np.log10(tbl_aper['aperture_sum_{:d}'.format(jj)])
            magerr_aper[:,jj] = 2.5/np.log(10)*tbl_aper['aperture_sum_err_{:d}'.format(jj)]/tbl_aper['aperture_sum_{:d}'.format(jj)]

        Table_Forced['FORCED_XCENTER_{:}'.format(cat_ids[ii])] = tbl_aper['xcenter']
        Table_Forced['FORCED_YCENTER_{:}'.format(cat_ids[ii])] = tbl_aper['ycenter']
        #Table_Forced['FORCED_SKY_CENTER_{:}'.format(cat_ids[ii])] = tbl_aper['sky_center']
        Table_Forced['FORCED_MAG_APER_{:}'.format(cat_ids[ii])] = mag_aper + zpt
        Table_Forced['FORCED_MAGERR_APER_{:}'.format(cat_ids[ii])] = magerr_aper
        Table_Forced['FORCED_FLUX_APER_{:}'.format(cat_ids[ii])] = flux_aper
        Table_Forced['FORCED_FLUXERR_APER_{:}'.format(cat_ids[ii])] = fluxerr_aper

        badmag = np.isinf(mag_aper) | np.isnan(mag_aper)
        Table_Forced['FORCED_MAG_APER_{:}'.format(cat_ids[ii])][badmag] = 99.
        Table_Forced['FORCED_MAGERR_APER_{:}'.format(cat_ids[ii])][badmag] = 99.

        badphot = (flux_aper == 0.)
        Table_Forced['FORCED_FLAG_APER_{:}'.format(cat_ids[ii])] = np.sum(badphot,axis=1)

    if outfile is not None:
        Table_Forced.write(outfile,format='fits', overwrite=True)

    return Table_Forced