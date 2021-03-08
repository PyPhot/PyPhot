import os
from astropy.io import fits

from pyphot import msgs
from pyphot import sex, scamp, swarp


def astrometric(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, science_path='./',
                detect_thresh=3.0, analysis_thresh=3.0, detect_minarea=5, crossid_radius=2.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', position_maxerr=0.5,
                pixscale_maxerr=1.1, mosaic_type='UNCHANGED',
                weight_type='MAP_WEIGHT',delete=False, log=True):

    ## ToDo: Not sure why, but the scamp fails for MMIRS and IMACS, so I will try to resample it with swarp
    ## This step is basically align the image to N to the up and E to the left
    # configuration for the first swarp run
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                    "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN",
                    "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type}
    # resample science image
    swarp.swarpall(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)
    # resample flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarp.swarpall(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)

    ## remove useless data and change flag image type to int32
    sci_fits_list_resample = []
    wht_fits_list_resample = []
    flag_fits_list_resample = []
    for i in range(len(sci_fits_list)):
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))
        sci_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.fits'))
        wht_fits_list_resample.append(sci_fits_list[i].replace('.fits', '.resamp.weight.fits'))
        flag_fits_list_resample.append(flag_fits_list[i].replace('.fits', '.resamp.fits'))
        par = fits.open(flag_fits_list[i].replace('.fits', '.resamp.fits'))
        par[0].data = par[0].data.astype('int32')
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
                  'IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS']
    sex.sexall(sci_fits_list_resample, task='sex', config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='995', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    # configuration for the first scamp run
    #
    scampconfig0 = {"CROSSID_RADIUS": crossid_radius,
                    "ASTREF_CATALOG": astref_catalog,
                    "ASTREF_BAND": astref_band,
                    "POSITION_MAXERR": position_maxerr,
                    "PIXSCALE_MAXERR": pixscale_maxerr,
                    "MOSAIC_TYPE": mosaic_type}
    scamp.scampall(sci_fits_list_resample, config=scampconfig0, workdir=science_path, defaultconfig='pyphot',
                   delete=delete, log=log)

    ## copy the .head for flag images
    for i in range(len(sci_fits_list_resample)):
        os.system('cp {:} {:}'.format(sci_fits_list_resample[i].replace('.fits', '.head'),
                                      flag_fits_list_resample[i].replace('.fits', '.head')))

    # configuration for the second swarp run
    swarpconfig['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    # resample the science image
    swarp.swarpall(sci_fits_list_resample, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)
    # resample the flag image
    swarpconfig_flag['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    swarp.swarpall(flag_fits_list_resample, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)

    # delete unnecessary weight maps and head
    for i in range(len(sci_fits_list)):
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp.cat')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp.head')))
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.head')))

    # change flag image type to int32
    for i in range(len(sci_fits_list)):
        par = fits.open(flag_fits_list_resample[i])
        par[0].data = par[0].data.astype('int32')
        par[0].writeto(flag_fits_list_resample[i], overwrite=True)

    return sci_fits_list_resample, wht_fits_list_resample, flag_fits_list_resample


def coadd(scifiles, flagfiles, coaddroot, pixscale, science_path, coadddir, weight_type='MAP_WEIGHT',
          rescale_weights=False, combine_type='median', clip_ampfrac=0.3, clip_sigma=4.0, blank_badpixels=False,
          subtract_back= False, back_type='AUTO', back_default=0.0, back_size=200, back_filtersize=3,
          back_filtthresh=0.0, delete=True, log=True):

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
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "RESAMPLE_SUFFIX": ".tmp.fits",
                   "PIXELSCALE_TYPE": "MANUAL", "PIXEL_SCALE": pixscale,
                   "WEIGHT_TYPE": weight_type,"RECALE_WEIGHTS": rescale,"BLANK_BADPIXELS":blank,
                   "COMBINE_TYPE": combine_type.upper(),"CLIP_AMPFRAC":clip_ampfrac,"CLIP_SIGMA":clip_sigma,
                   "SUBTRACT_BACK": subtract,"BACK_TYPE":back_type,"BACK_DEFAULT":back_default,"BACK_SIZE":back_size,
                   "BACK_FILTERSIZE":back_filtersize,"BACK_FILTTHRESH":back_filtthresh}

    swarp.swarpall(scifiles, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=coaddroot + '_sci', delete=delete, log=log)
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = "NONE"
    swarp.swarpall(flagfiles, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=coaddroot + '_flag', delete=delete, log=log)
    # delete unnecessary files
    os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.swarp.xml')))
    os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.weight.fits')))

    # useful file names
    coadd_file = os.path.join(coadddir,coaddroot+'_sci.fits')
    coadd_flag_file = os.path.join(coadddir, coaddroot + '_flag.fits')
    coadd_wht_file = os.path.join(coadddir, coaddroot + '_sci.weight.fits')

    # change flag image type to int32
    par = fits.open(coadd_flag_file)
    par[0].data = (par[0].data > 0.).astype('int32')
    par[0].writeto(coadd_flag_file, overwrite=True)

    return coadd_file, coadd_wht_file, coadd_flag_file
