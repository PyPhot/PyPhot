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

from pyphot import msgs
from pyphot import sex, scamp, swarp
from pyphot import query, crossmatch


def astrometric(sci_fits_list, wht_fits_list, flag_fits_list, pixscale, science_path='./',qa_path='./',
                detect_thresh=3.0, analysis_thresh=3.0, detect_minarea=5, crossid_radius=2.0,
                astref_catalog='GAIA-DR2', astref_band='DEFAULT', position_maxerr=0.5,
                pixscale_maxerr=1.1, mosaic_type='UNCHANGED',task='sex',
                weight_type='MAP_WEIGHT',delete=False, log=True):

    ## ToDo: Not sure why, but the scamp fails for MMIRS and IMACS, so I will try to resample it with swarp
    ## This step is basically align the image to N to the up and E to the left
    # configuration for the first swarp run
    # Note that I would apply the gain correction before doing the astrometric calibration, so I set Gain to 1.0
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                   "PIXEL_SCALE": pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT":1.0,
                   "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                   "HEADER_SUFFIX":"_cat.head"}
    # resample science image
    swarp.swarpall(sci_fits_list, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)
    # resample flag image
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarp.swarpall(flag_fits_list, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=False)

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
                  'FLUX_RADIUS','IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR', 'FLAGS', 'FLAGS_WEIGHT']
    sex.sexall(sci_fits_list_resample, task=task, config=sexconfig0, workdir=science_path, params=sexparams0,
               defaultconfig='pyphot', conv='sex995', nnw=None, dual=False, delete=delete, log=log,
               flag_image_list=flag_fits_list_resample, weight_image_list=wht_fits_list_resample)

    # configuration for the first scamp run
    #
    scampconfig0 = {"CROSSID_RADIUS": 1.0,
                    "ASTREF_CATALOG": astref_catalog,
                    "ASTREF_BAND": astref_band,
                    "POSITION_MAXERR": position_maxerr,
                    "PIXSCALE_MAXERR": pixscale_maxerr,
                    "MOSAIC_TYPE": mosaic_type,
                    "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                    "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
    scamp.scampall(sci_fits_list_resample, config=scampconfig0, workdir=science_path, QAdir=qa_path,
                   defaultconfig='pyphot', delete=delete, log=log)

    ## copy the .head for flag images
    for i in range(len(sci_fits_list_resample)):
        os.system('cp {:} {:}'.format(sci_fits_list_resample[i].replace('.fits', '_cat.head'),
                                      flag_fits_list_resample[i].replace('.fits', '_cat.head')))

    # configuration for the second swarp run
    swarpconfig['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    # resample the science image
    swarp.swarpall(sci_fits_list_resample, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=log)
    # resample the flag image
    swarpconfig_flag['RESAMPLE_SUFFIX'] = '.fits' # overwright the previous resampled image
    swarp.swarpall(flag_fits_list_resample, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                   coaddroot=None, delete=delete, log=False)

    # delete unnecessary weight maps and head
    for i in range(len(sci_fits_list)):
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp.weight.fits')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp_cat.fits')))
        os.system('rm {:}'.format(sci_fits_list[i].replace('.fits', '.resamp_cat.head')))
        os.system('rm {:}'.format(flag_fits_list[i].replace('.fits', '.resamp_cat.head')))

    # change flag image type to int32
    for i in range(len(sci_fits_list)):
        par = fits.open(flag_fits_list_resample[i])
        # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the last Swarp run and will be used by Swaro later on)
        # in order to get the correct flag when doing the coadd with swarp in the later step
        par[0].header['EXPTIME'] = 1.0
        par[0].header['FLXSCALE'] = 1.0
        par[0].header['FLASCALE'] = 1.0
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

    ## parameters for coadding science images
    swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "RESAMPLE_SUFFIX": ".tmp.fits",
                   "PIXELSCALE_TYPE": "MANUAL", "PIXEL_SCALE": pixscale,
                   "WEIGHT_TYPE": weight_type,"RECALE_WEIGHTS": rescale,"BLANK_BADPIXELS":blank,
                   "COMBINE_TYPE": combine_type.upper(),"CLIP_AMPFRAC":clip_ampfrac,"CLIP_SIGMA":clip_sigma,
                   "SUBTRACT_BACK": subtract,"BACK_TYPE":back_type,"BACK_DEFAULT":back_default,"BACK_SIZE":back_size,
                   "BACK_FILTERSIZE":back_filtersize,"BACK_FILTTHRESH":back_filtthresh}

    swarp.swarpall(scifiles, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                   coadddir=coadddir, coaddroot=coaddroot + '_sci', delete=delete, log=log)

    ## parameters for coadding flag images
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = "NONE"
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
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


def detect(data, wcs_info, rmsmap=None, bkgmap=None, mask=None, effective_gain=None, nsigma=2., npixels=5, fwhm=5,
           nlevels=32, contrast=0.001, back_nsigma=3, back_maxiters=10, back_type='Median', back_rms_type='Std',
           back_filter=(200, 200), back_filter_size=(3, 3), morp_filter=False,
           phot_apertures=[1.0,2.0,3.0,4.0,5.0]):
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
        back_filter:
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
        msgs.info('Estimating the 2D background that will be used for the detection and photometry.')
        bkg = Background2D(data, back_filter, mask=mask, filter_size=back_filter_size, sigma_clip=sigma_clip,
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

def calzpt(catalogfits, refcatalog='Panstarrs', primary='i', secondary='z', coefficients=[0.,0.,0.], outqa=None):

    try:
        catalog = Table.read(catalogfits)
        good_cat = catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO']>5
        catalog = catalog[good_cat]
        ra, dec = catalog['sky_centroid_icrs.ra'], catalog['sky_centroid_icrs.dec']
    except:
        catalog = Table.read(catalogfits, 2)
        good_cat = (catalog['IMAFLAGS_ISO']<1) & (catalog['NIMAFLAGS_ISO']<1) & (catalog['FLAGS']<1) & (catalog['CLASS_STAR']>0.5)
        good_cat &= catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO']>5
        catalog = catalog[good_cat]
        ra, dec = catalog['ALPHA_J2000'], catalog['DELTA_J2000']

    pos = np.zeros((len(ra), 2))
    pos[:,0], pos[:,1] = ra, dec

    ra_cen, dec_cen = np.median(ra), np.median(dec)
    distance = np.sqrt((ra-ra_cen)*np.cos(dec_cen/180.*np.pi)**2 + (dec-dec_cen)**2)
    radius = np.nanmax(distance)

    ref_data = query.query_region(ra_cen, dec_cen, catalog=refcatalog, radius=radius)
    good_ref = (1.0857/ref_data['e_{:}mag'.format(primary)]>5) & (1.0857/ref_data['e_{:}mag'.format(secondary)]>5)
    try:
        ref_data = ref_data[good_ref.data]
    except:
        ref_data = ref_data[good_ref]

    ref_mag = ref_data['{:}mag'.format(primary)] + coefficients[0] + \
              coefficients[1]*(ref_data['{:}mag'.format(primary)]-ref_data['{:}mag'.format(secondary)])+ \
              coefficients[2] * (ref_data['{:}mag'.format(primary)] - ref_data['{:}mag'.format(secondary)])**2

    ref_ra, ref_dec = ref_data['RAJ2000'], ref_data['DEJ2000']
    ref_pos = np.zeros((len(ref_ra), 2))
    ref_pos[:,0], ref_pos[:,1] = ref_ra, ref_dec

    ## cross-match with 1 arcsec
    dist, ind = crossmatch.crossmatch_angular(pos, ref_pos, max_distance=1.0/3600.)
    matched = np.invert(np.isinf(dist))

    matched_cat_mag = catalog['MAG_AUTO'][matched]
    matched_ref_mag = ref_mag[ind[matched]]
    nstar = np.sum(matched)

    _, zp, zp_std = stats.sigma_clipped_stats(matched_ref_mag - matched_cat_mag,
                                              sigma=3, maxiters=20, cenfunc='median', stdfunc='std')

    if outqa is not None:
        msgs.info('Make a histogram plot for the zpt')
        zp0 = zp - 7*zp_std
        zp1 = zp + 7*zp_std
        num = plt.hist(matched_ref_mag - matched_cat_mag, bins=np.linspace(zp0,zp1,int(nstar/100)))
        nmax = np.max(num[0]*1.1)
        plt.vlines(zp,0,nmax, linestyles='--', colors='r')
        plt.vlines(zp+zp_std,0,nmax, linestyles=':', colors='r')
        plt.vlines(zp-zp_std,0,nmax, linestyles=':', colors='r')
        plt.ylim(0,nmax)
        plt.xlabel('Zero point',fontsize=14)
        plt.ylabel('# stars used for the calibration',fontsize=14)
        plt.text(zp-6.5*zp_std,0.8*nmax,r'{:0.3f}$\pm${:0.3f}'.format(zp,zp_std),fontsize=14)
        plt.savefig(outqa)
    # rerun the SExtractor with the zero point
    msgs.warn('The zeropoint measured from {:} stars is {:0.3f}+/-{:0.3f}'.format(nstar, zp, zp_std))

    return zp, zp_std, nstar

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