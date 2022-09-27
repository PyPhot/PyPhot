import gc, os
import numpy as np
import numpy.ma as ma
import random
import string
from scipy import ndimage

from astropy import wcs
from astropy import stats
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.stats import SigmaClip, sigma_clip
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy import units as u

from photutils import detect_sources
from photutils import deblend_sources
#from photutils import source_properties
from photutils.utils import calc_total_error
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils.segmentation import SourceCatalog

from photutils import StdBackgroundRMS, MADStdBackgroundRMS, BiweightScaleBackgroundRMS
from photutils import Background2D, MeanBackground, MedianBackground, SExtractorBackground
from photutils import MMMBackground, BiweightLocationBackground, ModeEstimatorBackground

from pyphot import msgs, sex, io, utils
from pyphot import crossmatch


def BKG2D(data, back_size, mask=None, filter_size=(3, 3), sigclip=5, back_type='median', back_rms_type='std',
          back_maxiters=5, sextractor_task='sex', verbose=True):

    ## Sky background subtraction

    if 'median' in back_type.lower():
        bkg_estimator = MedianBackground()
    elif back_type.lower() == 'mean':
        bkg_estimator = MeanBackground()
    elif back_type.lower() == 'sextractor':
        #bkg_estimator = SExtractorBackground()
        bkg_estimator = 'sextractor'
    elif back_type.lower() == 'mmm':
        bkg_estimator = MMMBackground()
    elif back_type.lower() == 'biweight':
        bkg_estimator = BiweightLocationBackground()
    elif back_type.lower() == 'mode':
        bkg_estimator = ModeEstimatorBackground()
    else:
        msgs.warn('{:} Background is not found, using MedianBackground Instead.'.format(back_type))
        back_type = 'median'
        bkg_estimator = MedianBackground()

    if back_rms_type.lower() == 'std':
        bkgrms_estimator = StdBackgroundRMS()
    elif back_rms_type.lower() == 'mad':
        bkgrms_estimator = MADStdBackgroundRMS()
    elif back_rms_type.lower() == 'biweight':
        bkgrms_estimator = BiweightScaleBackgroundRMS()
    else:
        msgs.warn('{:} Background RMS type is not found, using STD Instead.'.format(back_rms_type))
        bkgrms_estimator = StdBackgroundRMS()

    if bkg_estimator == 'sextractor':
        if verbose:
            msgs.info('Estimating BACKGROUND with SExtractor.')
        letters = string.ascii_letters
        random_letter = ''.join(random.choice(letters) for i in range(15))
        tmp_root = 'sex_bkg_tmp_{:}_{:04d}'.format(random_letter, np.random.randint(1,9999))

        # perform rejections
        tmp_data = ma.masked_array(data, mask=mask, fill_value=np.nan)
        filtered_data = sigma_clip(tmp_data, sigma=sigclip, maxiters=back_maxiters, masked=True)
        tmp_data = data.copy()
        tmp_data[filtered_data.mask] = np.nan
        par = fits.PrimaryHDU(tmp_data)
        par.writeto('{:}.fits'.format(tmp_root),overwrite=True)

        # configuration for the first SExtractor run
        if np.size(back_size)==1:
            back_size = [back_size, back_size]
        if np.size(filter_size)==1:
            filter_size = [filter_size, filter_size]
        sexconfig = {"CHECKIMAGE_TYPE": "BACKGROUND, BACKGROUND_RMS", "WEIGHT_TYPE": "NONE", "CATALOG_TYPE": "FITS_LDAC",
                      "CHECKIMAGE_NAME":"{:}_bkg.fits, {:}_rms.fits".format(tmp_root,tmp_root),
                      "DETECT_THRESH": 5, "ANALYSIS_THRESH": 5, "DETECT_MINAREA": 5,
                      "BACK_SIZE": '{:},{:}'.format(back_size[0],back_size[1]),
                     "BACK_FILTERSIZE":'{:},{:}'.format(filter_size[0],filter_size[1])}
        sexparams = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                      'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE', 'ELLIPTICITY',
                      'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER']
        sex.sexone('{:}.fits'.format(tmp_root), task=sextractor_task, config=sexconfig, workdir='./', params=sexparams,
                   defaultconfig='pyphot', conv='sex', nnw=None, dual=False, delete=True, log=False, verbose=verbose)
        bkg_map = fits.getdata("{:}_bkg.fits".format(tmp_root))
        rms_map = fits.getdata("{:}_rms.fits".format(tmp_root))
        if verbose:
            msgs.info('Removing temporary files generated by SExtractor')
        os.system('rm {:}.fits'.format(tmp_root))
        os.system('rm {:}_bkg.fits'.format(tmp_root))
        os.system('rm {:}_rms.fits'.format(tmp_root))
        os.system('rm {:}_cat.fits'.format(tmp_root))
        del tmp_data, filtered_data
        gc.collect()
    else:
        if verbose:
            msgs.info('Estimating {:} BACKGROUND with Photutils Background2D.'.format(back_type))
        tmp = data.copy()
        Sigma_Clip = SigmaClip(sigma=sigclip, maxiters=back_maxiters)
        bkg = Background2D(tmp, back_size, mask=mask, filter_size=filter_size, sigma_clip=Sigma_Clip,
                           bkg_estimator=bkg_estimator, bkgrms_estimator=bkgrms_estimator)
        bkg_map, rms_map = bkg.background, bkg.background_rms
        bkg_map[tmp==0.] = 0.
        del tmp, bkg
        gc.collect()

    if back_type == 'GlobalMedian':
        bkg_map = np.ones_like(bkg_map) * np.nanmedian(bkg_map[np.invert(mask)])

    return bkg_map, rms_map


def photutils_detect(data, wcs_info=None, rmsmap=None, bkgmap=None, mask=None,
                     effective_gain=None, nsigma=2., npixels=5, fwhm=5, zpt=0.,
                     nlevels=32, contrast=0.001, back_nsigma=3, back_maxiters=10, back_type='median', back_rms_type='std',
                     back_size=(200, 200), back_filter_size=(3, 3), morp_filter=False, sextractor_task='sex',
                     phot_apertures=[1.0,2.0,3.0,4.0,5.0], return_seg_only=False, verbose=True):
    '''
        Detect sources from a FITS image
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

    if (rmsmap is None) or (bkgmap is None):
        background_array, background_rms = BKG2D(data, back_size, mask=mask, filter_size=back_filter_size,
                                                 sigclip=back_nsigma, back_type=back_type, back_rms_type=back_rms_type,
                                                 back_maxiters=back_maxiters, sextractor_task=sextractor_task, verbose=verbose)
        if rmsmap is None:
            rmsmap = background_rms
        if bkgmap is None:
            bkgmap = background_array

    threshold = bkgmap + nsigma * rmsmap
    error = calc_total_error(data,rmsmap, effective_gain)

    ## Build a Gaussian kernel
    sigma = fwhm * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()

    ## Do the detection using the Image Segmentation technique
    ## The return is a SegmentationImage
    msgs.info('Detecting targets with detect_sources')
    segm = detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)

    if return_seg_only:
        return segm

    # Source Deblending
    if verbose:
        msgs.info('Deblending with deblend_sources')
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
    if verbose:
        msgs.info('Measuring source properties with SourceCatalog')
    if morp_filter:
        cat = SourceCatalog(data, segm_deblend, mask=mask, background=bkgmap, error=error,
                            kernel=kernel, wcs=wcs_info)
    else:
        cat = SourceCatalog(data, segm_deblend, mask=mask, background=bkgmap, error=error, wcs=wcs_info)
    tbl = cat.to_table()
    tbl = tbl[np.invert(np.isnan(tbl['xcentroid']))] # remove sources with nan positions
    tbl['MAG_AUTO'] = -2.5*np.log10(tbl['segment_flux']) + zpt
    tbl['MAGERR_AUTO'] = 2.5/np.log(10)*tbl['segment_fluxerr']/tbl['segment_flux']
    tbl['FLUX_AUTO'] = tbl['segment_flux']
    tbl['FLUXERR_AUTO'] = tbl['segment_fluxerr']

    ## Perform Aperture photometry
    if verbose:
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
        mag_aper[:,ii] =  -2.5*np.log10(tbl_aper['aperture_sum_{:d}'.format(ii)]) + zpt
        magerr_aper[:,ii] = 2.5/np.log(10)*tbl_aper['aperture_sum_err_{:d}'.format(ii)]/tbl_aper['aperture_sum_{:d}'.format(ii)]
    tbl['MAG_APER'] = mag_aper
    tbl['MAGERR_APER'] = magerr_aper
    tbl['FLUX_APER'] = flux_aper
    tbl['FLUXERR_APER'] = fluxerr_aper

    ## ToDo: Add PSF photometry

    return tbl, rmsmap, bkgmap


def mask_bright_star(data, mask=None, brightstar_nsigma=3, back_nsigma=3, back_maxiters=10, npixels=5, fwhm=5,
                     method='sextractor', conv='sex', task='sex', erosion=11, dilation=50, verbose=True):

    data_copy = data.copy()
    if method.lower()=='photoutils':
        if mask is not None:
            data_copy[mask] = 0.  # zero out bad pixels
        if verbose:
            msgs.info('Masking bright stars with Photoutils')
        back_box_size = (data_copy.shape[0] // 10, data_copy.shape[1] // 10)
        seg = photutils_detect(data_copy, nsigma=brightstar_nsigma, npixels=npixels, fwhm=fwhm,
                               back_type='median', back_rms_type='mad', back_nsigma=back_nsigma, back_maxiters=back_maxiters,
                               back_size=back_box_size, back_filter_size=(3, 3), return_seg_only=True, verbose=verbose)
        starmask = seg.data > 0
        del data_copy, seg
        gc.collect()
    else:
        if mask is not None:
            data_copy[mask] = np.nan # set to zero would create junk detections at the edges
        if verbose:
            msgs.info('Masking bright stars with SExtractor.')
        letters = string.ascii_letters
        random_letter = ''.join(random.choice(letters) for i in range(15))
        tmp_root = 'mask_bright_star_tmp_{:}_{:04d}'.format(random_letter, np.random.randint(1,9999))
        par = fits.PrimaryHDU(data_copy)
        par.writeto('{:}.fits'.format(tmp_root),overwrite=True)
        # configuration for the first SExtractor run
        sexconfig0 = {"CHECKIMAGE_TYPE": "OBJECTS", "WEIGHT_TYPE": "NONE", "CATALOG_NAME": "dummy.cat",
                      "CATALOG_TYPE": "FITS_LDAC",
                      "CHECKIMAGE_NAME":"{:}_check.fits".format(tmp_root),
                      "DETECT_THRESH": brightstar_nsigma,
                      "ANALYSIS_THRESH": brightstar_nsigma,
                      "DETECT_MINAREA": npixels}
        sexparams0 = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                      'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE',
                      'ELLIPTICITY',
                      'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_APER', 'MAGERR_APER']
        sex.sexone('{:}.fits'.format(tmp_root), task=task, config=sexconfig0, workdir='./', params=sexparams0,
                   defaultconfig='pyphot', conv=conv, nnw=None, dual=False, delete=True, log=False, verbose=verbose)
        data_check = fits.getdata("{:}_check.fits".format(tmp_root))
        starmask = data_check > 0
        if verbose:
            msgs.info('Removing temporary files generated by SExtractor')
        os.system('rm {:}.fits'.format(tmp_root))
        os.system('rm {:}_check.fits'.format(tmp_root))
        os.system('rm {:}_cat.fits'.format(tmp_root))
        del data_copy
        gc.collect()

    # grow mask for bright stars
    bright_blob = ndimage.binary_erosion(starmask, iterations=erosion)
    mask_bright_blob = ndimage.binary_dilation(bright_blob, iterations=dilation)
    objmask = np.logical_or(mask_bright_blob, starmask)

    return objmask


def mag_limit(image, Nsigma=5, image_type='science', zero_point=None, phot_apertures=[1.0,2.0,3.0,4.0,5.0], Npositions=10000,
              sigclip=3, maxiters=10, back_type='median', back_size=(200,200), back_filtersize=(3, 3),
              maskbrightstar_method='sextractor', conv='sex', brightstar_nsigma=5, erosion=11, dilation=50,
              sextractor_task='sex'):
    '''
        Estimating limiting magnitude for a given fits image
    Args:
        image (str): image name
        Nsigma (int or float): 5-sigma limit?
        image_type (str): the image type of your input image: science, rms, variance, or inverse variance
        zero_point (int or float): zeropoint of your image, if None it will find ZP keyword from the image header
        phot_apertures (int, float, or list): aperture diameters
        Npoints (int): the number of random positions that we be measure from the image
        sigclip (int or float): how many sigma you want to use for rejections
        maxiters (int): The maximum iterations for the rejection
        back_type (str): Bakcground type for estimating the variance map. Only used for science image_type
        back_size (int or list): The size for estimating the varinace map.  Only used for science image_type
        back_filtersize (int or list):  The filter size for estimating the varinace map.  Only used for science image_type
        maskbrightstar_method (str): Method for masking bright stars. Only used for science image_type
        brightstar_nsigma (int or float): Nsigma used for masking bright star. Only used for science image_type
        sextractor_task (str): how to call your sextractor, sex or sextractor? Only used for science image_type
    Returns:
        maglims (1D numpy array): limiting magnitudes for the given apertures.
    '''

    if np.isscalar(phot_apertures):
        phot_apertures = [phot_apertures]

    # Load the data
    header, data, flag = io.load_fits(image)
    wcs_info = wcs.WCS(header)
    #pixscale = np.mean(wcs.utils.proj_plane_pixel_scales(wcs_info)) * 3600.0

    if zero_point is None:
        try:
            zpt = header['ZP']
        except:
            msgs.warn('Zero point was not given, set zpt=0')
            zpt = 0.
    else:
        zpt = zero_point

    # Determine the image type
    if image_type=='science':
        msgs.info('Getting limiting magnitudes from Science image {:}'.format(image))
        starmask = mask_bright_star(data, mask=flag>0, brightstar_nsigma=brightstar_nsigma, back_nsigma=sigclip,
                                    back_maxiters=maxiters, method=maskbrightstar_method, conv=conv,
                                    erosion=erosion, dilation=dilation, task=sextractor_task)
        mask_bkg = (flag>0) | starmask
        _, rmsmap = BKG2D(data, back_size, mask=mask_bkg, filter_size=back_filtersize,
                          sigclip=sigclip, back_type=back_type, back_rms_type='std',
                          back_maxiters=maxiters, sextractor_task=sextractor_task)
        variancemap = rmsmap**2
    elif image_type=='rms':
        msgs.info('Getting limiting magnitudes from RMS image {:}'.format(image))
        variancemap = data**2
    elif image_type=='variance':
        msgs.info('Getting limiting magnitudes from Variance image {:}'.format(image))
        variancemap = data
    elif image_type=='invar':
        msgs.info('Getting limiting magnitudes from Variance image {:}'.format(image))
        variancemap = utils.inverse(data)
    else:
        msgs.error('Only the following image_type are acceptable: science, rms, variance, invar.')
        variancemap = None

    # Generate random positions
    nx, ny = data.shape
    xx = np.random.randint(0,nx,Npositions)
    yy = np.random.randint(0,ny,Npositions)
    positions = wcs_info.pixel_to_world(xx, yy)

    apertures = [SkyCircularAperture(positions, r= d/2*u.arcsec) for d in phot_apertures]
    tbl_aper = aperture_photometry(variancemap, apertures, error=None, mask=None, method='exact', wcs=wcs_info)

    maglims = np.zeros(len(phot_apertures))
    for ii in range(len(phot_apertures)):
        flux = tbl_aper['aperture_sum_{:d}'.format(ii)]
        mask = np.isnan(flux) | (flux==0.)
        mean, median, stddev = stats.sigma_clipped_stats(flux, mask=mask, sigma=sigclip, maxiters=maxiters,
                                                         cenfunc='median', stdfunc='std')
        maglims[ii] = round(zpt - 2.5*np.log10(np.sqrt(median)*Nsigma),2)
        msgs.info('The {:}-sigma limit for {:} arcsec diameter aperture is {:0.2f} magnitude'.format(Nsigma, phot_apertures[ii], maglims[ii]))

    return maglims


def mergecat(catalogs, outfile=None, cat_ids=None, unique_dist=1.0):

    ncat = np.size(catalogs)
    if cat_ids is None:
        cat_ids = (np.arange(ncat)+1).astype('U').tolist()
    if ncat != np.size(cat_ids):
        msgs.error('The numbers of cat_ids and catalogs should be the same')

    ## Identify the coordinates of the unique sources
    Table0 = Table.read(catalogs[0], 2)
    Table_Merged = Table()
    Table_Merged['DET_CAT_ID'] = [cat_ids[0]]*len(Table0)
    try:
        Table_Merged['RA'] = Table0['ALPHA_J2000']
        Table_Merged['DEC'] = Table0['DELTA_J2000']
    except:
        Table_Merged['RA'] = Table0['sky_centroid_icrs.ra']
        Table_Merged['DEC'] = Table0['sky_centroid_icrs.dec']

    for icat in range(1,ncat):
        table_icat = Table.read(catalogs[icat], 2)
        table_icat['DET_CAT_ID'] = cat_ids[icat]

        pos1 = np.zeros((len(Table_Merged), 2))
        pos1[:, 0], pos1[:, 1] = Table_Merged['RA'],Table_Merged['DEC']

        pos2 = np.zeros((len(table_icat), 2))
        try:
            pos2[:, 0], pos2[:, 1] = table_icat['ALPHA_J2000'],table_icat['DELTA_J2000']
        except:
            pos2[:, 0], pos2[:, 1] = table_icat['sky_centroid_icrs.ra'],table_icat['sky_centroid_icrs.dec']

        ## cross-match with pos2 as the left table and pos1 as the right table
        ## i.e. For every pos2 target, we will check whether it has a counterpart in pos1
        dist, ind = crossmatch.crossmatch_angular(pos2, pos1, max_distance=unique_dist / 3600.)
        no_match = np.isinf(dist)

        ## stack the catalog
        this_table = Table()
        this_table['DET_CAT_ID'] = table_icat[no_match]['DET_CAT_ID']
        this_table['RA'] = pos2[:, 0][no_match]
        this_table['DEC'] = pos2[:, 1][no_match]

        Table_Merged =vstack([Table_Merged, this_table])

    ## Add a source_id column
    Table_Merged.add_column(np.arange(len(Table_Merged))+1, name='SOURCE_ID', index=0)

    ## Assign the original columns to unique sources
    pos1 = np.zeros((len(Table_Merged), 2))
    pos1[:, 0], pos1[:, 1] = Table_Merged['RA'], Table_Merged['DEC']
    for icat in range(ncat):
        table_icat = Table.read(catalogs[icat], 2)
        pos2 = np.zeros((len(table_icat), 2))
        try:
            pos2[:, 0], pos2[:, 1] = table_icat['ALPHA_J2000'],table_icat['DELTA_J2000']
        except:
            pos2[:, 0], pos2[:, 1] = table_icat['sky_centroid_icrs.ra'],table_icat['sky_centroid_icrs.dec']

        ## cross-match with unique_dist, we will use pos1 as the left table and pos2 as the right table this time
        ## i.e. we will check  for every targets in pos1 to see whether it has a counterparts in pos2
        dist, ind = crossmatch.crossmatch_angular(pos1, pos2, max_distance=unique_dist / 3600.)
        matched = np.invert(np.isinf(dist)) # indices of pos1 that having counterparts in pos2
        for ikey in table_icat.keys():
            if len(table_icat[ikey].shape)==1:
                Table_Merged['{:}_{:}'.format(ikey,cat_ids[icat])] = np.zeros(len(Table_Merged),dtype=table_icat[ikey].dtype)
            elif len(table_icat[ikey].shape)==2:
                Table_Merged['{:}_{:}'.format(ikey,cat_ids[icat])] = np.zeros((len(Table_Merged),table_icat[ikey].shape[1]),dtype=table_icat[ikey].dtype)
            else:
                msgs.error('Column {:} format is not supported yet.'.format(ikey))
            Table_Merged['{:}_{:}'.format(ikey,cat_ids[icat])][matched] = table_icat[ikey][ind[matched]]

        Table_Merged['dist_merge_{:}'.format(cat_ids[icat])] = np.zeros(len(Table_Merged), dtype=dist.dtype)
        Table_Merged['dist_merge_{:}'.format(cat_ids[icat])][matched] = dist[matched]

    if outfile is not None:
        Table_Merged.write(outfile, format='fits', overwrite=True)

    return Table_Merged


def ForcedAperPhot(input_table, images, rmsmaps=None, flagmaps=None, phot_apertures=[1.0,2.0,3.0,4.0,5.0], image_ids=None,
                   flux_no_flagpix=False, effective_gains=None, zero_points=None, outfile=None):


    # If images is a string name, make it to a list
    if isinstance(images, str):
        images = [images]
    if isinstance(rmsmaps, str):
        rmsmaps = [rmsmaps]
    if isinstance(flagmaps, str):
        flagmaps = [flagmaps]

    if image_ids is None:
        image_ids = []
        image_ids_float = np.arange(len(images))+1
        for ii in range(len(images)):
            image_ids.append('{:02d}'.format(ii))

    if zero_points is not None:
        if np.size(zero_points) != len(images):
            msgs.error('The number of zero_points should be macthed with the number of input images.')

    ## Get the position
    try:
        ra, dec = input_table['RA'], input_table['DEC']
    except:
        ra, dec = input_table['ALPHA_J2000'], input_table['DELTA_J2000']

    ## setup the output table
    Table_Forced = input_table.copy()
    #Table_Forced = Table()
    #Table_Forced['RA'] = ra
    #Table_Forced['DEC'] = dec
    ## Let's perform the forced aperture photometry on each image
    positions = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

    ## Perform aperture photometry for your targets
    ## Loops over each images
    for ii, this_image in enumerate(images):
        msgs.info('Performing forced aperture photometry on {:}'.format(this_image))
        header, data, _ = io.load_fits(this_image)
        wcs_info = wcs.WCS(header)

        ## zero point
        if zero_points is not None:
            zpt = zero_points[ii]
        else:
            try:
                zpt = header['ZP']
            except:
                msgs.warn('Zero point was not given, set zpt=0')
                zpt = 0.

        ## effective gain which will be used for calculating total error
        if effective_gains is not None:
            gain = effective_gains[ii]
        else:
            try:
                gain = header['GAIN']
            except:
                msgs.warn('Effective gain was not given, set gain=1')
                gain = 1.0

        ## set up the mask
        if flagmaps is not None:
            ## good pixels with flag==0
            _, flag, _ = io.load_fits(flagmaps[ii])
            mask = flag > 0.
        else:
            mask = None

        ## set up error
        if rmsmaps is not None:
            _, error, _ = io.load_fits(rmsmaps[ii])
        else:
            ## ToDo: Make it a more careful error
            msgs.info('RMS map was not given, generating a rough error map from science image')
            _, error = BKG2D(data, 100, mask=mask, filter_size=(3,3), sigclip=5, back_maxiters=10,
                             back_type='median', back_rms_type='std')
            error[data==0.] = 0. # set zero pixels to be zero in error map

        if mask is None:
            mask = error<=0.

        ## Get the total error, i.e. including both background noise and photon noise
        total_error = calc_total_error(data, error, gain)

        ## Set up apertures
        apertures = [SkyCircularAperture(positions, r=d/2*u.arcsec) for d in phot_apertures]

        ## Get the Aperture flux with exact method
        if flux_no_flagpix:
            # Exclude flagged pixels when measure the flux
            tbl_aper = aperture_photometry(data, apertures, error=total_error, mask=mask, method='exact', wcs=wcs_info)
        else:
            # Include flagged pixels when measure the flux
            tbl_aper = aperture_photometry(data, apertures, error=total_error, mask=None, method='exact', wcs=wcs_info)
        flux_aper = np.zeros((len(tbl_aper), np.size(phot_apertures)))
        fluxerr_aper = np.zeros_like(flux_aper)
        mag_aper = np.zeros_like(flux_aper)
        magerr_aper = np.zeros_like(flux_aper)
        for jj in range(np.size(phot_apertures)):
            flux_aper[:,jj] = tbl_aper['aperture_sum_{:d}'.format(jj)]
            fluxerr_aper[:,jj] = tbl_aper['aperture_sum_err_{:d}'.format(jj)]
            mag_aper[:,jj] =  -2.5*np.log10(tbl_aper['aperture_sum_{:d}'.format(jj)])
            magerr_aper[:,jj] = 2.5/np.log(10)*tbl_aper['aperture_sum_err_{:d}'.format(jj)]/tbl_aper['aperture_sum_{:d}'.format(jj)]

        Table_Forced['FORCED_XCENTER_{:}'.format(image_ids[ii])] = tbl_aper['xcenter']+1*u.pix # to be consistent with SExtractor and ds9
        Table_Forced['FORCED_YCENTER_{:}'.format(image_ids[ii])] = tbl_aper['ycenter']+1*u.pix
        Table_Forced['FORCED_MAG_APER_{:}'.format(image_ids[ii])] = mag_aper + zpt
        Table_Forced['FORCED_MAGERR_APER_{:}'.format(image_ids[ii])] = magerr_aper
        Table_Forced['FORCED_FLUX_APER_{:}'.format(image_ids[ii])] = flux_aper
        Table_Forced['FORCED_FLUXERR_APER_{:}'.format(image_ids[ii])] = fluxerr_aper

        ## Change bad MAG to 999.
        badmag = np.isinf(mag_aper) | np.isnan(mag_aper)
        Table_Forced['FORCED_MAG_APER_{:}'.format(image_ids[ii])][badmag] = 999.
        Table_Forced['FORCED_MAGERR_APER_{:}'.format(image_ids[ii])][badmag] = 999.

        ## Set a mask bit for zero fluxes and change mag to zero as well
        zero_obj = (flux_aper == 0.)
        Table_Forced['FORCED_ZERO_APER_{:}'.format(image_ids[ii])] = zero_obj
        Table_Forced['FORCED_MAG_APER_{:}'.format(image_ids[ii])][zero_obj] = 0.
        Table_Forced['FORCED_MAGERR_APER_{:}'.format(image_ids[ii])][zero_obj] = 0.

        ## Get the FLAG 'FLUX' with center method
        if flagmaps is not None:
            bpm_zeros = (data==0.)
            if error is not None:
                bpm_zeros = bpm_zeros | (error==0.)
            flag += bpm_zeros*np.int(2**3) ## flag additional zero pixels. Not that this is consistent with the flag bit in procimg
            ## Get the FLAG 'flux'
            flag_tbl_aper = aperture_photometry(flag, apertures, error=None, mask=None, method='center', wcs=wcs_info)
            flag_aper = np.zeros((len(tbl_aper), np.size(phot_apertures)), dtype='int32')
            for jj in range(np.size(phot_apertures)):
                flag_aper[:, jj] = flag_tbl_aper['aperture_sum_{:d}'.format(jj)]

            ## Get the Number of bad pixels within each aperture
            nflag_tbl_aper = aperture_photometry((flag>0).astype('float'), apertures, error=None, mask=None, method='center', wcs=wcs_info)
            nflag_aper = np.zeros((len(tbl_aper), np.size(phot_apertures)), dtype='int32')
            for jj in range(np.size(phot_apertures)):
                nflag_aper[:, jj] = nflag_tbl_aper['aperture_sum_{:d}'.format(jj)]

            ## Assign it to to the final table
            Table_Forced['FORCED_FLAG_APER_{:}'.format(image_ids[ii])] = flag_aper
            Table_Forced['FORCED_NFLAG_APER_{:}'.format(image_ids[ii])] = nflag_aper

    ## Save the final table
    if outfile is not None:
        Table_Forced.write(outfile,format='fits', overwrite=True)

    return Table_Forced


def ForcedAperPhot_OLD(catalogs, images, rmsmaps, flagmaps, outfile=None, phot_apertures=[1.0,2.0,3.0,4.0,5.0], cat_ids=None, unique_dist=1.0):

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