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
from pyphot.query import query, ldac
from pyphot.photometry import mask_bright_star, photutils_detect
from pyphot.psf import  psf


def coadd(scifiles, flagfiles, ivarfiles, coaddroot, pixscale, science_path, coadddir, weight_type='MAP_WEIGHT',
          rescale_weights=False, combine_type='median', clip_ampfrac=0.3, clip_sigma=4.0, blank_badpixels=False,
          subtract_back= False, back_type='AUTO', back_default=0.0, back_size=100, back_filtersize=3,
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
                   "WEIGHT_TYPE": weight_type,"RESCALE_WEIGHTS": rescale,"BLANK_BADPIXELS":blank,
                   "COMBINE_TYPE": combine_type.upper(),"CLIP_AMPFRAC":clip_ampfrac,"CLIP_SIGMA":clip_sigma,
                   "SUBTRACT_BACK": subtract,"BACK_TYPE":back_type,"BACK_DEFAULT":back_default,"BACK_SIZE":back_size,
                   "BACK_FILTERSIZE":back_filtersize,"BACK_FILTTHRESH":back_filtthresh, "RESAMPLING_TYPE":resampling_type}
    ## Run swarp for science image
    msgs.info('Coadding science images.')
    swarp.run_swarp(scifiles, config=swarpconfig, workdir=science_path, defaultconfig='pyphot',
                    coadddir=coadddir, coaddroot=coaddroot + '_sci', delete=delete, log=log)

    ## parameters for coadding flag images
    swarpconfig_flag = swarpconfig.copy()
    swarpconfig_flag['WEIGHT_TYPE'] = "NONE"
    swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
    swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
    ## Run swarp for flag image
    msgs.info('Coadding flag images.')
    swarp.run_swarp(flagfiles, config=swarpconfig_flag, workdir=science_path, defaultconfig='pyphot',
                    coadddir=coadddir, coaddroot=coaddroot + '_flag', delete=True, log=False)

    ## parameters for coadding inverse variance images
    swarpconfig_ivar = swarpconfig.copy()
    ivarlist = os.path.join(coadddir,'{:}_ivar.list'.format(coaddroot))
    swarpconfig_ivar['WEIGHT_IMAGE'] = '@{:}'.format(ivarlist)
    textfile = open(ivarlist, "w")
    for this_ivar in ivarfiles:
        textfile.write(this_ivar + "\n")
    textfile.close()

    ## Run swarp for inverse variance image
    msgs.info('Generating inverse variance image for coadded science image.')
    swarp.run_swarp(scifiles, config=swarpconfig_ivar, workdir=science_path, defaultconfig='pyphot',
                    coadddir=coadddir, coaddroot=coaddroot + '_ivar', delete=True, log=False)
    os.system('mv {:} {:}'.format(os.path.join(coadddir, coaddroot + '_ivar.weight.fits'),
                                  os.path.join(coadddir, coaddroot + '_ivar.fits')))
    # delete unnecessary files
    os.system('rm {:}'.format(ivarlist))
    #os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_ivar.swarp.xml')))
    #os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.swarp.xml')))
    os.system('rm {:}'.format(os.path.join(coadddir, coaddroot + '_flag.weight.fits')))

    # useful file names
    coadd_file = os.path.join(coadddir,coaddroot+'_sci.fits')
    coadd_wht_file = os.path.join(coadddir, coaddroot + '_sci.weight.fits')
    coadd_flag_file = os.path.join(coadddir, coaddroot + '_flag.fits')
    coadd_ivar_file = os.path.join(coadddir, coaddroot + '_ivar.fits')

    # ToDO: Subtract background for the final coadded image?

    return coadd_file, coadd_wht_file, coadd_flag_file, coadd_ivar_file

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
                effective_gain = header['GAIN']
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
                      "GAIN": effective_gain,
                      "PHOT_APERTURES": np.array(phot_apertures) * utils.inverse(pixscale)}
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
        #       Currently we only remove saturated targets, catalog['NIMAFLAGS_ISO'] & 2**2<1 used for remove saturated targets, see procimg.detproc
        if external_flag:
            #flag = (catalog['NIMAFLAGS_ISO'] & 2**2<1) & (catalog['IMAFLAGS_ISO'] & 2**2<1) #& (catalog['IMAFLAGS_ISO']<1)  #(catalog['NIMAFLAGS_ISO']<1)
            flag = (catalog['IMAFLAGS_ISO']<1) & (catalog['NIMAFLAGS_ISO']<1)
        else:
            flag = np.ones_like(catalog['FLAGS'], dtype='bool')
        good_cat = (catalog['FLAGS']<1) & flag
        #& (catalog['CLASS_STAR']>0.9) & (catalog['NIMAFLAGS_ISO']<1)
        good_cat &= catalog['FLUX_AUTO'] * utils.inverse(catalog['FLUXERR_AUTO']) > 10
        catalog = catalog[good_cat]
        ra, dec = catalog['ALPHA_J2000'], catalog['DELTA_J2000']
    except:
        if verbose:
            msgs.info('Reading SExtractor catalog failed. Reading photoutils catalog')
        catalog = Table.read(catalogfits)
        good_cat = catalog['FLUX_AUTO'] * utils.inverse(catalog['FLUXERR_AUTO']) > 10
        catalog = catalog[good_cat]
        ra, dec = catalog['sky_centroid_icrs.ra'], catalog['sky_centroid_icrs.dec']

    if len(ra)>0:
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
            dist_cen = np.sqrt(((ref_ra_cen-ra_cen)*np.cos(dec_cen/180.*np.pi))**2 + (ref_dec_ren-dec_cen)**2)
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
            good_ref = (1.0857 * utils.inverse(ref_data['{:}_MAG_ERR'.format(primary)]) > 10)
            if coefficients[1]*coefficients[2] !=0:
                good_ref &= (1.0857 * utils.inverse(ref_data['{:}_MAG_ERR'.format(secondary)]) > 10)
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
    else:
        nstar=0
        msgs.warn('No stars were found in the input catalog {:}'.format(catalogfits))

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
        FLXSCALE = 1.0 * utils.inverse(par[0].header['EXPTIME'])
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

    sci_fits_all = np.zeros_like(np.array(sci_fits_list))
    zp_all, zp_std_all = np.zeros(n_file), np.zeros(n_file)
    nstar_all = np.zeros(n_file)
    fwhm_all = np.zeros(n_file)
    idx_all = np.zeros(n_file)

    if n_process == 1:
        for ii in range(n_file):
            zp_this, zp_this_std, nstar, fwhm = _cal_chip(cat_fits_list[ii], sci_fits=sci_fits_list[ii],
                        ref_fits=ref_fits_list[ii], outqa_root=outqa_root_list[ii],
                        ZP=ZP, external_flag=external_flag, refcatalog=refcatalog, primary=primary, secondary=secondary,
                        coefficients=coefficients, nstar_min=nstar_min, pixscale=pixscale, verbose=verbose)
            idx_all[ii] = ii
            sci_fits_all[ii] = sci_fits_list[ii]
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
            work_queue.put((ii, cat_fits_list[ii], sci_fits_list[ii], ref_fits_list[ii], outqa_root_list[ii]))

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
            idx_this, sci_fits_this, zp_this, zp_this_std, nstar, fwhm = done_queue.get()
            idx_all[ii] = idx_this
            sci_fits_all[ii] = sci_fits_this
            zp_all[ii] = zp_this
            zp_std_all[ii] = zp_this_std
            nstar_all[ii] = nstar
            fwhm_all[ii] = fwhm
            ii +=1

    # sort the data based on input. This is necessary for multiproccessing at least. I do this for both way just in case.
    zp_all_sort = zp_all[np.argsort(idx_all)]
    zp_std_all_sort = zp_std_all[np.argsort(idx_all)]
    nstar_all_sort = nstar_all[np.argsort(idx_all)]
    fwhm_all_sort = fwhm_all[np.argsort(idx_all)]
    sci_fits_all_sort = sci_fits_all[np.argsort(idx_all)]

    '''
    zp_all_sort, zp_std_all_sort = np.zeros(n_file), np.zeros(n_file)
    nstar_all_sort = np.zeros(n_file)
    fwhm_all_sort = np.zeros(n_file)
    sci_fits_all_sort = np.copy(sci_fits_all)

    for ii, ifile in enumerate(sci_fits_list):
        this_idx = sci_fits_all==ifile
        sci_fits_all_sort[ii] = sci_fits_all[this_idx][0]
        zp_all_sort[ii] = zp_all[this_idx]
        zp_std_all_sort[ii] = zp_std_all[this_idx]
        nstar_all_sort[ii] = nstar_all[this_idx]
        fwhm_all_sort[ii] = fwhm_all[this_idx]
    '''

    return zp_all_sort, zp_std_all_sort, nstar_all_sort, fwhm_all_sort

def _cal_chip_worker(work_queue, done_queue, ZP=25.0, external_flag=True, refcatalog='Panstarrs',
                     primary='i', secondary='z', coefficients=[0.,0.,0.], nstar_min=10, pixscale=None, verbose=False):

    """Multiprocessing worker for cal_chips."""
    while not work_queue.empty():
        idx, cat_fits, sci_fits, ref_fits, outqa_root = work_queue.get()
        zp_this, zp_this_std, nstar, fwhm = _cal_chip(cat_fits, sci_fits=sci_fits, ref_fits=ref_fits, outqa_root=outqa_root,
                ZP=ZP, external_flag=external_flag, refcatalog=refcatalog, primary=primary, secondary=secondary,
                coefficients=coefficients, nstar_min=nstar_min, pixscale=pixscale, verbose=verbose)

        done_queue.put((idx, sci_fits, zp_this, zp_this_std, nstar, fwhm))



class PostProc():

    def __init__(self, par, detectors, setup_id, scifiles, coadd_ids, sci_ra, sci_dec, sci_airmass, sci_exptime,
                 sci_filter, sci_target, pixscale, science_path='./', qa_path='./', coadd_path='./',
                 overwrite=True, reuse_masters=True):
        '''

        Parameters
        ----------
        par
        detectors
        setup_id
        scifiles
        coadd_ids
        sci_ra
        sci_dec
        pixscale
        science_path
        qa_path
        '''

        self.par = par
        self.detectors = detectors
        self.setup_id = setup_id
        self.ndet = len(detectors)
        self.scifiles = scifiles
        self.coadd_ids = coadd_ids
        self.sci_ra = sci_ra.data.astype('float')
        self.sci_dec = sci_dec.data.astype('float')
        self.sci_airmass = sci_airmass.data.astype('float')
        self.sci_exptime = sci_exptime.data.astype('float')
        self.sci_filter = sci_filter
        self.sci_target = sci_target
        self.pixscale = pixscale
        self.science_path = science_path
        self.qa_path = qa_path
        self.coadd_path = coadd_path
        self.overwrite = overwrite
        self.reuse_masters = reuse_masters
        if self.ndet == 1:
            self.mosaic = False #ToDo: Might deprecate this in the future since mosaic works good for everything.
        else:
            self.mosaic = self.par['postproc']['astrometry']['mosaic']

        self.n_process = par['rdx']['n_process']
        # Initial n_process
        n_file = len(scifiles)
        n_cpu = multiprocessing.cpu_count()
        if self.n_process > n_cpu:
            self.n_process = n_cpu
        if self.n_process > n_file:
            self.n_process = n_file

        # SExtractor task in your terminal
        self.sextask = self.par['rdx']['sextractor']
        self.conv = self.par['postproc']['detection']['conv'] # convolve template used by SExtractor

        # Other parameters
        self.skip_swarp_align = self.par['postproc']['astrometry']['skip_swarp_align']
        self.match_flipped = self.par['postproc']['astrometry']['match_flipped']
        self.solve_photom_scamp = self.par['postproc']['astrometry']['solve_photom_scamp']
        self.group = self.par['postproc']['astrometry']['group']
        self.delete = self.par['postproc']['astrometry']['delete']
        self.log = self.par['postproc']['astrometry']['log']
        self.verbose = True

        # Load parameters for SExtractor
        detect_thresh = self.par['postproc']['astrometry']['detect_thresh']
        analysis_thresh = self.par['postproc']['astrometry']['analysis_thresh']
        detect_minarea = self.par['postproc']['astrometry']['detect_minarea']
        # Load parameters for Scamp
        self.astref_catalog = self.par['postproc']['astrometry']['astref_catalog']
        self.astrefmag_limits = self.par['postproc']['astrometry']['astrefmag_limits']
        self.astrefsn_limits = self.par['postproc']['astrometry']['astrefsn_limits']
        crossid_radius = self.par['postproc']['astrometry']['crossid_radius']
        astref_band = self.par['postproc']['astrometry']['astref_band']
        position_maxerr = self.par['postproc']['astrometry']['position_maxerr']
        pixscale_maxerr = self.par['postproc']['astrometry']['pixscale_maxerr']
        posangle_maxerr = self.par['postproc']['astrometry']['posangle_maxerr']
        distort_degrees = self.par['postproc']['astrometry']['distort_degrees']
        stability_type = self.par['postproc']['astrometry']['stability_type']
        mosaic_type = self.par['postproc']['astrometry']['mosaic_type']
        # Load parameters for Swarp
        weight_type = self.par['postproc']['astrometry']['weight_type']

        # Configuration for SExtractor
        self.sexconfig = {"CHECKIMAGE_TYPE": "NONE", "WEIGHT_TYPE": "MAP_WEIGHT", "CATALOG_TYPE": "FITS_LDAC",
                          "DETECT_THRESH":detect_thresh,"ANALYSIS_THRESH": analysis_thresh,"DETECT_MINAREA": detect_minarea}
        # Parameters for SExtractor
        self.sexparams = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE',
                          'ERRTHETAWIN_IMAGE', 'ALPHA_J2000', 'DELTA_J2000', 'ISOAREAF_IMAGE', 'ISOAREA_IMAGE',
                          'ELLIPTICITY', 'ELONGATION', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO',
                          'MAG_APER', 'MAGERR_APER', 'FLUX_RADIUS', 'IMAFLAGS_ISO', 'NIMAFLAGS_ISO', 'CLASS_STAR',
                          'FLAGS', 'FLAGS_WEIGHT']

        # Configuration for Scamp
        if self.solve_photom_scamp:
            SOLVE_PHOTOM = 'Y'
        else:
            SOLVE_PHOTOM = 'N'
        if self.match_flipped:
            MATCH_FLIPPED = 'Y'
        else:
            MATCH_FLIPPED = 'N'
        if self.astrefmag_limits is not None:
            ASTREFMAG_LIMITS = '{:},{:}'.format(self.astrefmag_limits[0], self.astrefmag_limits[1])
        else:
            ASTREFMAG_LIMITS = '-99.0,99.0'
        if self.astrefsn_limits is not None:
            SN_THRESHOLDS = '{:},{:}'.format(self.astrefsn_limits[0], self.astrefsn_limits[1])
        else:
            SN_THRESHOLDS = '10.0,100.0'

        self.scamp_supported_cat = ['USNO-A2','USNO-B1','GSC-2.3','TYCHO-2','UCAC-4','URAT-1','NOMAD-1','PPMX',
                'CMC-15','2MASS', 'DENIS-3', 'SDSS-R9','SDSS-R12','IGSL','GAIA-DR1','GAIA-DR2','GAIA-EDR3',
                'PANSTARRS-1','ALLWISE']
        self.pyphot_supported_cat = ['LS-DR9','DES-DR2']
        if self.astref_catalog in self.scamp_supported_cat:
            self.scampconfig = {"CROSSID_RADIUS": crossid_radius,
                                "ASTREF_CATALOG": self.astref_catalog,
                                "ASTREF_BAND": astref_band,
                                "ASTREFMAG_LIMITS": ASTREFMAG_LIMITS,
                                "SN_THRESHOLDS": SN_THRESHOLDS,
                                "POSITION_MAXERR": position_maxerr,
                                "PIXSCALE_MAXERR": pixscale_maxerr,
                                "POSANGLE_MAXERR": posangle_maxerr,
                                "STABILITY_TYPE": stability_type,
                                "MOSAIC_TYPE": mosaic_type,
                                "SOLVE_PHOTOM": SOLVE_PHOTOM,
                                "MATCH_FLIPPED": MATCH_FLIPPED,
                                "DISTORT_DEGREES": distort_degrees,
                                "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                                "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
        elif self.astref_catalog in self.pyphot_supported_cat:
            astrefcat_name = self.par['postproc']['astrometry']['astrefcat_name']
            astrefcent_keys = 'RA, DEC'
            astreferr_keys = 'RA_ERR, DEC_ERR, THETA_ERR'
            astrefmag_key = astref_band+'_MAG'
            astrefmagerr_key = astref_band+'_MAG_ERR'
            self.scampconfig = {"CROSSID_RADIUS": crossid_radius,
                                "ASTREF_CATALOG": 'FILE',
                                "ASTREF_BAND": 'DEFAULT',
                                "ASTREFMAG_LIMITS": ASTREFMAG_LIMITS,
                                "ASTREFCAT_NAME":astrefcat_name,
                                "ASTREFCENT_KEYS":astrefcent_keys,
                                "ASTREFERR_KEYS":astreferr_keys,
                                "ASTREFMAG_KEY":astrefmag_key,
                                "ASTREFMAGERR_KEY":astrefmagerr_key,
                                "POSITION_MAXERR": position_maxerr,
                                "PIXSCALE_MAXERR": pixscale_maxerr,
                                "POSANGLE_MAXERR": posangle_maxerr,
                                "STABILITY_TYPE": stability_type,
                                "MOSAIC_TYPE": mosaic_type,
                                "SOLVE_PHOTOM": SOLVE_PHOTOM,
                                "MATCH_FLIPPED": MATCH_FLIPPED,
                                "DISTORT_DEGREES": distort_degrees,
                                "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                                "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}
        else:
            msgs.warn('The astrometric reference catalog is not supported by SCAMP or PyPhot. Make sure'
                      'to set related keyworlds properly.')
            astrefcat_name = self.par['postproc']['astrometry']['astrefcat_name']
            astrefcent_keys = ','.join(str(e) for e in self.par['postproc']['astrometry']['astrefcent_keys'])
            astreferr_keys = ','.join(str(e) for e in self.par['postproc']['astrometry']['astreferr_keys'])
            astrefmag_key = self.par['postproc']['astrometry']['astrefmag_key']
            astrefmagerr_key = self.par['postproc']['astrometry']['astrefmagerr_key']
            self.scampconfig = {"CROSSID_RADIUS": crossid_radius,
                                "ASTREF_CATALOG": 'FILE',
                                "ASTREF_BAND": 'DEFAULT',
                                "ASTREFMAG_LIMITS": ASTREFMAG_LIMITS,
                                "ASTREFCAT_NAME":astrefcat_name,
                                "ASTREFCENT_KEYS":astrefcent_keys,
                                "ASTREFERR_KEYS":astreferr_keys,
                                "ASTREFMAG_KEY":astrefmag_key,
                                "ASTREFMAGERR_KEY":astrefmagerr_key,
                                "POSITION_MAXERR": position_maxerr,
                                "PIXSCALE_MAXERR": pixscale_maxerr,
                                "POSANGLE_MAXERR": posangle_maxerr,
                                "STABILITY_TYPE": stability_type,
                                "MOSAIC_TYPE": mosaic_type,
                                "SOLVE_PHOTOM": SOLVE_PHOTOM,
                                "MATCH_FLIPPED": MATCH_FLIPPED,
                                "DISTORT_DEGREES": distort_degrees,
                                "CHECKPLOT_TYPE": 'ASTR_REFERROR1D,ASTR_REFERROR2D,FGROUPS,DISTORTION',
                                "CHECKPLOT_NAME": 'astr_referror1d,astr_referror2d,fgroups,distort'}


        ## Configuration for Swarp
        self.swarpconfig = {"RESAMPLE": "Y", "DELETE_TMPFILES": "Y", "CENTER_TYPE": "ALL", "PIXELSCALE_TYPE": "MANUAL",
                            "PIXEL_SCALE": self.pixscale, "SUBTRACT_BACK": "N", "COMBINE_TYPE": "MEDIAN", "GAIN_DEFAULT": 1.0,
                            "RESAMPLE_SUFFIX": ".resamp.fits", "WEIGHT_TYPE": weight_type,
                            "RESAMPLING_TYPE": 'NEAREST',
                            # I would always set this to NEAREST for individual exposures to avoid interpolation
                            "HEADER_SUFFIX": "_cat.head"}
        # configuration for swarp the flag image
        self.swarpconfig_flag = self.swarpconfig.copy()
        self.swarpconfig_flag['WEIGHT_TYPE'] = 'NONE'
        self.swarpconfig_flag['COMBINE_TYPE'] = 'SUM'
        self.swarpconfig_flag['RESAMPLING_TYPE'] = 'FLAGS'
        # configuration for swarp the inverse variance image
        self.swarpconfig_ivar = self.swarpconfig.copy()
        self.swarpconfig_ivar['WEIGHT_TYPE'] = 'NONE'

        ## Configuration for photometric calibration
        self.photref_catalog = self.par['postproc']['photometry']['photref_catalog']
        self.primary = self.par['postproc']['photometry']['primary']
        self.secondary = self.par['postproc']['photometry']['secondary']
        self.coefficients = self.par['postproc']['photometry']['coefficients']
        self.zeropoint = self.par['postproc']['photometry']['zpt']
        self.nstar_min = self.par['postproc']['photometry']['nstar_min']
        self.external_flag = self.par['postproc']['photometry']['external_flag']

        ## Prepare lists
        self.sci_proc_list, self.ivar_proc_list, self.wht_proc_list, self.flag_proc_list = [], [], [], []
        self.sci_resample_list, self.ivar_resample_list, self.wht_resample_list, self.flag_resample_list = [], [], [], []
        self.cat_proc_list = []
        self.cat_resample_list = []
        self.proc_coadd_ids = []
        self.resamp_coadd_ids = []
        self.resamp_det_ids = []
        self.master_astref_cats = []  # reference catalog for astronometric calibrations,
                                      # The astrometric reference catalog is only used when the reference is not supported by scamp
        self.master_phoref_cats = []  # reference catalog for photometric calibrations
        self.outqa_list = []  # for ploting resampled images

        ## Coadd root names
        self.coadd_root_ids = []
        self.coadd_root_names = []
        groups = np.unique(self.coadd_ids.data)
        for igroup in groups:
            this_target = self.sci_target[self.coadd_ids==igroup][0]
            this_filter = self.sci_filter[self.coadd_ids==igroup][0]
            this_coadd_root = '{:}_{:}_coadd_ID{:03d}'.format(this_target, this_filter, igroup)
            self.coadd_root_ids.append(igroup)
            self.coadd_root_names.append(this_coadd_root)

        if self.mosaic:
            # Prepare list for mosaic mode
            for ii, ifile in enumerate(self.scifiles):
                # Save MEF files
                rootname = os.path.join(self.science_path, os.path.basename(ifile))
                sci_proc_file = io.build_mef(rootname, self.detectors, img_type='SCI', returnname_only=True)
                ivar_proc_file = io.build_mef(rootname, self.detectors, img_type='IVAR', returnname_only=True)
                wht_proc_file = io.build_mef(rootname, self.detectors, img_type='WEIGHT', returnname_only=True)
                flag_proc_file = io.build_mef(rootname, self.detectors, img_type='FLAG', returnname_only=True)
                self.sci_proc_list.append(sci_proc_file)
                self.ivar_proc_list.append(ivar_proc_file)
                self.wht_proc_list.append(wht_proc_file)
                self.flag_proc_list.append(flag_proc_file)
                self.cat_proc_list.append(sci_proc_file.replace('.fits','_cat.fits'))
                self.proc_coadd_ids.append(self.coadd_ids[ii])
                for jj, idet in enumerate(self.detectors):
                    sci_resample_file = sci_proc_file.replace('.fits', '.{:04d}.resamp.fits'.format(jj+1))
                    wht_resample_file = sci_proc_file.replace('.fits', '.{:04d}.resamp.weight.fits'.format(jj+1))
                    ivar_resample_file = sci_proc_file.replace('.fits', '.ivar.{:04d}.resamp.fits'.format(jj+1))
                    flag_resample_file = sci_proc_file.replace('sci.fits', 'flag.{:04d}.resamp.fits'.format(jj+1))
                    cat_resample_file = sci_proc_file.replace('.fits', '.{:04d}.resamp_cat.fits'.format(jj+1))
                    if (self.par['rdx']['skip_astrometry']) and not (os.path.exists(sci_resample_file)):
                        self.sci_resample_list.append(rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet)))
                        self.ivar_resample_list.append(rootname.replace('.fits', '_det{:02d}_sci.ivar.fits'.format(idet)))
                        self.wht_resample_list.append(rootname.replace('.fits', '_det{:02d}_sci.weight.fits'.format(idet)))
                        self.flag_resample_list.append(rootname.replace('.fits', '_det{:02d}_flag.fits'.format(idet)))
                        self.cat_resample_list.append(rootname.replace('.fits', '_det{:02d}_sci_cat.fits'.format(idet)))
                        this_qa = os.path.basename(rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet))).replace('.fits', '')
                    else:
                        self.sci_resample_list.append(sci_resample_file)
                        self.wht_resample_list.append(wht_resample_file)
                        self.ivar_resample_list.append(ivar_resample_file)
                        self.flag_resample_list.append(flag_resample_file)
                        self.cat_resample_list.append(cat_resample_file)
                        this_qa = os.path.basename(sci_resample_file).replace('.fits', '')
                    self.outqa_list.append(os.path.join(self.qa_path, this_qa))
                    this_astcat = 'MasterAstRefCat_{:}_{:}_ID{:03d}.fits'.format(self.astref_catalog, self.setup_id,
                                                                                  self.coadd_ids[ii])
                    this_phocat = 'MasterPhoRefCat_{:}_{:}_ID{:03d}.fits'.format(self.photref_catalog, self.setup_id,
                                                                                  self.coadd_ids[ii])
                    self.master_astref_cats.append(os.path.join(self.par['calibrations']['master_dir'], this_astcat))
                    self.master_phoref_cats.append(os.path.join(self.par['calibrations']['master_dir'], this_phocat))
                    self.resamp_coadd_ids.append(self.coadd_ids[ii])
                    self.resamp_det_ids.append(idet)
        else:
            # Prepare list for non-mosaic mode
            for ii, ifile in enumerate(self.scifiles):
                rootname = os.path.join(self.science_path, os.path.basename(ifile))
                for idet in self.detectors:
                    sci_proc_file = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet))
                    ivar_proc_file = rootname.replace('.fits', '_det{:02d}_sci.ivar.fits'.format(idet))
                    wht_proc_file = rootname.replace('.fits', '_det{:02d}_sci.weight.fits'.format(idet))
                    flag_proc_file = rootname.replace('.fits', '_det{:02d}_flag.fits'.format(idet))
                    cat_proc_file = rootname.replace('.fits', '_det{:02d}_sci_cat.fits'.format(idet))
                    self.sci_proc_list.append(sci_proc_file)
                    self.ivar_proc_list.append(ivar_proc_file)
                    self.wht_proc_list.append(wht_proc_file)
                    self.flag_proc_list.append(flag_proc_file)
                    self.cat_proc_list.append(cat_proc_file)

                    sci_resample_file = rootname.replace('.fits', '_det{:02d}_sci.resamp.fits'.format(idet))
                    ivar_resample_file = rootname.replace('.fits', '_det{:02d}_sci.ivar.resamp.fits'.format(idet))
                    wht_resample_file = rootname.replace('.fits', '_det{:02d}_sci.resamp.weight.fits'.format(idet))
                    flag_resample_file = rootname.replace('.fits', '_det{:02d}_flag.resamp.fits'.format(idet))
                    cat_resample_file = rootname.replace('.fits', '_det{:02d}_sci.resamp_cat.fits'.format(idet))

                    if (self.par['rdx']['skip_astrometry']) and not (os.path.exists(sci_resample_file)):
                        self.sci_resample_list.append(sci_proc_file)
                        self.ivar_resample_list.append(ivar_proc_file)
                        self.wht_resample_list.append(wht_proc_file)
                        self.flag_resample_list.append(flag_proc_file)
                        self.cat_resample_list.append(cat_proc_file)
                        this_qa = os.path.basename(sci_proc_file).replace('.fits', '')
                    else:
                        self.sci_resample_list.append(sci_resample_file)
                        self.ivar_resample_list.append(ivar_resample_file)
                        self.wht_resample_list.append(wht_resample_file)
                        self.flag_resample_list.append(flag_resample_file)
                        self.cat_resample_list.append(cat_resample_file)
                        this_qa = os.path.basename(sci_resample_file).replace('.fits', '')

                    self.outqa_list.append(os.path.join(self.qa_path, this_qa))
                    this_astcat = 'MasterAstRefCat_{:}_{:}_ID{:03d}.fits'.format(self.astref_catalog, self.setup_id,
                                                                                  self.coadd_ids[ii])
                    this_phocat = 'MasterPhoRefCat_{:}_{:}_ID{:03d}.fits'.format(self.photref_catalog, self.setup_id,
                                                                                  self.coadd_ids[ii])
                    self.master_astref_cats.append(os.path.join(self.par['calibrations']['master_dir'], this_astcat))
                    self.master_phoref_cats.append(os.path.join(self.par['calibrations']['master_dir'], this_phocat))
                    self.proc_coadd_ids.append(self.coadd_ids[ii])
                    self.resamp_coadd_ids.append(self.coadd_ids[ii])
                    self.resamp_det_ids.append(idet)

        self.proc_coadd_ids = np.array(self.proc_coadd_ids, dtype=self.coadd_ids.dtype)
        self.resamp_coadd_ids = np.array(self.resamp_coadd_ids, dtype=self.coadd_ids.dtype)
        self.resamp_det_ids = np.array(self.resamp_det_ids, dtype=self.coadd_ids.dtype)

    def run_astrometry(self):
        '''

        Returns
        -------

        '''
        if not self.skip_swarp_align:
            # Prepare list for mosaic mode
            sci_proc_list, ivar_proc_list, wht_proc_list, flag_proc_list = [], [], [], []
            if self.mosaic:
                for ii, ifile in enumerate(self.scifiles):
                    rootname = os.path.join(self.science_path, os.path.basename(ifile))
                    for idet in self.detectors:
                        sci_proc_file = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet))
                        ivar_proc_file = rootname.replace('.fits', '_det{:02d}_sci.ivar.fits'.format(idet))
                        wht_proc_file = rootname.replace('.fits', '_det{:02d}_sci.weight.fits'.format(idet))
                        flag_proc_file = rootname.replace('.fits', '_det{:02d}_flag.fits'.format(idet))
                        cat_proc_file = rootname.replace('.fits', '_det{:02d}_sci_cat.fits'.format(idet))
                        sci_proc_list.append(sci_proc_file)
                        ivar_proc_list.append(ivar_proc_file)
                        wht_proc_list.append(wht_proc_file)
                        flag_proc_list.append(flag_proc_file)
            else:
                sci_proc_list = self.sci_proc_list
                ivar_proc_list = self.ivar_proc_list
                wht_proc_list = self.wht_proc_list
                flag_proc_list = self.flag_proc_list

            ## This step is to align the image to N to the up and E to the left.
            ## It might be necessary if your image has a ~180 degree from the nominal orientation.
            msgs.info('Running Swarp to rotate the science image.')
            swarpconfig0 = self.swarpconfig.copy()
            swarpconfig0['RESAMPLE_SUFFIX'] = '.fits' # Replace the original image
            swarp.run_swarp(sci_proc_list, config=swarpconfig0, workdir=self.science_path, defaultconfig='pyphot',
                            n_process=self.n_process, delete=self.delete, log=self.log, verbose=False)
            # resample inverse variance image
            msgs.info('Running Swarp to rotate the inverse variance image.')
            swarpconfig_ivar0 = self.swarpconfig_ivar.copy()
            swarpconfig_ivar0['RESAMPLE_SUFFIX'] = '.fits'
            swarp.run_swarp(ivar_proc_list, config=swarpconfig_ivar0, workdir=self.science_path, defaultconfig='pyphot',
                            n_process=self.n_process, delete=self.delete, log=False, verbose=False)
            # resample flag image
            msgs.info('Running Swarp to rotate the flag image.')
            swarpconfig_flag0 = self.swarpconfig_flag.copy()
            swarpconfig_flag0['RESAMPLE_SUFFIX'] = '.fits'
            swarp.run_swarp(flag_proc_list, config=swarpconfig_flag0, workdir=self.science_path, defaultconfig='pyphot',
                            n_process=self.n_process, delete=self.delete, log=False, verbose=False)

            # remove useless data and change flag image type
            for ii, flag_fits in enumerate(flag_proc_list):
                ## It seems swarp does not change the dtype now therefore I do not need the following.
                #par = fits.open(flag_fits, memmap=False)
                #par[0].data = par[0].data.astype('int32')  # change the dtype to be int32
                #par[0].writeto(flag_fits, overwrite=True)
                #del par[0].data
                #par.close()
                #gc.collect()
                os.system('rm {:}'.format(flag_fits.replace('.fits', '.weight.fits')))
                os.system('rm {:}'.format(ivar_proc_list[ii].replace('.fits', '.weight.fits')))

        if self.mosaic:
            msgs.info('Build MEF format fits files')
            rootnames = []
            for ifile in self.scifiles:
                # Save MEF files
                rootname = os.path.join(self.science_path, os.path.basename(ifile))
                rootnames.append(rootname)
                #sci_proc_file = io.build_mef(rootname, self.detectors, img_type='SCI', returnname_only=False,
                #                             overwrite=self.overwrite)
                #ivar_proc_file = io.build_mef(rootname, self.detectors, img_type='IVAR', returnname_only=False,
                #                              overwrite=self.overwrite)
                #wht_proc_file = io.build_mef(rootname, self.detectors, img_type='WEIGHT', returnname_only=False,
                #                             overwrite=self.overwrite)
                #flag_proc_file = io.build_mef(rootname, self.detectors, img_type='FLAG', returnname_only=False,
                #                              overwrite=self.overwrite)
            ## Build MEF format fits files in parallel
            sci_proc_files = io.build_mef_parallel(rootnames, detectors=self.detectors, img_type='SCI',
                                                   returnname_only=False, overwrite=self.overwrite, n_process=self.n_process)
            ivar_proc_files = io.build_mef_parallel(rootnames, detectors=self.detectors, img_type='IVAR',
                                                   returnname_only=False, overwrite=self.overwrite, n_process=self.n_process)
            wht_proc_files = io.build_mef_parallel(rootnames, detectors=self.detectors, img_type='WEIGHT',
                                                   returnname_only=False, overwrite=self.overwrite, n_process=self.n_process)
            flag_proc_files = io.build_mef_parallel(rootnames, detectors=self.detectors, img_type='FLAG',
                                                   returnname_only=False, overwrite=self.overwrite, n_process=self.n_process)


        ## Step one: extract catalog from sciproc files
        msgs.info('Running SExtractor for the first pass to extract catalog used for SCAMP.')
        sex.run_sex(self.sci_proc_list, flag_image_list=self.flag_proc_list, weight_image_list=self.wht_proc_list,
                    n_process=self.n_process, task=self.sextask, config=self.sexconfig, workdir=self.science_path, params=self.sexparams,
                    defaultconfig='pyphot', conv=self.conv, nnw=None, dual=False, delete=self.delete, log=self.log, verbose=self.verbose)

        ## Step two: run scamp on extracted catalog
        #ToDo: The scamp part need to be grouped if use SCAMP non-supported reference catalogs.
        #ToDo: Group them by calibID?
        groups = np.unique(self.coadd_ids.data)
        for igroup in groups:
            this_group = self.proc_coadd_ids == igroup
            this_cat_proc_list = np.array(self.cat_proc_list)[this_group].tolist()

            if self.astref_catalog in self.pyphot_supported_cat:
                self.scampconfig['ASTREFCAT_NAME'] = self.get_astref_cat(igroup)

            if self.mosaic and self.scampconfig['MOSAIC_TYPE'] !='LOOSE':
                msgs.info('Running SCAMP for the first loop with LOOSE mode.')
                scampconfig0 = self.scampconfig.copy()
                scampconfig0['MOSAIC_TYPE'] = 'LOOSE'
                scamp.run_scamp(this_cat_proc_list, config=scampconfig0, workdir=self.science_path, QAdir=self.qa_path,
                                n_process=self.n_process, defaultconfig='pyphot', group=self.group, delete=self.delete,
                                log=self.log, verbose=self.verbose)
                # Copy .head to .ahead
                for ii, icat in enumerate(this_cat_proc_list):
                    os.system('mv {:} {:}'.format(icat.replace('.fits', '.head'),
                                                  icat.replace('.fits', '.ahead')))
                msgs.info('Running SCAMP for the second loop with {:} mode.'.format(self.scampconfig['MOSAIC_TYPE']))
                scamp.run_scamp(this_cat_proc_list, config=self.scampconfig, workdir=self.science_path, QAdir=self.qa_path,
                                n_process=self.n_process, defaultconfig='pyphot', group=self.group, delete=self.delete,
                                log=self.log, verbose=self.verbose)
                for ii, icat in enumerate(this_cat_proc_list):
                    # remove .ahead
                    os.system('rm {:}'.format(icat.replace('.fits', '.ahead')))
            else:
                msgs.info('Running SCAMP to solve the astronometric solutions.')
                scamp.run_scamp(this_cat_proc_list, config=self.scampconfig, workdir=self.science_path, QAdir=self.qa_path,
                                n_process=self.n_process, defaultconfig='pyphot', group=self.group, delete=self.delete,
                                log=self.log, verbose=self.verbose)

        ## Step three: run swarp to resample sciproc images
        # resample science image
        msgs.info('Running Swarp to resample science images.')
        swarp.run_swarp(self.sci_proc_list, config=self.swarpconfig, workdir=self.science_path, defaultconfig='pyphot',
                        n_process=self.n_process, delete=self.delete, log=self.log, verbose=self.verbose)
        #swarp.run_swarp(self.sci_proc_list, config=self.swarpconfig, workdir=self.science_path, defaultconfig='pyphot',
        #                n_process=self.n_process, delete=self.delete, log=self.log, verbose=self.verbose,
        #                coadddir='./', coaddroot='test_coadd')

        # resample flag image
        msgs.info('Running Swarp to resample flag images.')
        # copy the .head for flag images
        for ii, icat in enumerate(self.cat_proc_list):
            os.system('cp {:} {:}'.format(icat.replace('.fits', '.head'),
                                          self.flag_proc_list[ii].replace('.fits', '_cat.head')))
        swarp.run_swarp(self.flag_proc_list, config=self.swarpconfig_flag, workdir=self.science_path, defaultconfig='pyphot',
                        n_process=self.n_process, delete=self.delete, log=False, verbose=False)

        # resample inverse variance image
        msgs.info('Running Swarp to resample flag images.')
        # copy the .head for flag images
        for ii, icat in enumerate(self.cat_proc_list):
            os.system('cp {:} {:}'.format(icat.replace('.fits', '.head'),
                                          self.ivar_proc_list[ii].replace('.fits', '_cat.head')))
        swarp.run_swarp(self.ivar_proc_list, config=self.swarpconfig_ivar, workdir=self.science_path, defaultconfig='pyphot',
                        n_process=self.n_process, delete=self.delete, log=False, verbose=False)

        # remove useless data and change flag image type
        msgs.info('Cleaning up temporary files.')
        for ii, flag_fits in enumerate(self.flag_resample_list):
            par = fits.open(flag_fits, memmap=False)
            # Force the exptime=1 and FLXSCALE=1 (FLXSCALE was generated from the scamp run and will be used by Swarp later on)
            # in order to get the correct flag when doing the coadd with swarp in the later step
            par[0].header['EXPTIME'] = 1.0
            par[0].header['FLXSCALE'] = 1.0
            par[0].header['FLASCALE'] = 1.0
            par[0].data = par[0].data.astype('int32')
            par[0].writeto(flag_fits, overwrite=True)
            del par[0].data
            par.close()
            gc.collect()
            # remove weight map for flag and ivar
            os.system('rm {:}'.format(self.flag_resample_list[ii].replace('.fits', '.weight.fits')))
            os.system('rm {:}'.format(self.ivar_resample_list[ii].replace('.fits', '.weight.fits')))
        for ii, sci_fits in enumerate(self.sci_proc_list): # not that for mosaic, sci_proc_list has different size with sci_resample_list
            # remove _cat.head
            os.system('rm {:}'.format(self.sci_proc_list[ii].replace('.fits', '_cat.head')))
            os.system('rm {:}'.format(self.flag_proc_list[ii].replace('.fits', '_cat.head')))
            os.system('rm {:}'.format(self.ivar_proc_list[ii].replace('.fits', '_cat.head')))

        ## Step four: run SExtractor on resampled sciproc images
        msgs.info('Running SExtractor for the second pass to extract catalog from resampled images.')
        sex.run_sex(self.sci_resample_list, flag_image_list=self.flag_resample_list, weight_image_list=self.wht_resample_list,
                    n_process=self.n_process, task=self.sextask, config=self.sexconfig, workdir=self.science_path, params=self.sexparams,
                    defaultconfig='pyphot', conv=self.conv, nnw=None, dual=False, delete=self.delete, log=self.log, verbose=self.verbose)

    def run_chip_cal(self):
        '''
        Calibrate the zeropoint of individual chips
        Returns
        -------

        '''
        # Prepare reference catalogs
        groups = np.unique(self.coadd_ids.data)
        for igroup in groups:
            _ = self.get_phoref_cat(igroup)

        # Calibrate zero point chip by chip
        zp_all, zp_std_all, nstar_all, fwhm_all = cal_chips(self.cat_resample_list,
                                                            sci_fits_list=self.sci_resample_list,
                                                            ref_fits_list=self.master_phoref_cats,
                                                            outqa_root_list=self.outqa_list,
                                                            refcatalog=self.photref_catalog,
                                                            primary=self.primary,
                                                            secondary=self.secondary,
                                                            coefficients=self.coefficients,
                                                            ZP=self.zeropoint,
                                                            nstar_min=self.nstar_min,
                                                            external_flag=self.external_flag,
                                                            pixscale=self.pixscale,
                                                            n_process=self.n_process)

        # The FITS table that stores individual zero-points
        # ToDo: add filter to master_zpt_name
        master_zpt_name = os.path.join(self.par['calibrations']['master_dir'],'MasterChipZPT_{:}.fits'.format(self.setup_id))
        master_zpt_tbl = Table()
        master_zpt_tbl['Name'] = [os.path.basename(i) for i in self.sci_resample_list]
        master_zpt_tbl['DETID'] = self.resamp_det_ids
        master_zpt_tbl['exptime'] = np.repeat(self.sci_exptime, len(self.detectors))
        master_zpt_tbl['airmass'] = np.repeat(self.sci_airmass, len(self.detectors))
        master_zpt_tbl['ZPT'] = zp_all
        master_zpt_tbl['ZPT_Std'] = zp_std_all
        master_zpt_tbl['FWHM'] = fwhm_all
        master_zpt_tbl['NStar'] = nstar_all.astype('int32')
        master_zpt_tbl.write(master_zpt_name, overwrite=True)
        msgs.info('Calibrated zeropoint information for individual chips are saved to {:}'.format(master_zpt_name))

    def run_img_qa(self):
        '''
        Make QA plot for calibrated individual chips
        Returns
        -------

        '''
        outroots = []
        for this_image in self.sci_resample_list:
            outroots.append(os.path.join(self.qa_path, os.path.basename(this_image).replace('.fits', '_img')))
        utils.showimages(self.sci_resample_list, outroots=outroots,
                         interval_method=self.par['postproc']['qa']['interval_method'],
                         vmin=self.par['postproc']['qa']['vmin'],
                         vmax=self.par['postproc']['qa']['vmax'],
                         stretch_method=self.par['postproc']['qa']['stretch_method'],
                         cmap=self.par['postproc']['qa']['cmap'],
                         plot_wcs=self.par['postproc']['qa']['plot_wcs'],
                         show=self.par['postproc']['qa']['show'],
                         n_process=self.n_process)

    def run_coadd(self):
        '''
        Coadd images for each target based on coadd_ids
        Returns
        -------

        '''
        if self.par['postproc']['coadd']['pixscale'] is not None:
            self.pixscale = self.par['postproc']['coadd']['pixscale'] # Update pixscale

        # Coadd by group
        groups = np.unique(self.coadd_ids.data)
        for igroup in groups:
            this_group = self.resamp_coadd_ids == igroup
            this_sci_list = np.array(self.sci_resample_list)[this_group].tolist()
            this_flag_list = np.array(self.flag_resample_list)[this_group].tolist()
            this_ivar_list = np.array(self.ivar_resample_list)[this_group].tolist()
            #this_target = self.sci_target[self.coadd_ids==igroup][0]
            #this_filter = self.sci_filter[self.coadd_ids==igroup][0]
            #this_coadd_root = '{:}_{:}_coadd_ID{:03d}'.format(this_target, this_filter, igroup)
            this_coadd_root = self.coadd_root_names[np.where(np.array(self.coadd_root_ids)==igroup)[0][0]]
            this_photref_cat = np.array(self.master_phoref_cats)[this_group].tolist()[0]

            # run it
            coadd_file, coadd_wht_file, coadd_flag_file, coadd_ivar_file = coadd(this_sci_list, this_flag_list, this_ivar_list, this_coadd_root,
                                                        self.pixscale, self.science_path, self.coadd_path,
                                                        weight_type=self.par['postproc']['coadd']['weight_type'],
                                                        rescale_weights=self.par['postproc']['coadd']['rescale_weights'],
                                                        combine_type=self.par['postproc']['coadd']['combine_type'],
                                                        clip_ampfrac=self.par['postproc']['coadd']['clip_ampfrac'],
                                                        clip_sigma=self.par['postproc']['coadd']['clip_sigma'],
                                                        blank_badpixels=self.par['postproc']['coadd']['blank_badpixels'],
                                                        subtract_back=self.par['postproc']['coadd']['subtract_back'],
                                                        back_type=self.par['postproc']['coadd']['back_type'],
                                                        back_default=self.par['postproc']['coadd']['back_default'],
                                                        back_size=self.par['postproc']['coadd']['back_size'],
                                                        back_filtersize=self.par['postproc']['coadd']['back_filtersize'],
                                                        back_filtthresh=self.par['postproc']['coadd']['back_filtthresh'],
                                                        resampling_type=self.par['postproc']['coadd']['resampling_type'],
                                                        delete=self.par['postproc']['coadd']['delete'],
                                                        log=self.par['postproc']['coadd']['log'])

            # Extract a catalog for zero point calibration.
            msgs.info('Extracting a catalog using SExtractor for zero-point calibration.')
            sex.sexone(coadd_file, catname=coadd_file.replace('.fits', '_zptcat.fits'),
                       flag_image=coadd_flag_file, weight_image=coadd_wht_file,
                       task=self.sextask, config=self.sexconfig, workdir=self.coadd_path, params=self.sexparams,
                       defaultconfig='pyphot', dual=False, conv='sex', nnw=None, delete=True, log=False)

            # Calibrate zeropoint for the coadded image
            if self.par['postproc']['coadd']['cal_zpt']:
                msgs.info('Calcuating the zeropoint for {:}'.format(coadd_file))
                zpt, zpt_std, nstar, matched_table = calzpt(coadd_file.replace('.fits', '_zptcat.fits'),
                        refcatalog=self.photref_catalog, primary=self.primary, secondary=self.secondary,
                        coefficients=self.coefficients, FLXSCALE=1.0, FLASCALE=1.0,
                        out_refcat=this_photref_cat, external_flag=self.external_flag, nstar_min=self.nstar_min,
                        outqaroot=os.path.join(self.qa_path, this_coadd_root))

                if matched_table is not None:
                    star_table = Table()
                    star_table['x'] = matched_table['XWIN_IMAGE']
                    star_table['y'] = matched_table['YWIN_IMAGE']
                    fwhm, _, _, _ = psf.buildPSF(star_table, coadd_file, pixscale=self.pixscale,
                                                 outroot=os.path.join(self.qa_path, this_coadd_root))
                else:
                    fwhm = 0.
            else:
                zpt = self.par['postproc']['photometry']['zpt']
                zpt_std = 0.
                nstar = 0

            ## Write the ZPT into fits header
            par = fits.open(coadd_file, memmap=False)
            par[0].header['ZP'] = (zpt, 'Zero point')
            par[0].header['ZP_STD'] = (zpt_std, 'The standard deviration of ZP')
            par[0].header['ZP_NSTAR'] = (nstar, 'The number of stars used for ZP and FWHM')
            par[0].header['FWHM'] = (fwhm, 'FWHM in units of arcsec measured from stars')
            par.writeto(coadd_file, overwrite=True)

    def extract_catalog(self):
        '''
        Extract catalog from coadded images
        Returns
        -------

        '''
        if self.par['postproc']['coadd']['pixscale'] is not None:
            self.pixscale = self.par['postproc']['coadd']['pixscale'] # Update pixscale

        # Coadd by group
        groups = np.unique(self.coadd_ids.data)
        for igroup in groups:
            this_group = self.resamp_coadd_ids == igroup
            this_photref_cat = np.array(self.master_phoref_cats)[this_group].tolist()[0]
            this_coadd_root = self.coadd_root_names[np.where(np.array(self.coadd_root_ids)==igroup)[0][0]]

            coadd_file = this_coadd_root + '_sci.fits'
            coadd_wht_file = this_coadd_root + '_sci.weight.fits'
            coadd_flag_file = this_coadd_root + '_flag.fits'

            this_zpt = fits.getheader(os.path.join(self.coadd_path, coadd_file))['ZP']

            phot_table, rmsmap, bkgmap = detect(coadd_file, outroot=this_coadd_root,
                                            flag_image=coadd_flag_file, weight_image=coadd_wht_file,
                                            workdir=self.coadd_path, bkg_image=None, rms_image=None,
                                            zpt=this_zpt, effective_gain=None, pixscale=self.pixscale,
                                            detection_method=self.par['postproc']['detection']['detection_method'],
                                            detect_thresh=self.par['postproc']['detection']['detect_thresh'],
                                            analysis_thresh=self.par['postproc']['detection']['analysis_thresh'],
                                            detect_minarea=self.par['postproc']['detection']['detect_minarea'],
                                            fwhm=self.par['postproc']['detection']['fwhm'],
                                            nlevels=self.par['postproc']['detection']['nlevels'],
                                            contrast=self.par['postproc']['detection']['contrast'],
                                            back_type=self.par['postproc']['detection']['back_type'],
                                            back_rms_type=self.par['postproc']['detection']['back_rms_type'],
                                            back_size=self.par['postproc']['detection']['back_size'],
                                            back_filter_size=self.par['postproc']['detection']['back_filtersize'],
                                            back_default=self.par['postproc']['detection']['back_default'],
                                            backphoto_type=self.par['postproc']['detection']['backphoto_type'],
                                            backphoto_thick=self.par['postproc']['detection']['backphoto_thick'],
                                            weight_type=self.par['postproc']['detection']['weight_type'],
                                            check_type=self.par['postproc']['detection']['check_type'],
                                            back_nsigma=self.par['postproc']['detection']['back_nsigma'],
                                            back_maxiters=self.par['postproc']['detection']['back_maxiters'],
                                            morp_filter=self.par['postproc']['detection']['morp_filter'],
                                            defaultconfig='pyphot', dual=False,
                                            conv=self.par['postproc']['detection']['conv'],
                                            nnw=self.par['postproc']['detection']['nnw'],
                                            delete=self.par['postproc']['detection']['delete'],
                                            log=self.par['postproc']['detection']['log'],
                                            phot_apertures=self.par['postproc']['detection']['phot_apertures'],
                                            sextractor_task = self.sextask)

    def get_astref_cat(self, igroup):
        '''
        Download astrometric reference catalog and save to FITS-LDAC format
        Parameters
        ----------
        igroup

        Returns
        -------

        '''
        ra = np.median(self.sci_ra[self.coadd_ids == igroup])
        dec = np.median(self.sci_dec[self.coadd_ids == igroup])
        this_ref_cat = os.path.join(self.par['calibrations']['master_dir'],
                                    'MasterAstRefCat_{:}_{:}_ID{:03d}.fits'.format(self.astref_catalog, self.setup_id, igroup))

        if os.path.exists(this_ref_cat) and self.reuse_masters:
            msgs.info('Astrometric reference catalog {:} exists.'.format(this_ref_cat))
        else:
            ## ToDo: parameterize radius. Either use FGROUP_RADIUS parameter or add a new Par
            msgs.info('Querying astrometric reference catalog {:}.'.format(this_ref_cat))
            if self.astref_catalog == 'LS-DR9':
                tbl = query.query_standard(ra, dec, radius=0.5, catalog='Legacy', data_release='ls_dr9')
            elif self.astref_catalog == 'DES-DR2':
                tbl = query.query_standard(ra, dec, radius=0.5, catalog='DES')

            hdulist = ldac.convert_table_to_ldac(tbl)
            hdulist.writeto(this_ref_cat, overwrite=True)
        # mag_min=self.astrefmag_limits[0], mag_max=self.astrefmag_limits[1],
        #tbl = query.get_tbl_for_scamp(this_ref_cat, ra, dec, radius=0.5,
        #                              catalog=self.astref_catalog, reuse_master=self.reuse_masters)

        return this_ref_cat

    def get_phoref_cat(self, igroup):
        '''
        Download photometric reference catalog and save to FITS-LDAC format
        Parameters
        ----------
        igroup

        Returns
        -------

        '''
        ra = np.median(self.sci_ra[self.coadd_ids == igroup])
        dec = np.median(self.sci_dec[self.coadd_ids == igroup])
        this_ref_cat = os.path.join(self.par['calibrations']['master_dir'],
                                    'MasterPhoRefCat_{:}_{:}_ID{:03d}.fits'.format(self.photref_catalog, self.setup_id, igroup))

        if os.path.exists(this_ref_cat) and self.reuse_masters:
            msgs.info('Photometric reference catalog {:} exists.'.format(this_ref_cat))
        else:
            msgs.info('Querying photometric reference catalog {:}.'.format(this_ref_cat))
            ## ToDo: parameterize radius.
            tbl = query.query_standard(ra, dec, catalog=self.photref_catalog, radius=0.5)
            tbl.write(this_ref_cat, overwrite=True, format='fits')

        return this_ref_cat
