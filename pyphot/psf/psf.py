import numpy as np
import matplotlib.pyplot as plt

from astropy import stats
from astropy.table import Table
from astropy.nddata import NDData
from astropy.visualization import simple_norm

from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder

from pyphot import io, msgs
from pyphot import utils

def buildPSF(table, image, size=51, oversampling=4.0, sigclip=5, maxiters=10, shape=None, smoothing_kernel='quartic',
             recentering_maxiters=20, norm_radius=2.5, shift_val=0.5, recentering_boxsize=(5, 5), center_accuracy=0.001,
             pixscale=None, cenfunc='median',outroot=None):
    '''
    Build an effective PSF model from a given fits image and a fits table

    Args:
        startable: The fullpath of your FITS table. It can be either SExtractor catalog or
            photutils catalog, or any FITS table with columns of x and y which corresponding
            to the center positions of your stars. It could also be a astropy Table include
            x, y or sky coordinates (in a ``skycoord`` column containing a
            `~astropy.coordinates.SkyCoord` object).
        image: The fullpath of your FITS image or a 2D array.
        size: The extraction box size along each axis.  If ``size`` is a
            scalar then a square box of size ``size`` will be used.  If
            ``size`` has two elements, they should be in ``(ny, nx)`` order.
            The size must be greater than or equal to 3 pixel for both axes.
            Size must be odd in both axes; if either is even, it is padded
            by one to force oddness.
        outroot:

    Returns:
        ndarray: mask of cosmic rays (0=no CR, 1=CR)

    '''

    if isinstance(image, str):
        head, data, flag = io.load_fits(image)
        nddata = NDData(data=data)
    else:
        nddata = NDData(data=image)

    if isinstance(table, str):
        ## ToDo: distinguish SExtractor catalog from photutils catalog
        startable = Table.read(table)
    else:
        startable = table

    msgs.info('Extracting cutouts for {:} stars.'.format(len(startable)))
    stars = extract_stars(nddata, startable, size=size)

    # build PSF model with simple median procedure
    if np.size(size)==1:
        nx, ny = size, size
    else:
        nx,ny = size[0], size[1]
    xcen, ycen = nx//2-1, ny//2-1
    nz = len(stars)
    data3D = np.zeros((nx, ny, nz))
    for i in range(len(stars)):
        tmp = stars[i].data
        norm = np.sum(tmp[int(xcen-norm_radius):int(xcen+norm_radius), int(ycen-norm_radius):int(ycen+norm_radius)])
        data3D[:,:,i] = stars[i].data / norm

    ## constructing the master PSF frame
    mean, median, stddev = stats.sigma_clipped_stats(data3D, mask=np.isnan(data3D), sigma=sigclip, maxiters=maxiters,
                                                     cenfunc=cenfunc, stdfunc='std', axis=2)
    varirance = stddev**2

    if cenfunc=='median':
        psf2D = median
    elif cenfunc=='mean':
        psf2D = mean
    # correcting offset
    tmp = psf2D.copy()
    mask = np.zeros_like(tmp,dtype='bool')
    tmp[int(xcen - norm_radius):int(xcen + norm_radius), int(ycen - norm_radius):int(ycen + norm_radius)] = np.nan
    mask[int(xcen - norm_radius):int(xcen + norm_radius), int(ycen - norm_radius):int(ycen + norm_radius)] = 1
    offset = stats.sigma_clipped_stats(tmp, mask=mask,sigma=sigclip, maxiters=maxiters)
    psf2D -= offset[1]
    # normalize the 2D
    psf2D = psf2D*utils.inverse(np.nanmax(psf2D))

    # normalize the 1D
    x1D, psf1DX, sig1DX = np.arange(nx)-xcen, np.sum(psf2D,axis=0), np.sqrt(np.sum(varirance,axis=0))
    y1D, psf1DY, sig1DY = np.arange(ny)-ycen, np.sum(psf2D,axis=1), np.sqrt(np.sum(varirance,axis=1))
    psf1DX = psf1DX*utils.inverse(np.nanmax(psf1DX))
    psf1DY = psf1DY*utils.inverse(np.nanmax(psf1DY))

    ## ToDo: this is a hack need to debug the robust_curve_fit. It fails to some cases.
    try:
        popt, pcov = utils.robust_curve_fit(utils.gauss1D, np.hstack([x1D,y1D]), np.hstack([psf1DX,psf1DY]),
                                            sigma=np.hstack([sig1DX,sig1DY]), niters=3, sigclip=sigclip, maxiters_sigclip=maxiters)
        fwhm = popt[-2] * 2 * np.sqrt(2 * np.log(2))
        msgs.info('FWHM = {:0.2f} pixels'.format(fwhm))
    except:
        fwhm= 0.
        popt = None
        msgs.warn('PSF fitting failed')

    # Fit 2D gaussian to the PSF image to derive the FWHM of the PSF
    # ToDO: Fails to some cases. need a better fitting.
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    #xx, yy = np.meshgrid(x, y)
    #initial_guess = (1.,int(nx/2),int(ny/2),1.,0.,0.)
    #xdata = np.vstack((xx.ravel(), yy.ravel()))
    #ydata = psf2D.ravel()
    #popt, pcov = utils.robust_curve_fit(utils.gauss2D, xdata, ydata, p0=initial_guess)
    #psfmodel2D = utils.gauss2D((xx, yy),*popt).reshape(nx,ny)

    if outroot is not None:
        plt.figure()
        plt.plot(x1D, np.sum(data3D,axis=1)/np.max(np.sum(data3D,axis=1),axis=0),'-',lw=0.5,color='0.7')
        plt.errorbar(x1D, psf1DX, yerr=sig1DX, linestyle='None', marker='o', color='darkorange', ecolor='darkorange', mfc='None',zorder=100)
        plt.errorbar(y1D, psf1DY, yerr=sig1DY, linestyle='None', marker='o', color='dodgerblue', ecolor='dodgerblue', mfc='None',zorder=100)
        if popt is not None:
            plotxx = np.linspace(-np.max([nx,ny]),np.max([nx,ny]),1000)
            plotyy = utils.gauss1D(plotxx,*popt)
            plt.plot(plotxx, plotyy,'r-')
        plt.xlim(-np.max([nx,ny])/2,np.max([nx,ny])/2)
        plt.ylim(-0.05,1.1)

        plt.text(-np.max([nx,ny])/2.1, 0.9, 'Nstar={:}'.format(len(stars)),fontsize=12)
        plt.text(-np.max([nx,ny])/2.1, 0.8, 'FWHM={:0.2f} pixels'.format(fwhm),fontsize=12)
        if pixscale is not None:
            plt.text(-np.max([nx, ny]) / 2.1, 0.7, 'FWHM={:0.2f} arcsec'.format(fwhm*pixscale), fontsize=12)

        #plt.plot([-0.5*fwhm,0.5*fwhm],[0.5,0.5],'k:')
        if popt is not None:
            plt.plot([-0.5*fwhm,0.5*fwhm],[popt[-1]+popt[0]/2,popt[-1]+popt[0]/2],'k:')
        plt.xlabel('X [Pixels]',fontsize=14)
        plt.ylabel('Normalized PSF',fontsize=14)
        plt.savefig('{:}_PSF_1D.pdf'.format(outroot))
        plt.close()

        fig = plt.figure(figsize=(6,6))
        fig.subplots_adjust(left=0.,right=1.,bottom=0.,top=1.,wspace =0,hspace =0)
        f = plt.imshow(psf2D,interpolation='nearest', aspect='auto',cmap='RdYlBu_r', origin='lower')
        sortedvals = np.sort(psf2D.ravel())
        f.set_clim([sortedvals[int(nx*ny*.1)], sortedvals[int(nx*ny*0.95)]])
        plt.contour(x, y, psf2D.reshape(nx, ny), 10, colors='lightgreen')
        #plt.contour(x, y, psfmodel2D.reshape(nx, ny), 10, colors='lightgreen')
        plt.text(5, 8, 'Nstar={:}'.format(len(stars)),fontsize=12)
        plt.text(5,5, 'FWHM={:0.2f} pixels'.format(fwhm),fontsize=18, color='white')
        if pixscale is not None:
            plt.text(5, 2, 'FWHM={:0.2f} arcsec'.format(fwhm*pixscale), fontsize=18, color='white')
        plt.xlim([0,nx-1])
        plt.ylim([0,ny-1])
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        plt.savefig('{:}_PSF_2D.pdf'.format(outroot))
        plt.close()

    if pixscale is not None:
        fwhm *= pixscale
        msgs.info('FWHM = {:0.2f} arcsec'.format(fwhm))

    '''
    # build PSF model with EPSBuilder
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, shape=shape, smoothing_kernel=smoothing_kernel,
                               recentering_maxiters=recentering_maxiters, norm_radius=norm_radius, shift_val=shift_val,
                               recentering_boxsize=recentering_boxsize, center_accuracy=center_accuracy, progress_bar=False)

    msgs.info('Building the effective PSF model.')
    epsf, fitted_stars = epsf_builder(stars)

    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.show()
    '''
    return fwhm, psf2D

'''
#image = '/Volumes/Work/Imaging/J0100_LBC_WIRCAM/J0100_D3_Y_coadd_ID001_sci.fits'
#fitstable = '/Volumes/Work/Imaging/J0100_LBC_WIRCAM/J0100_D3_Y_coadd_ID001_sci_cat.fits'
#image = '/Volumes/Work/Imaging/J0100_LBC_WIRCAM/z6_QSO_i-SLOAN_coadd_ID002_sci.fits'
#fitstable = '/Volumes/Work/Imaging/J0100_LBC_WIRCAM/z6_QSO_i-SLOAN_coadd_ID002_sci_cat.fits'
image = '/Volumes/Work/Imaging/LBC/redux/2015/lbt_lbcr_B/Science/lbcr.20151122.034736_det01_sci.resamp.fits'
fitstable = '/Volumes/Work/Imaging/LBC/redux/2015/lbt_lbcr_B/Science/lbcr.20151122.034736_det01_sci.resamp_cat.fits'
tbl = Table.read(fitstable,2)
point = (tbl['CLASS_STAR']>0.3) & (tbl['FLAGS']<1) & (tbl['IMAFLAGS_ISO']<1) & (tbl['FLUX_AUTO']/tbl['FLUXERR_AUTO']>5) &\
        (tbl['FLUX_AUTO']/tbl['FLUXERR_AUTO']<100)
startable = Table()
startable['x'] = tbl['XWIN_IMAGE'][point]
startable['y'] = tbl['YWIN_IMAGE'][point]
buildPSF(startable, image, size=31, outroot='test')

image='/Volumes/Work/Imaging/LBC/redux/2015/pylbc_redux/J0100+2802.z.fits'
fitstable = '/Volumes/Work/Imaging/LBC/redux/2015/pylbc_redux/J0100+2802.merge.cat.corr.fits'
tbl = Table.read(fitstable,2)
point = (tbl['CLASS_STAR_z']>0.98) & (tbl['FLAGS_z']<1) & (1.0857/tbl['MAGERR_AUTO_z']>50) &\
        (1.0857/tbl['MAGERR_AUTO_z']<100)
startable = Table()
startable['x'] = tbl['X_IMAGE'][point]
startable['y'] = tbl['Y_IMAGE'][point]

buildPSF(startable, image, size=31, outroot='test2')
'''
