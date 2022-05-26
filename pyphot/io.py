import sys, os, gc
import time
import scipy
import astropy

import numpy as np
from astropy.io import fits
from astropy.table import Table

from pkg_resources import resource_filename

import pyphot
from pyphot import msgs

def initialize_header(hdr=None, primary=False):
    """
    Initialize a FITS header.

    Args:
        hdr (`astropy.io.fits.Header`, optional):
            Header object to update with basic summary
            information. The object is modified in-place and also
            returned. If None, an empty header is instantiated,
            edited, and returned.
        primary (bool, optional):
            If true and hdr is None, generate a Primary header

    Returns:
        `astropy.io.fits.Header`: The initialized (or edited)
        fits header.
    """
    # Add versioning; this hits the highlights but should it add
    # the versions of all packages included in the requirements.txt
    # file?
    if hdr is None:
        if primary:
            hdr = fits.PrimaryHDU().header
        else:
            hdr = fits.Header()
    hdr['VERSPYT'] = ('.'.join([ str(v) for v in sys.version_info[:3]]), 'Python version')
    hdr['VERSNPY'] = (np.__version__, 'Numpy version')
    hdr['VERSSCI'] = (scipy.__version__, 'Scipy version')
    hdr['VERSAST'] = (astropy.__version__, 'Astropy version')
    hdr['VERSPYP'] = (pyphot.__version__, 'PyPhot version')

    # Save the date of the reduction
    hdr['DATE'] = (time.strftime('%Y-%m-%d',time.gmtime()), 'UTC date created')

    # TODO: Anything else?

    # Return
    return hdr

def save_fits(fitsname, data, header, img_type, mask=None, overwrite=True):

    if header.get('VERSPYP') is None:
        # Add some Header card
        hdr = initialize_header(hdr=None, primary=True)
        hdr['IMGTYP'] = (img_type, 'PyPhot image type')
        for i in range(len(hdr)):
            header.append(hdr.cards[i])

    if mask is None:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(fitsname, overwrite=overwrite)
    else:
        hdu = fits.PrimaryHDU(header=header)
        hdu1 = fits.ImageHDU(data, header=header, name='IMAGE')
        hdu2 = fits.ImageHDU(mask.astype('int32'), header=header, name='MASK')
        new_hdul = fits.HDUList([hdu, hdu1, hdu2])
        new_hdul.writeto(fitsname, overwrite=True)
        #mask_hdu = fits.ImageHDU(mask.astype('int32'), name='MASK')
        #hdulist = fits.HDUList([hdu,mask_hdu])
        #hdulist.writeto(fitsname,overwrite=overwrite)
        del new_hdul[1].data
        del new_hdul[2].data
        new_hdul.close()
        gc.collect()


def load_fits(fitsname):
    par = fits.open(fitsname, memmap=False)
    if 'PROD_VER' in par[0].header.keys():
        msgs.info('Loading HST drizzled images')
        head, data, flag = par[0].header, par[0].data, np.zeros_like(par[0].data,dtype='int32')
        del par[0].data
    else:
        if len(par)==1:
            head, data, flag = par[0].header, par[0].data, np.zeros_like(par[0].data,dtype='int32')
            del par[0].data
        elif len(par)==3:
            head, data, flag = par[1].header, par[1].data, par[2].data
            del par[1].data
            del par[2].data
        else:
            msgs.error('{:} is not a PyPhot FITS Image.'.format(fitsname))
            return None
    par.close()
    gc.collect()

    return head, data, flag


def build_mef(rootname, detectors, img_type='SCI', returnname_only=False):

    primary_hdu = fits.PrimaryHDU(header=initialize_header(hdr=None, primary=True))
    primary_hdu.header['IMGTYP'] = (img_type, 'PyPhot image type')
    hdul_sci = fits.HDUList([primary_hdu])

    if img_type == 'SCI':
        app = 'sci.fits'
    elif img_type == 'IVAR':
        app = 'sci.ivar.fits'
    elif img_type == 'WEIGHT':
        app = 'sci.weight.fits'
    elif img_type == 'FLAG':
        app = 'flag.fits'
    else:
        msgs.error('Image Type {:} is not supported.'.format(img_type))

    out_sci_name = rootname.replace('.fits', '_mef_{:}'.format(app))

    if not returnname_only:
        for idet in detectors:
            this_sci_file = rootname.replace('.fits', '_det{:02d}_{:}'.format(idet, app))
            this_sci_hdr, this_sci_data, _ = load_fits(this_sci_file)
            this_hdu_sci = fits.ImageHDU(this_sci_data, header=this_sci_hdr, name='{:}-DET{:02d}'.format(img_type, idet))
            hdul_sci.append(this_hdu_sci)

        hdul_sci.writeto(out_sci_name, overwrite=True)
        msgs.info('MEF file saved to {:}'.format(out_sci_name))

    return out_sci_name

def build_mef_old(rootname, detectors, type='SCI'):

    primary_hdu = fits.PrimaryHDU(header=initialize_header(hdr=None, primary=True))
    hdul_sci = fits.HDUList([primary_hdu])
    hdul_ivar = fits.HDUList([primary_hdu])
    hdul_wht = fits.HDUList([primary_hdu])
    hdul_flag = fits.HDUList([primary_hdu])

    for idet in detectors:
        this_sci_file = rootname.replace('.fits', '_det{:02d}_sci.fits'.format(idet))
        this_ivar_file = rootname.replace('.fits', '_det{:02d}_sci.ivar.fits'.format(idet))
        this_wht_file = rootname.replace('.fits', '_det{:02d}_sci.weight.fits'.format(idet))
        this_flag_file = rootname.replace('.fits', '_det{:02d}_flag.fits'.format(idet))

        this_sci_hdr, this_sci_data, _ = io.load_fits(this_sci_file)
        this_hdu_sci = fits.ImageHDU(this_sci_data, header=this_sci_hdr, name='SCI-DET{:02d}'.format(idet))
        hdul_sci.append(this_hdu_sci)

        this_ivar_hdr, this_ivar_data, _ = io.load_fits(this_ivar_file)
        this_hdu_ivar = fits.ImageHDU(this_ivar_data, header=this_ivar_hdr, name='IVAR-DET{:02d}'.format(idet))
        hdul_ivar.append(this_hdu_ivar)

        this_wht_hdr, this_wht_data, _ = io.load_fits(this_wht_file)
        this_hdu_wht = fits.ImageHDU(this_wht_data, header=this_wht_hdr, name='WHT-DET{:02d}'.format(idet))
        hdul_wht.append(this_hdu_wht)

        this_flag_hdr, this_flag_data, _ = io.load_fits(this_flag_file)
        this_hdu_flag = fits.ImageHDU(this_flag_data, header=this_flag_hdr, name='FLAG-DET{:02d}'.format(idet))
        hdul_flag.append(this_hdu_flag)

    out_sci_name = rootname.replace('.fits', '_mef_sci.fits')
    hdul_sci[0].header['IMGTYP'] = 'SCI'
    hdul_sci.writeto(out_sci_name, overwrite=True)
    out_ivar_name = rootname.replace('.fits', '_mef_sci.ivar.fits')
    hdul_ivar[0].header['IMGTYP'] = 'IVAR'
    hdul_ivar.writeto(out_ivar_name, overwrite=True)
    out_wht_name = rootname.replace('.fits', '_mef_sci.weight.fits')
    hdul_sci[0].header['IMGTYP'] = 'WEIGHT'
    hdul_wht.writeto(out_wht_name, overwrite=True)
    out_flag_name = rootname.replace('.fits', '_mef_flag.fits')
    hdul_sci[0].header['IMGTYP'] = 'MASK'
    hdul_flag.writeto(out_flag_name, overwrite=True)

def load_filter(filter):
    """
    Load a system response curve for a given filter

    Args:
        filter (str): Name of filter

    Returns:
        ndarray, ndarray: wavelength, instrument throughput

    """
    filter_file = resource_filename('pyphot', os.path.join('data', 'filters', 'filter_list.ascii'))
    tbl = Table.read(filter_file, format='ascii')
    allowed_options = tbl['filter'].data

    # Check
    if filter not in allowed_options:
        msgs.error("PyPhot is not ready for filter = {}".format(filter))

    trans_file = resource_filename('pyphot', os.path.join('data', 'filters', 'filtercurves.fits'))
    trans = fits.open(trans_file, memmap=False)
    wave = trans[filter].data['lam']  # Angstroms
    instr = trans[filter].data['Rlam']  # Am keeping in atmospheric terms
    keep = instr > 0.
    # Parse
    wave = wave[keep]
    instr = instr[keep]

    # Return
    return wave, instr
