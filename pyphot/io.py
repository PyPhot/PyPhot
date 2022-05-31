import sys, os, gc
import time
import scipy
import astropy

import numpy as np
from astropy.io import fits
from astropy.table import Table

import multiprocessing
from multiprocessing import Process, Queue

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


def build_mef(rootname, detectors, img_type='SCI', returnname_only=False, overwrite=True):

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
        if not overwrite and os.path.exists(out_sci_name):
            msgs.info('Use existing MEF file {:}'.format(out_sci_name))
        else:
            for idet in detectors:
                this_sci_file = rootname.replace('.fits', '_det{:02d}_{:}'.format(idet, app))
                this_sci_hdr, this_sci_data, _ = load_fits(this_sci_file)
                this_hdu_sci = fits.ImageHDU(this_sci_data, header=this_sci_hdr, name='{:}-DET{:02d}'.format(img_type, idet))
                hdul_sci.append(this_hdu_sci)
            hdul_sci.writeto(out_sci_name, overwrite=True)
            msgs.info('MEF file saved to {:}'.format(out_sci_name))
            hdul_sci.close()
            gc.collect()

    return out_sci_name

def _build_mef_worker(work_queue, done_queue, detectors=None, img_type='SCI', returnname_only=False, overwrite=True):

    """Multiprocessing worker for _build_mef"""
    while not work_queue.empty():
        idx, rootname = work_queue.get()
        out_sci_name = build_mef(rootname, detectors,
                        img_type=img_type, returnname_only=returnname_only, overwrite=overwrite)

        done_queue.put((idx, out_sci_name))

def build_mef_parallel(rootnames, detectors=None, img_type='SCI', returnname_only=False, overwrite=True, n_process=1):
    '''
    Running build_mef in parallel
    Parameters
    ----------
    rootnames
    detectors
    img_type
    returnname_only
    overwrite
    n_process

    Returns
    -------

    '''

    n_file = len(rootnames)
    n_cpu = multiprocessing.cpu_count()

    if n_process > n_cpu:
        n_process = n_cpu

    if n_process>n_file:
        n_process = n_file

    out_sci_names = []
    idx_all = np.zeros(n_file)

    msgs.info('Building MEF {:} files with n_process={:}'.format(img_type, n_process))
    work_queue = Queue()
    done_queue = Queue()
    processes = []

    for ii in range(n_file):
        work_queue.put((ii, rootnames[ii]))

    # creating processes
    for w in range(n_process):
        p = Process(target=_build_mef_worker, args=(work_queue, done_queue), kwargs={
            'detectors': detectors, 'img_type': img_type,
            'returnname_only': returnname_only, 'overwrite': overwrite})
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    ii=0
    while not done_queue.empty():
        idx_all[ii], out_sci_name = done_queue.get()
        out_sci_names.append(out_sci_name)
        ii+=1

    out_sci_names_sort = np.array(out_sci_names)[np.argsort(idx_all)].tolist()

    return out_sci_names_sort

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
