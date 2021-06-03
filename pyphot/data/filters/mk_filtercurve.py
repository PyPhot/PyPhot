import sys

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table

import matplotlib.pyplot as plt
import numpy as np

from IPython import embed

import requests

def tohdu(wave,trans,extname):
    col1 = fits.Column(name='lam',format='1D',array=wave)
    col2 = fits.Column(name='Rlam',format='1D',array=trans)
    cols = fits.ColDefs([col1,col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header.update({"EXTNAME":extname})
    return hdu

def orig_build():
    '''
    Code for generating a fits file of filter transmission curves.
    Wavelength units are Angstrom
    Data are collected by Feige Wang in 2019 Feb.
    The orginal transmission curves from each instruments/telescopes are in the dev suite.
    '''
    pri_hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(pri_hdu)

    ## SDSS filter curves
    # Download from https://www.sdss3.org/instruments/camera.php#Filters
    # filter_curves.fits
    par = fits.open('filter_curves_sdss.fits')
    for i in ['SDSS-U','SDSS-G','SDSS-I','SDSS-G','SDSS-Z']:
        wave, trans = par[i[-1]].data['wavelength'], par[i[-1]].data['respt']
        hdu = tohdu(wave,trans,i)
        hdulist.append(hdu)

    ## PS1 filter curves
    ps1_tbl = ascii.read('panstarrs_Tonry2012.txt')
    wave = ps1_tbl['col1'] * 10.0
    for i,ifilter in enumerate(['PS1-G','PS1-R','PS1-I','PS1-Z','PS1-Y']):
        trans = ps1_tbl['col{:d}'.format(i+3)]
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## DECAM filter curves
    des_u = ascii.read('DECAM_ufilter.dat')
    wave_u,trans_u = des_u['LAMBDA'],des_u['u']
    hdu = tohdu(wave_u, trans_u, 'DECAM-U')
    hdulist.append(hdu)

    des_tbl = ascii.read('STD_BANDPASSES_DR1.dat')
    wave = des_tbl['LAMBDA']
    for i,ifilter in enumerate(['DECAM-G','DECAM-R','DECAM-I','DECAM-Z','DECAM-Y']):
        trans = des_tbl[ifilter[-1]]
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ##BASS+MZLS filter curves
    # dowload from http://legacysurvey.org/dr6/description/
    for i,ifilter in enumerate(['BASS-G','BASS-R']):
        tbl = ascii.read('BASS_{:s}_corr.bp'.format(ifilter[-1:]))
        wave = tbl['col1']
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)
    tbl = ascii.read('kpzdccdcorr3.txt')
    wave = tbl['col1']
    trans = tbl['col2']
    hdu = tohdu(wave, trans, 'MZLS-Z')
    hdulist.append(hdu)

    ## HSC SSP survey filter curves
    # ToDo: I don't think they include atmosphere transmission though.
    # download from https://www.naoj.org/Projects/HSC/forobservers.html
    for i,ifilter in enumerate(['HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y']):
        hsc_tbl = ascii.read(ifilter+'.dat')
        wave = hsc_tbl['col1']
        trans = hsc_tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## LSST filter curves
    # download from https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/master/components/camera/filters/y_band_Response.dat
    for i,ifilter in enumerate(['LSST-U','LSST-G','LSST-R','LSST-I','LSST-Z','LSST-Y']):
        tbl = ascii.read('LSST_{:s}_band_Response.dat'.format(ifilter[-1:]))
        wave = tbl['col1']*10.
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## CFHT filter curves
    # download from http://www.cfht.hawaii.edu/Instruments/Filters/megaprimenew.html
    # u = cfh9302, g = cfh9402, r = cfh9602, i = cfh9703, z = cfh9901
    for i,ifilter in enumerate(['CFHT-U','CFHT-G','CFHT-R','CFHT-I','CFHT-Z']):
        tbl = ascii.read('cfh_{:s}.dat'.format(ifilter[-1:]))
        wave = tbl['col1']*10.
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## UKIRT filter curves
    for i,ifilter in enumerate(['UKIRT-Z','UKIRT-Y','UKIRT-J','UKIRT-H','UKIRT-K']):
        tbl = ascii.read('ukidss{:s}.txt'.format(ifilter[-1:]))
        wave = tbl['col1']*1e4
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## VISTA filter curves
    #download from http://www.eso.org/sci/facilities/paranal/instruments/vircam/inst/Filters_QE_Atm_curves.tar.gz
    for i,ifilter in enumerate(['VISTA-Z','VISTA-Y','VISTA-J','VISTA-H','VISTA-K']):
        tbl = ascii.read('VISTA_Filters_at80K_forETC_{:s}.txt'.format(ifilter[-1:]))
        wave = tbl['col1']*10.
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ##TWO MASS filter curves
    #download from https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html#rsr
    for i,ifilter in enumerate(['TMASS-J','TMASS-H','TMASS-K']):
        tbl = ascii.read('2mass{:s}.txt'.format(ifilter[-1:]))
        wave = tbl['col1']*1e4
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## WISE filter curves
    for i,ifilter in enumerate(['WISE-W1','WISE-W2','WISE-W3','WISE-W4']):
        tbl = ascii.read('RSR-{:s}.txt'.format(ifilter[-2:]))
        wave = tbl['col1']*1e4
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## GAIA DR2 revised filter curves
    #download from https://www.cosmos.esa.int/web/gaia/iow_20180316/
    gaia_tbl = ascii.read('GaiaDR2_RevisedPassbands.dat')
    wave = gaia_tbl['wave'] * 10.0
    for i,ifilter in enumerate(['GAIA-G','GAIA-B','GAIA-R']):
        trans = gaia_tbl[ifilter[-1]]
        gdp = trans<90. # bad values in the file was set to 99.0
        hdu = tohdu(wave[gdp],trans[gdp],ifilter)
        hdulist.append(hdu)

    ## GALEX filter curves
    # download from http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=GALEX
    for i,ifilter in enumerate(['GALEX-N','GALEX-F']):
        tbl = ascii.read('GALEX_GALEX.{:s}UV.dat'.format(ifilter[-1]))
        wave = tbl['col1']
        trans = tbl['col2']
        hdu = tohdu(wave,trans,ifilter)
        hdulist.append(hdu)

    ## Write out the final filter curve table
    hdulist.writeto('filtercurves_20190215.fits',overwrite=True)


def append_FORS():
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    filters = ['BESS_B','BESS_I','BESS_R','BESS_U','BESS_V','u_HIGH','b_HIGH',
               'v_HIGH','g_HIGH','GUNN_G',
               'GUNN_R','GUNN_U','GUNN_V','GUNN_Z',
               'SPECIAL_R','SPECIAL_U']
    for filt in filters:
        vlt_name = 'VLT-{}'.format(filt)
        if vlt_name in curr_filters:
            print("Filter {} already there, skipping".format(vlt_name))
            continue
        if 'HIGH' in filt:
            url = 'http://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/{}.txt'.format(filt)
        else:
            url = 'http://www.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/M_{}.txt'.format(filt)
        r = requests.get(url)
        # Parse me
        ss = r.text.split('\n')
        lam, T = [], []
        for row in ss:
            if 'fors' in row.lower():
                continue
            elif len(row) == 0:
                continue
            elif 'Wavelen' in row:
                continue
            items = row.split(' ')
            if len(items[0]) == 0: # Special
                items = items[1:]
                items[1] = items[-1]
            try:
                lam.append(float(items[0])*10)
            except ValueError:
                items = row.split('\t') # HIGH
                try:
                    lam.append(float(items[0])*10)
                except:
                    embed(header='196')
                T.append(float(items[1])/10)
            else:
                try:
                    T.append(float(items[1]))
                except:
                    embed(header='205')
        # Recast
        wave = np.array(lam)
        trans = np.array(T)
        # Add it
        hdu = tohdu(wave,trans,vlt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)


def append_NIRCam():
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    nircam = Table.read('nircam_filter/nircam_modABmean_plus_ote_filter_properties.txt',format='ascii.basic')
    filters = nircam['Filter']
    for filt in filters:
        flt_name = 'NIRCAM-{}'.format(filt)
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # load in the transmission
        filt_table = Table.read('nircam_filter/{:}_NRC_and_OTE_ModAB_mean.txt'.format(filt),format='ascii.basic')
        wave = filt_table['microns']*1e4
        trans = filt_table['throughput']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_MIRI():
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    miri = Table.read('miri_filter/miri_imaging.txt',format='ascii.basic')
    filters = ['F560W','F770W','F1000W', 'F1130W', 'F1280W','F1500W','F1800W','F2100W','F2550W']
    for filt in filters:
        flt_name = 'MIRI-{}'.format(filt)
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # load in the transmission
        wave = miri['Wave']*1e4
        trans = miri[filt]
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def fix_SDSS():

    par = fits.open('filter_curves_sdss.fits')
    pri_hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(pri_hdu)

    for i in ['SDSS-U','SDSS-G','SDSS-R','SDSS-I','SDSS-Z']:
        wave, trans = par[i[-1]].data['wavelength'], par[i[-1]].data['respt']
        hdu = tohdu(wave,trans,i)
        hdulist.append(hdu)

    # Load
    hdulist_orig = fits.open('filtercurves.fits')
    for i in range(len(hdulist_orig[6:])):
        hdulist.append(hdulist_orig[6+i])

    curr_filters = [hdu.name for hdu in hdulist]
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_EUCLID():
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    filters = ['Y', 'J', 'H', 'VIS']

    for filt in filters:
        flt_name = 'EUCLID-{}'.format(filt)
        euclid = fits.open('euclid_filter/{}.fits'.format(flt_name))[1].data
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
        wave = euclid['lam']
        trans = euclid['Rlam']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_WFIRST():
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    filters = ['F', 'H', 'J', 'R', 'W', 'Y', 'Z', 'Grism', 'Prism']

    for filt in filters:
        flt_name = 'WFIRST-{}'.format(filt)
        wfirst = fits.open('wfirst_filter/{}.fits'.format(flt_name))[1].data
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
        wave = wfirst['lam']
        trans = wfirst['Rlam']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def update_LSST():

    hdulist = fits.open('filtercurves.fits')
    for i, ifilter in enumerate(['LSST-U', 'LSST-G', 'LSST-R', 'LSST-I', 'LSST-Z', 'LSST-Y']):
        hdulist.pop(ifilter)
        tbl = ascii.read('lsst_filter/total_{:s}.dat'.format(ifilter[-1:]))
        hdu = tohdu(tbl['col1'] * 10., tbl['col2'], ifilter)
        hdulist.append(hdu)

    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_HST_WFC():
    ## ACS/WFC
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    filters = ['F435W', 'F475W', 'F502N', 'F555W', 'F606W', 'F625W', 'F775W', 'F850LP', 'F892N']

    for filt in filters:
        flt_name = 'WFC-{:}'.format(filt)
        wfc = fits.open('hst_filters/acs_{:}_wfc_002.fits'.format(filt.lower()))[1].data
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
        wave = wfc['WAVELENGTH']
        trans = wfc['THROUGHPUT']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_HST_WFC3():
    ## ACS/WFC
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    data = Table.read('hst_filters/wfc3.info',format='ascii.basic')
    files = data['FltName']
    for file in files:
        flt_name = 'WFC3-{:}'.format(file.split('_')[0])
        wfc = Table.read('hst_filters/{:}'.format(file))#fits.open('hst_filters/{:}'.format(file))[1].data
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
        print(file, wfc.keys())
        try:
            wave = wfc['Wave (Angstroms)']
            trans = wfc['Throughput']
        except:
            wave = wfc['Chip 1 Wave (Angstroms)']
            trans = wfc['Chip 1 Throughput']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_IMACS_NB919_F2():
    ## IMACS NB919
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    flt_name = 'IMACSF2-NB919'

    data1 = Table.read('IMACS_NB919_Throughput_Asahi.fits', 1)

    # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
    wave = data1['lam'] * 10.0
    trans = data1['Rlam_f2'] / 100.0
    # Add it
    hdu = tohdu(wave,trans,flt_name)
    hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def append_LBT_LBC():
    ## LBT/LBC
    # Load
    hdulist = fits.open('filtercurves.fits')
    curr_filters = [hdu.name for hdu in hdulist]
    #
    filters = ['BESS_U', 'BESS_B', 'BESS_V', 'BESS_R', 'BESS_I', 'Y']

    for filt in filters:
        flt_name = 'LBC-{:}'.format(filt)
        wfc = Table.read('/Users/feige/Downloads/{:}.txt'.format(filt), format='ascii.basic')
        if flt_name in curr_filters:
            print("Filter {} already there, skipping".format(flt_name))
            continue
        # First column wavelength 'lam' in angstrom, second column transmission 'Rlam'.
        if filt in ['BESS_R', 'BESS_I', 'Y']:
            wave = wfc['wave']*10.0
            trans = wfc['trans']
            indwave = np.argsort(wave)
            wave, trans = wave[indwave], trans[indwave]
        else:
            wave = wfc['wave']
            trans = wfc['trans']
        # Add it
        hdu = tohdu(wave,trans,flt_name)
        hdulist.append(hdu)
    # Write
    hdulist.writeto('filtercurves.fits', overwrite=True)

def write_filter_list():
    # Write the filter list
    hdulist = fits.open('filtercurves.fits')
    all_filters = [hdu.name for hdu in hdulist]
    tbl = Table()
    tbl['filter'] = all_filters
    # Write
    tbl.write('filter_list.ascii', format='ascii', overwrite=True)


#### ########################## #########################
def main(flg):

    flg = int(flg)

    # BPT
    if flg & (2**0):
        orig_build()

    # FORS filters
    if flg & (2**1):
        append_FORS()

    # Filter list
    if flg & (2**2):
        write_filter_list()

    # Add NIRCam filters
    if flg & (2**3):
        append_NIRCam()

    # Add MIRI filters
    if flg & (2**4):
        append_MIRI()

    # Fix SDSS filters
    if flg & (2**5):
        fix_SDSS()

    # Add EUCLID and WFIRST filters
    if flg & (2**6):
        append_EUCLID()
        append_WFIRST()

    # Update LSST
    if flg & (2**7):
        update_LSST()

    # ADD ACS/WFC
    if flg & (2**8):
        append_HST_WFC()

    # ADD WFC3
    if flg & (2**9):
        append_HST_WFC3()

    # ADD NB919
    if flg & (2**10):
        append_IMACS_NB919_F2()

    # ADD LBC
    if flg & (2**11):
        append_LBT_LBC()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2**0   # First build
        #flg += 2**1   # FORS filters
        flg += 2**2   # Filter list
    else:
        flg = sys.argv[1]

    main(flg)
'''
For example
python mk_filtercurve.py 8 
   -- this will append NIRCAM filters
'''