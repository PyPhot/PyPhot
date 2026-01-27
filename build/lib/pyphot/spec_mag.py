import scipy
from scipy import interpolate
import numpy as np
from astropy import constants
from astropy import units

from pyphot.io import load_filter
from pyphot import msgs

vega_2_AB = {'UKIRT-Z':0.528,
             'UKIRT-Y':0.634,
             'UKIRT-J':0.938,
             'UKIRT-H':1.379,
             'UKIRT-K':1.900,
             'VISTA-Z':0.521,
             'VISTA-Y':0.618,
             'VISTA-J':0.937,
             'VISTA-H':1.384,
             'VISTA-K':1.839,
             'TMASS-J':0.890,
             'TMASS-H':1.370,
             'TMASS-K':1.840,
             'WISE-W1':2.699,
             'WISE-W2':3.339,
             'WISE-W3':5.174,
             'WISE-W4':6.620}

def mag2spec(filter, mag, mag_err=0.0, mag_type='AB', return_fnu=False):

    if mag_type=='AB':
        magAB = mag
    elif mag_type=='VEGA':
        if filter in vega_2_AB.keys():
            magAB = mag + vega_2_AB[filter]
        else:
            msgs.warn('VEGA to AB conversion factor was not found. Assuming AB!')
    else:
        msgs.error('The supported system are AB and VEGA.')

    # Load filter and estimate the effective wavelength
    fwave, trans = load_filter(filter)
    wav_eff = np.sum(fwave * trans) / np.sum(trans) * units.AA # in units of A
    fre_eff = (constants.c / wav_eff).to('Hz')

    # AB Magnitude zero point
    MAB0 = -2.5 * np.log10(3631.e-23)  # about 48.6
    ferr0 = np.log(10)

    fnu = 10. ** ((magAB + MAB0) / (-2.5))  # In units of erg s^-1 cm^-2 Hz^-1
    fnu_err = 0.4 * ferr0 * 10. ** ((magAB + MAB0) / (-2.5)) * mag_err

    if return_fnu:
        return fre_eff.value, fnu.value, fnu_err.value

    flam = fnu * constants.c.to('AA/s').value / (wav_eff.value ** 2)
    flam_err = fnu_err * constants.c.to('AA/s').value / (wav_eff.value ** 2)
    #flam = fnu * 1e23 * constants.c.to('km/s').value / (wav_eff.value ** 2)  # In units of erg s^-1 cm^-2 A^-1
    #flam_err = fnu_err * 1e23 * constants.c.to('km/s').value / (wav_eff.value ** 2)

    return wav_eff.value, flam, flam_err

def spec2flux(filter, wave, flam, sig, gpm=None, masks=None):
    """
    Calculate magnitude in a given filter with the given spectra

    Args:
        filter (str): name of filter
        wave (array): wavelength in units of A
        flam (array): flux in units of erg/s/cm2/A
        sig (array): flux error in units of erg/s/cm2/A
        gpm (bool array): good pixel masks for your spectrum
        masks (list, optional): Wavelength ranges to mask in calculation

    Returns:
        flux in the given filter in units of erg/s/cm2/Hz

    """

    if gpm is None:
        gpm = sig > 0.
    wave = wave[gpm]
    flam = flam[gpm]
    sig= sig[gpm]
    gpm = gpm[gpm]

    # Mask further?
    if masks is not None:
        gdp = np.ones_like(wave, dtype=bool)
        for mask in masks:
            bad = (wave > mask[0]) & (wave < mask[1])
            gdp[bad] = False
        # Cut again
        wave = wave[gdp]
        flam = flam[gdp]

    # Grab the instrument response function
    fwave, trans = load_filter(filter)

    ## This is a hack, need to remove this
    if filter=='VISTA-K':
        fwave, trans = fwave[(fwave>18000.)&(fwave<25000.)], trans[(fwave>18000.)&(fwave<25000.)]
    tfunc = interpolate.interp1d(fwave, trans, bounds_error=False, fill_value=0.)

    # Convolve
    allt = tfunc(wave)
    wflam = np.sum(flam*allt) / np.sum(allt)* units.erg/units.s/units.cm**2/units.AA
    wflam_err = np.sqrt(np.sum((sig*allt)**2)) / np.sum(allt)* units.erg/units.s/units.cm**2/units.AA

    mean_wv = np.sum(fwave*trans)/np.sum(trans)* units.AA

    # Convert flam to fnu
    fnu = wflam * mean_wv ** 2 / constants.c
    fnu_err = wflam_err * mean_wv ** 2 / constants.c

    return fnu.to('erg/s/cm**2/Hz').value, fnu_err.to('erg/s/cm**2/Hz').value

def spec2mag(filter, wave, flam, sig, gpm=None, masks=None, mag_type='AB', verbose=True):
    """
    Calculate magnitude in a given filter with the given spectra

    Args:
        filter (str): name of filter
        wave (array): wavelength in units of A
        flam (array): flux in units of erg/s/cm2/A
        sig (array): flux error in units of erg/s/cm2/A
        gpm (bool array): good pixel masks for your spectrum
        masks (list, optional): Wavelength ranges to mask in calculation

    Returns:
        Magnitude in the given filter

    """
    ### convert from spec to flux in the given filter
    fnu, fnu_err = spec2flux(filter, wave, flam, sig, gpm=gpm, masks=masks)

    # Apparent AB
    MAB0 = -2.5 * np.log10(3631.e-23)
    ABmag = -2.5 * np.log10(fnu) - MAB0
    mag_err = 2.5/np.log(10)*(fnu_err)/fnu
    if verbose:
        msgs.info("The magnitude of your spectrum in filter {:} is {:} AB mag".format(filter, ABmag))

    if mag_type == 'AB':
        mag =  ABmag
    elif mag_type == 'VEGA':
        if filter in vega_2_AB.keys():
            mag = ABmag - vega_2_AB[filter]
            if verbose:
                msgs.info("The magnitude of your spectrum in filter {:} is {:} VEGA mag".format(filter, mag))
        else:
            msgs.warn('VEGA to AB conversion factor was not found. Returning AB magnitude!')
    else:
        msgs.error('The supported system are AB and VEGA.')

    return mag, mag_err

def scale_in_filter(scale_dict, wave, flam, sig, gpm=None,verbose=True):
    """
    Scale spectra to input magnitude in given filter

    scale_dict has data model:
      'filter' (str): name of filter
      'mag' (float): magnitude
      'mag_type' (str, optional): type of magnitude.  Assumed 'AB'
      'masks' (list, optional): Wavelength ranges to mask in calculation

    Args:
        scale_dict (dict):
        wave (array): wavelength in units of A
        flam (array): flux in units of erg/s/cm2/A
        sig (array): flux error in units of erg/s/cm2/A
        gpm (bool array): good pixel masks for your spectrum


    Returns:
        linetools.spectra.xspectrum1d.XSpectrum1D, float:  Scaled spectrum


    """
    # parse the masks
    if ('masks' in scale_dict) and (scale_dict['masks'] is not None):
        masks = scale_dict['masks']
    else:
        masks = None

    # parse the magnitude system
    if ('mag_type' in scale_dict) and (scale_dict['mag_type'] is not None):
        mag_type = scale_dict['mag_type']
    else:
        mag_type = 'AB'

    specmag,_ = spec2mag(scale_dict['filter'], wave, flam, sig, gpm=gpm, masks=masks, mag_type=mag_type,verbose=verbose)

    # Scale factor
    Dm = specmag - scale_dict['mag']
    scale = 10**(Dm/2.5)
    if verbose:
        msgs.info("Scaling spectrum by {}".format(scale))

    # Scale spectrum
    flam_new, sig_new = flam*scale, sig*scale

    return scale, flam_new, sig_new