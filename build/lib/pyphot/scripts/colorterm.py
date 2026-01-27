import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pkg_resources import resource_filename

from pyphot import msgs
from pyphot import utils, spec_mag


def func(x, a, b):
    return b * x + a


def colorterm(filter, primary, secondary, show=True, path=None):


    if path is None:
        # Path for the XSL DR2 spectra which can be downloaded from the following link.
        # http://xsl.u-strasbg.fr/page_dr2.html
        path = resource_filename('pyphot', os.path.join('data', 'all_dr2_fits'))

    nstar = 911
    #nstar = 50
    mag = np.ones(nstar)
    mag1 = np.ones(nstar)
    mag2 = np.ones(nstar)

    for istar in range(nstar):

        msgs.info('Calculating {:}/{:} star'.format(istar+1,nstar))
        # Define the spectrum
        xsl_short = 'X' + '{:04d}'.format(istar + 1)

        # Define the full frame
        frame_uvb = os.path.join(path,'xsl_spectrum_' + xsl_short + '_uvb.fits')
        frame_vis = os.path.join(path,'xsl_spectrum_' + xsl_short + '_vis.fits')
        frame_nir = os.path.join(path,'xsl_spectrum_' + xsl_short + '_nir.fits')
        if os.path.exists(frame_uvb):
            par = fits.open(frame_uvb)
            flux_uvb = par[1].data['FLUX']
            wave_uvb = par[1].data['WAVE']*10.0
            wave_uvb, flux_uvb =  wave_uvb[wave_uvb<5500.], flux_uvb[wave_uvb<5500.]
        else:
            wave_uvb, flux_uvb = np.zeros(1),np.zeros(1)
        if os.path.exists(frame_vis):
            par = fits.open(frame_vis)
            flux_vis = par[1].data['FLUX']
            wave_vis = par[1].data['WAVE']*10.0
            wave_vis, flux_vis =  wave_vis[(wave_vis>5500.)&(wave_vis<1e4)], flux_vis[(wave_vis>5500.)&(wave_vis<1e4)]
        else:
            wave_vis, flux_vis = np.zeros(1),np.zeros(1)
        if os.path.exists(frame_nir):
            par = fits.open(frame_nir)
            flux_nir = par[1].data['FLUX']
            wave_nir = par[1].data['WAVE']*10.0
            wave_nir, flux_nir =  wave_nir[wave_nir>1e4], flux_nir[wave_nir>1e4]
        else:
            wave_nir, flux_nir = np.zeros(1),np.zeros(1)
        wave = np.hstack([wave_uvb,wave_vis,wave_nir])
        flux = np.hstack([flux_uvb,flux_vis,flux_nir])
        good_pix = (wave>0) & np.invert(np.isnan(flux))
        wave, flux =  wave[good_pix], flux[good_pix]
        sig = np.ones_like(flux)

        mag[istar], _ = spec_mag.spec2mag(filter, wave, flux, sig, mag_type='AB')
        mag1[istar], _ = spec_mag.spec2mag(primary, wave, flux, sig, mag_type='AB')
        mag2[istar], _ = spec_mag.spec2mag(secondary, wave, flux, sig, mag_type='AB')

    bad = np.isnan(mag) + np.isnan(mag1) + np.isnan(mag2)

    if filter == 'IMACSF2-NB919':
        bad += (mag-mag1<-0.2) + (mag-mag1>0.1)

    xx = mag1[np.invert(bad)]-mag2[np.invert(bad)]
    yy = mag[np.invert(bad)]-mag1[np.invert(bad)]
    popt, pcov = utils.robust_curve_fit(func, xx, yy, niters=5, sigclip=3, maxiters_sigclip=5)

    msgs.info('The color-term coefficient is {:}={:} + ({:0.3f}*({:}-{:})) + ({:0.3f})'.format(filter, primary, popt[1],primary,secondary, popt[0]))

    if show:
        plt.plot(xx, yy,'k.')
        plt.plot(xx, func(xx,popt[0],popt[1]))
        plt.show()
    from IPython import embed
    embed()

def parse_args(options=None, return_parser=False):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filter', type = str, default = None, help = 'Your filter name')
    parser.add_argument('primary', type = str, default = None, help = 'The primary filter for the calibration')
    parser.add_argument('secondary', type = str, default = None, help = 'The secondary filter for the calibration')
    parser.add_argument('--path', type=str, default = None, help="The path of the ESO XSL DR2 database")
    parser.add_argument('--show', type=bool, default = True, help="Show the difference image with ds9?")

    if return_parser:
        return parser

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    colorterm(args.filter, args.primary, args.secondary, show=args.show, path=args.path)

'''
Some pre-calculated coefficients:

## UKIRT vs 2MASS
pyphot_colorterm UKIRT-Y TMASS-J TMASS-H --path /Volumes/Work/Imaging/all_dr2_fits
 ==> UKIRT-Y=TMASS-J + (0.694*(TMASS-J-TMASS-H)) + (0.003)

pyphot_colorterm UKIRT-J TMASS-J TMASS-H --path /Volumes/Work/Imaging/all_dr2_fits
 ==> UKIRT-J=TMASS-J + (-0.059*(TMASS-J-TMASS-H)) + (-0.001)
 this is fully consistent with that derived by the WIRwolf image stacking pipeline:
    https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/wirwolf/docs/filt.html

pyphot_colorterm UKIRT-H TMASS-H TMASS-J --path /Volumes/Work/Imaging/all_dr2_fits
 ==> UKIRT-H=TMASS-H + (-0.035*(TMASS-H-TMASS-J)) + (-0.007)
 this is fully consistent with that derived by the WIRwolf image stacking pipeline:
    https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/wirwolf/docs/filt.html

pyphot_colorterm UKIRT-K TMASS-K TMASS-H --path /Volumes/Work/Imaging/all_dr2_fits
 ==> UKIRT-K=TMASS-K + (0.017*(TMASS-K-TMASS-H)) + (0.050)

## VISTA Ks vs 2MASS
pyphot_colorterm VISTA-K TMASS-K TMASS-H --path /Volumes/Work/Imaging/all_dr2_fits
 ==> VISTA-K=TMASS-K + (-0.031*(TMASS-K-TMASS-H)) + (0.023)

## SDSS vs PanStarrs
pyphot_colorterm SDSS-G PS1-G PS1-R --path /Volumes/Work/Imaging/all_dr2_fits
 ==> SDSS-G=PS1-G + (0.160*(PS1-G-PS1-R)) + (0.016)
pyphot_colorterm SDSS-R PS1-R PS1-I --path /Volumes/Work/Imaging/all_dr2_fits
 ==> SDSS-R=PS1-R + (0.024*(PS1-R-PS1-I)) + (0.002) #ToDo: Need second order for this!!!
pyphot_colorterm SDSS-I PS1-I PS1-Z --path /Volumes/Work/Imaging/all_dr2_fits
 ==> SDSS-I=PS1-I + (0.058*(PS1-I-PS1-Z)) + (-0.000)
pyphot_colorterm SDSS-Z PS1-Z PS1-Y --path /Volumes/Work/Imaging/all_dr2_fits
 ==> SDSS-Z=PS1-Z + (-0.258*(PS1-Z-PS1-Y)) + (-0.011) # seems better than using Z-I
pyphot_colorterm SDSS-Z PS1-Z PS1-I --path /Volumes/Work/Imaging/all_dr2_fits
 ==> SDSS-Z=PS1-Z + (0.120*(PS1-Z-PS1-I)) + (-0.018)
 
## IMACS NB919 VS PanStarrs
pyphot_colorterm IMACSF2-NB919 PS1-Z PS1-Y --path /Volumes/Work/Imaging/all_dr2_fits
 ==> IMACSF2-NB919=PS1-Z + (-0.618*(PS1-Z-PS1-Y)) + (0.015)

## LBT
pyphot_colorterm LBC-BESS_R PS1-R PS1-I --path /Volumes/Work/Imaging/all_dr2_fits
 ==> LBC-BESS_R=PS1-R + (-0.218*(PS1-R-PS1-I)) + (-0.010) # ToDo: need second order
pyphot_colorterm LBC-BESS_I PS1-I PS1-Z --path /Volumes/Work/Imaging/all_dr2_fits
 ==> LBC-BESS_I=PS1-I + (-0.411*(PS1-I-PS1-Z)) + (-0.003)
'''

'''
Subaru/HSC color term coefficients
config.data = {
    "hsc*": ColortermDict(data={
        'g': Colorterm(primary="g", secondary="g"),
        'r': Colorterm(primary="r", secondary="r"),
        'i': Colorterm(primary="i", secondary="i"),
        'z': Colorterm(primary="z", secondary="z"),
        'y': Colorterm(primary="y", secondary="y"),
    }),
    "sdss*": ColortermDict(data={
        'g': Colorterm(primary="g", secondary="r", c0=-0.00816446, c1=-0.08366937, c2=-0.00726883),
        'r': Colorterm(primary="r", secondary="i", c0=0.00231810, c1=0.01284177, c2=-0.03068248),
        'r2': Colorterm(primary="r", secondary="i", c0=0.00074087, c1=-0.00830543, c2=-0.02848420),
        'i': Colorterm(primary="i", secondary="z", c0=0.00130204, c1=-0.16922042, c2=-0.01374245),
        'i2': Colorterm(primary="i", secondary="z", c0=0.00124676, c1=-0.20739606, c2=-0.01067212),
        'z': Colorterm(primary="z", secondary="i", c0=-0.00680620, c1=0.01353969, c2=0.01479369),
        'y': Colorterm(primary="z", secondary="i", c0=0.01739708, c1=0.35652971, c2=0.00574408),
        'N816': Colorterm(primary="i", secondary="z", c0=0.00927133, c1=-0.63558358, c2=-0.05474862),
        'N921': Colorterm(primary="z", secondary="i", c0=0.00752972, c1=0.09863530, c2=-0.05451118),
        'N926': Colorterm(primary="z", secondary="i",c0=0.009369, c1=0.130261, c2=-0.119282),
    }),
    "ps1*": ColortermDict(data={
        'g': Colorterm(primary="g", secondary="r", c0=0.005905, c1=0.063651, c2=-0.000716),
        'r': Colorterm(primary="r", secondary="i", c0=-0.000294, c1=-0.005458, c2=-0.009451),
        'r2': Colorterm(primary="r", secondary="i", c0=0.000118, c1=-0.002790, c2=-0.014363),
        'i': Colorterm(primary="i", secondary="z", c0=0.000979, c1=-0.154608, c2=-0.010429),
        'i2': Colorterm(primary="i", secondary="z", c0=0.001653, c1=-0.206313, c2=-0.016085),
        'z': Colorterm(primary="z", secondary="y", c0=-0.005585, c1=-0.220704, c2=-0.298211),
        'y': Colorterm(primary="y", secondary="z", c0=-0.001952, c1=0.199570, c2=0.216821),
        'I945': Colorterm(primary="y", secondary="z", c0=0.005275, c1=-0.194285, c2=-0.125424),
        'N387': Colorterm(primary="g", secondary="r", c0=0.427879, c1=1.869068, c2=0.540580),
        'N468': Colorterm(primary="g", secondary="r", c0=-0.042240, c1=0.121756, c2=0.027599),
        'N515': Colorterm(primary="g", secondary="r", c0=-0.021913, c1=-0.253159, c2=0.151553),
        'N527': Colorterm(primary="g", secondary="r", c0=-0.020641, c1=-0.366167, c2=0.038497),
        'N656': Colorterm(primary="r", secondary="i", c0=0.035658, c1=-0.512071, c2=0.042824),
        'N718': Colorterm(primary="i", secondary="r", c0=-0.016294, c1=-0.233139, c2=0.252505),
        'N816': Colorterm(primary="i", secondary="z", c0=0.013806, c1=-0.717681, c2=0.049289),
        'N921': Colorterm(primary="z", secondary="y", c0=0.002039, c1=-0.477412, c2=-0.492151),
        'N926': Colorterm(primary="z", secondary="y",c0=0.005230, c1=-0.574448, c2=-0.330899),
        'N973': Colorterm(primary="y", secondary="z", c0=-0.007775, c1=-0.050972, c2=-0.197278),
    }),
}
'''