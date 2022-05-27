
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
