import os
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.table import Table

from pyphot import msgs, utils
from pyphot.query import ps1query, ldac

def query_datalab(ra, dec, radius=0.1, catalog='Legacy', data_release='ls_dr9'):

    try:
        from dl import queryClient as qc
    except:
        msgs.error('Please install astro-datalab with: pip install --ignore-installed --no-cache-dir astro-datalab')
        from dl import queryClient as qc

    delta = radius*np.sqrt(2) # I query a box region first and the select according to distance.
    ra1, ra2, dec1, dec2 = ra-delta, ra+delta, dec-delta, dec+delta
    if catalog=='Legacy':
        msgs.info('Using astro-datalab for downloading Legacy Imaging Survey catalog.')
        result = qc.query(sql='SELECT * from {:}.tractor where '
                              'ra>{:} and ra<{:} and '
                              'dec>{:} and dec<{:}'.format(data_release, ra1, ra2, dec1, dec2), fmt='table')
        data = Table()
        data['RA'] = result['ra']
        data['DEC'] = result['dec']
        data['RA_ERR'] = utils.inverse(np.sqrt(result['ra_ivar']))
        data['DEC_ERR'] = utils.inverse(np.sqrt(result['dec_ivar']))
        data['THETA_ERR'] = np.zeros_like(result['ra'])
        data['PRIMARY'] = result['brick_primary'].astype(bool) ## primary is True
        data['POINT_SOURCE'] = result['type'] == 'PSF' ## point source is True
        data['g_MAG'] = 22.5-2.5*np.log10(result['flux_g'])
        data['g_MAG_ERR'] = 2.5/np.log(10)*(np.sqrt(1/result['flux_ivar_g']))/result['flux_g']
        data['r_MAG'] = 22.5-2.5*np.log10(result['flux_r'])
        data['r_MAG_ERR'] = 2.5/np.log(10)*(np.sqrt(1/result['flux_ivar_r']))/result['flux_r']
        data['z_MAG'] = 22.5-2.5*np.log10(result['flux_z'])
        data['z_MAG_ERR'] = 2.5/np.log(10)*(np.sqrt(1/result['flux_ivar_z']))/result['flux_z']
        data['DIST'] = np.sqrt(((result['ra']-ra)*np.cos(dec/180.*np.pi))**2 + (result['dec']-dec)**2)
        data['maskbits'] = result['maskbits']
        sel = data['DIST']<radius
        final = data[sel]

    return final

def query_region(ra, dec, radius=0.1, catalog='Panstarrs', data_release='dr2', table='mean'):
    '''

    :param ra: in units of degree
    :param dec: in units of degree
    :param radius: in units of degree
    :param catalog:
    :param data_release:
    :return:
    '''

    vizier_cat = {'SDSS':'V/147', 'TwoMass':'II/246', 'UKIDSS':'II/319', 'VHS':'II/367',
                  'VIKING':'II/343','Panstarrs':'II/349', 'DES':'II/371'}

    mast_cat = {'Gaia':'Gaia'}

    datalab_cat = {'Legacy':'Legacy'}

    msgs.info('Loading catalogs from {:} survey'.format(catalog))
    if catalog in mast_cat.keys():
        ## MAST has a query limit of 500,000 rows. See https://mast.stsci.edu/api/v0/
        result = Catalogs.query_region(SkyCoord(ra=round(ra,6), dec=round(dec,6), unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg,
                                       catalog=mast_cat[catalog], data_release=data_release)
    elif catalog in vizier_cat.keys():
        #ToDo: a bug in SkyCoord. The following does not work ra=106.46233492244818, dec=29.100575319582383
        # I round it to 7 digits and then works!!!
        v = Vizier(columns=["**", "+_r"], catalog=vizier_cat[catalog])
        v.ROW_LIMIT = -1
        result = v.query_region(SkyCoord(ra=round(ra,6), dec=round(dec,6), unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg)

        if len(result)>0:
            result = result[0]
        else:
            result = None

        if result is None and catalog=='Panstarrs':
            msgs.info('Querying Panstarrs failed from VIZIER, using the PS1 query which is slower but more stable.')
            result = ps1query.conesearch(ra, dec, radius, table=table, release="dr2", format="csv", pagesize=-1)
    elif catalog in datalab_cat.keys():
        result = query_datalab(ra, dec, radius=radius, catalog=catalog, data_release=data_release)
    else:
        msgs.warn('No catalog was found!')
        result = None

    return result

def query_standard(ra, dec, radius=0.1, catalog='Panstarrs', data_release='dr2'):

    result = query_region(ra, dec, radius=radius, catalog=catalog, data_release=data_release)

    if result is not None:
        data = Table()
        if catalog == 'SDSS': # SDSS DR12
            msgs.info('Selecting pointing sources from SDSS DR12.')
            data['RA'] = result['RA_ICRS']
            data['DEC'] = result['DE_ICRS']
            data['RA_ERR'] = np.zeros_like(data['RA'])
            data['DEC_ERR'] = np.zeros_like(data['RA'])
            data['THETA_ERR'] = np.zeros_like(data['RA'])
            data['PRIMARY'] = (result['mode']==1).data ## primary is True
            data['POINT_SOURCE'] = (result['class'] == 6).data ## point source is True
            data['u_MAG'] = result['upmag'] - 0.04
            data['u_MAG_ERR'] = result['e_upmag']
            data['g_MAG'] = result['gpmag']
            data['g_MAG_ERR'] = result['e_gpmag']
            data['r_MAG'] = result['rpmag']
            data['r_MAG_ERR'] = result['e_rpmag']
            data['i_MAG'] = result['ipmag']
            data['i_MAG_ERR'] = result['e_ipmag']
            data['z_MAG'] = result['zpmag'] + 0.02
            data['z_MAG_ERR'] = result['e_zpmag']
            data['DIST'] = result['_r'] # in units of degree
        elif catalog == 'DES': # DES DR2:
            msgs.info('Selecting pointing sources from DES DR2.')
            data['RA'] = result['RA_ICRS']
            data['DEC'] = result['DE_ICRS']
            data['RA_ERR'] = np.zeros_like(data['RA'])
            data['DEC_ERR'] = np.zeros_like(data['RA'])
            data['THETA_ERR'] = np.zeros_like(data['RA'])
            data['PRIMARY'] = (result['mode']==1).data ## primary is True
            data['POINT_SOURCE'] = (result['class'] == 6).data ## point source is True
            data['g_MAG'] = result['gmagPSF']
            data['g_MAG_ERR'] = result['e_gmagPSF']
            data['r_MAG'] = result['rmagPSF']
            data['r_MAG_ERR'] = result['e_rmagPSF']
            data['i_MAG'] = result['imagPSF']
            data['i_MAG_ERR'] = result['e_imagPSF']
            data['z_MAG'] = result['zmagPSF']
            data['z_MAG_ERR'] = result['e_zmagPSF']
            data['Y_MAG'] = result['YmagPSF']
            data['Y_MAG_ERR'] = result['e_YmagPSF']
            data['DIST'] = result['_r'] # in units of degree
        elif catalog == 'TwoMass':
            msgs.info('Selecting pointing sources from TwoMass.')
            data['RA'] = result['RAJ2000']
            data['DEC'] = result['DEJ2000']
            data['RA_ERR'] = np.zeros_like(data['RA'])
            data['DEC_ERR'] = np.zeros_like(data['RA'])
            data['THETA_ERR'] = np.zeros_like(data['RA'])
            data['PRIMARY'] = True
            data['POINT_SOURCE'] = True
            data['J_MAG'] = result['Jmag'] + 0.89
            data['J_MAG_ERR'] = result['Jcmsig']
            data['H_MAG'] = result['Hmag'] + 1.37
            data['H_MAG_ERR'] = result['Hcmsig']
            data['K_MAG'] = result['Kmag'] + 1.84
            data['K_MAG_ERR'] = result['Kcmsig']
            data['DIST'] = result['_r'] # in units of degree
        elif catalog == 'UKIDSS': # UKIDSS LAS DR9
            msgs.info('Selecting pointing sources from UKIDSS DR9.')
            data['RA'] = result['RAJ2000']
            data['DEC'] = result['DEJ2000']
            data['RA_ERR'] = np.zeros_like(data['RA'])
            data['DEC_ERR'] = np.zeros_like(data['RA'])
            data['THETA_ERR'] = np.zeros_like(data['RA'])
            primary = (result['Yflags']<1000) & (result['Jflags1']<1000) & (result['Hflags']<1000) & (result['Kflags']<1000)
            data['PRIMARY'] = primary.data
            data['POINT_SOURCE'] = ((result['cl']<0) & (result['cl']>-3)).data ## -1 or -2
            data['Y_MAG'] = result['Ymag'] + 0.634
            data['Y_MAG_ERR'] = result['e_Ymag']
            data['J_MAG'] = result['Jmag1'] + 0.938
            data['J_MAG_ERR'] = result['e_Jmag1']
            data['H_MAG'] = result['Hmag'] + 1.379
            data['H_MAG_ERR'] = result['e_Hmag']
            data['K_MAG'] = result['Kmag'] + 1.900
            data['K_MAG_ERR'] = result['e_Kmag']
            data['DIST'] = result['_r'] # in units of degree
        elif catalog == 'Panstarrs':  # PS1 DR2 from MAST
            msgs.info('Selecting pointing sources from Panstarrs.')
            try:
                # if data queryed from ps1query
                data['RA'] = result['raMean']
                data['DEC'] = result['decMean']
                data['RA_ERR'] = np.zeros_like(data['RA'])
                data['DEC_ERR'] = np.zeros_like(data['RA'])
                data['THETA_ERR'] = np.zeros_like(data['RA'])
                data['PRIMARY'] = True
                point = (abs(result['gMeanPSFMag'] - result['gMeanKronMag']) < 0.5) & (abs(result['rMeanPSFMag'] - result['rMeanKronMag']) < 0.3) & \
                        (abs(result['iMeanPSFMag'] - result['iMeanKronMag']) < 0.3) & (abs(result['zMeanPSFMag'] - result['zMeanKronMag']) < 0.5) & \
                        (abs(result['yMeanPSFMag'] - result['yMeanKronMag']) < 0.5) & (result['gMeanPSFMag']>0) & (result['rMeanPSFMag']>0) & \
                        (result['iMeanPSFMag'] > 0) & (result['zMeanPSFMag']>0) & (result['yMeanPSFMag']>0)
                data['POINT_SOURCE'] = point.data
                data['g_MAG'] = result['gMeanPSFMag']
                data['g_MAG_ERR'] = result['gMeanPSFMagErr']
                data['r_MAG'] = result['rMeanPSFMag']
                data['r_MAG_ERR'] = result['rMeanPSFMagErr']
                data['i_MAG'] = result['iMeanPSFMag']
                data['i_MAG_ERR'] = result['iMeanPSFMagErr']
                data['z_MAG'] = result['zMeanPSFMag']
                data['z_MAG_ERR'] = result['zMeanPSFMagErr']
                data['y_MAG'] = result['yMeanPSFMag']
                data['y_MAG_ERR'] = result['yMeanPSFMagErr']
                data['DIST'] = result['distance']  # in units of degree
            except:
                # if data queryed from VIZIER
                data['RA'] = result['RAJ2000']
                data['DEC'] = result['DEJ2000']
                data['RA_ERR'] = np.zeros_like(data['RA'])
                data['DEC_ERR'] = np.zeros_like(data['RA'])
                data['THETA_ERR'] = np.zeros_like(data['RA'])
                data['PRIMARY'] =True
                point = (abs(result['gmag']-result['gKmag'])<0.3) & (abs(result['rmag']-result['rKmag'])<0.3) & \
                        (abs(result['imag'] - result['iKmag']) < 0.3) & (abs(result['zmag']-result['zKmag'])<0.3) & \
                        (abs(result['ymag'] - result['yKmag']) < 0.3)
                data['POINT_SOURCE'] = point.data
                data['g_MAG'] = result['gmag']
                data['g_MAG_ERR'] = result['e_gmag']
                data['r_MAG'] = result['rmag']
                data['r_MAG_ERR'] = result['e_rmag']
                data['i_MAG'] = result['imag']
                data['i_MAG_ERR'] = result['e_imag']
                data['z_MAG'] = result['zmag']
                data['z_MAG_ERR'] = result['e_zmag']
                data['y_MAG'] = result['ymag']
                data['y_MAG_ERR'] = result['e_ymag']
                data['DIST'] = result['_r'] # in units of degree
        elif catalog == 'Legacy':  # Legacy data from datalab
            msgs.info('Selecting pointing sources from Legacy survey {:}.'.format(data_release))
            # Reject objects in masks.
            # BRIGHT BAILOUT GALAXY CLUSTER (1, 10, 12, 13) bits not set.
            maskbits = result['maskbits'].data
            sel  = result['PRIMARY'].data & result['POINT_SOURCE'].data
            sel &= result['g_MAG_ERR']<5.0/1.0857
            sel &= result['r_MAG_ERR']<5.0/1.0857
            sel &= result['z_MAG_ERR']<5.0/1.0857
            for bit in [1, 10, 12, 13]:
                sel &= ((maskbits & 2 ** bit) == 0)
            data = result[sel]
        else:
            msgs.error('TBD')
        ## make the selection
        good = data['PRIMARY'] & data['POINT_SOURCE']
        msgs.info('Identified {:} good point sources within {:} degree'.format(np.sum(good), radius))
        msgs.info('All the returned magnitudes are in AB system.')
        return data[good]
    else:
        msgs.warn('Failed to query catalog from {:}'.format(catalog))
        return None

'''
def get_tbl_for_scamp(outname, ra, dec, radius=0.1, catalog='LS-DR9', reuse_master=True):

    if os.path.exists(outname) and reuse_master:
        tbl = Table.read(outname, hdu=2)
    else:
        if catalog=='LS-DR9':
            tbl = query_standard(ra, dec, radius=radius, catalog='Legacy', data_release='ls_dr9')
        elif catalog=='DES-DR2':
            tbl = query_standard(ra, dec, radius=0.1, catalog='DES')

        hdulist = ldac.convert_table_to_ldac(tbl)
        hdulist.writeto(outname, overwrite=True)

    return tbl
'''
