import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astropy.table import Table

from pyphot import msgs

def query_region(ra, dec, radius=0.1, catalog='Panstarrs', data_release='dr2'):
    '''

    :param ra: in units of degree
    :param dec: in units of degree
    :param radius: in units of degree
    :param catalog:
    :param data_release:
    :return:
    '''

    vizier_cat = {'SDSS':'V/147', 'Panstarrs':'II/349', 'TwoMass':'II/246', 'UKIDSS':'II/319', 'VHS':'II/367',
                  'VIKING':'II/343'}

    mast_cat = {'Gaia':'Gaia'}

    msgs.info('Loading catalogs from {:}'.format(catalog))
    if catalog in mast_cat.keys():
        result = Catalogs.query_region(SkyCoord(ra=round(ra,7), dec=round(dec,7), unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg,
                                       catalog=mast_cat[catalog], data_release=data_release)
    elif catalog in vizier_cat.keys():
        #ToDo: a bug in SkyCoord. The following does not work ra=106.46233492244818, dec=29.100575319582383
        # I round it to 7 digits and then works!!!
        v = Vizier(columns=["**", "+_r"], catalog=vizier_cat[catalog])
        v.ROW_LIMIT = -1
        result = v.query_region(SkyCoord(ra=round(ra,7), dec=round(dec,7), unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg)

        if len(result)>0:
            result = result[0]
        else:
            result = None
    else:
        msgs.warn('No catalog was found!')
        result = None

    return result

def query_standard(ra, dec, radius=0.1, catalog='Panstarrs', data_release='dr2'):

    result = query_region(ra, dec, radius=radius, catalog=catalog, data_release=data_release)

    data = Table()
    if catalog == 'SDSS': # SDSS DR12
        msgs.info('Selecting pointing sources in SDSS DR12 for photometric calibrations.')
        data['RA'] = result['RA_ICRS']
        data['DEC'] = result['DE_ICRS']
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
    elif catalog == 'TwoMass':
        msgs.info('Selecting pointing sources in TwoMass for photometric calibrations.')
        data['RA'] = result['RAJ2000']
        data['DEC'] = result['DEJ2000']
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
        msgs.info('Selecting pointing sources in UKIDSS DR9 for photometric calibrations.')
        data['RA'] = result['RAJ2000']
        data['DEC'] = result['DEJ2000']
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
    elif catalog == 'Panstarrs': # SDSS DR12
        msgs.info('Selecting pointing sources in Panstarrs DR1 for photometric calibrations.')
        data['RA'] = result['RAJ2000']
        data['DEC'] = result['DEJ2000']
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
    else:
        msgs.error('TBD')
    ## make the selection
    good = data['PRIMARY'] & data['POINT_SOURCE']
    msgs.info('Identified {:} good point sources within {:} degree'.format(np.sum(good), radius))
    msgs.info('All the returned magnitudes are in AB system.')

    return data[good]