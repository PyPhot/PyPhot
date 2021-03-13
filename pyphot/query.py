from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs

from pyphot import msgs

def query_region(ra, dec, radius='0.1', catalog='Panstarrs', data_release='dr2'):
    '''

    :param ra: in units of degree
    :param dec: in units of degree
    :param radius: in units of degree
    :param catalog:
    :param data_release:
    :return:
    '''

    vizier_cat = {'SDSS':'V/147', 'Panstarrs':'II/349', 'Twomass':'II/246', 'UKIDSS':'II/319', 'VHS':'II/367',
                  'VIKING':'II/343'}

    mast_cat = {'Gaia':'Gaia'} #'Panstarrs':'Panstarrs',

    if catalog in mast_cat.keys():
        result = Catalogs.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg,
                                       catalog=mast_cat[catalog], data_release=data_release)
    elif catalog in vizier_cat.keys():
        Vizier.ROW_LIMIT = -1
        result = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=radius*u.deg,
                                     catalog=vizier_cat[catalog])
        if len(result)>0:
            result = result[0]
        else:
            result = None
    else:
        msgs.warn('No catalog was found!')
        result = None

    return result